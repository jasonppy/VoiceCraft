# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import stat
import time
import warnings
import random
import getpass
import shutil
import subprocess
import torch
import numpy as np
import torchaudio

from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import get_tokenizer
from cog import BasePredictor, Input, Path, BaseModel

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["USER"] = getpass.getuser()

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from models import voicecraft
from inference_tts_scale import inference_one_sample
from edit_utils import get_span
from inference_speech_editing_scale import get_mask_interval
from inference_speech_editing_scale import (
    inference_one_sample as inference_one_sample_editing,
)

ENV_NAME = "myenv"

MODEL_URL = "https://weights.replicate.delivery/default/pyp1/VoiceCraft.tar"
MODEL_CACHE = "model_cache"


class ModelOutput(BaseModel):
    whisper_transcript_orig_audio: str
    generated_audio: Path


class WhisperModel:
    def __init__(self, model_cache, model_name="base.en", device="cuda"):

        with open(f"{model_cache}/{model_name}.pt", "rb") as fp:
            checkpoint = torch.load(fp, map_location="cpu")
            dims = ModelDimensions(**checkpoint["dims"])
            self.model = Whisper(dims)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(device)

        tokenizer = get_tokenizer(multilingual=False)
        self.supress_tokens = [-1] + [
            i
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
        ]

    def transcribe(self, audio_path):
        return self.model.transcribe(
            audio_path, suppress_tokens=self.supress_tokens, word_timestamps=True
        )["segments"]


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        encodec_fn = f"{MODEL_CACHE}/encodec_4cb2048_giga.th"
        self.models, self.ckpt, self.phn2num = {}, {}, {}
        for voicecraft_name in [
            "giga830M.pth",
            "giga330M.pth",
            "gigaHalfLibri330M_TTSEnhanced_max16s.pth",
        ]:
            ckpt_fn = f"{MODEL_CACHE}/{voicecraft_name}"

            self.ckpt[voicecraft_name] = torch.load(ckpt_fn, map_location="cpu")
            self.models[voicecraft_name] = voicecraft.VoiceCraft(
                self.ckpt[voicecraft_name]["config"]
            )
            self.models[voicecraft_name].load_state_dict(
                self.ckpt[voicecraft_name]["model"]
            )
            self.models[voicecraft_name].to(self.device)
            self.models[voicecraft_name].eval()

            self.phn2num[voicecraft_name] = self.ckpt[voicecraft_name]["phn2num"]

        self.text_tokenizer = TextTokenizer(backend="espeak")
        self.audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=self.device)
        self.transcribe_models = {
            k: WhisperModel(MODEL_CACHE, k, self.device)
            for k in ["base.en", "small.en", "medium.en"]
        }

    def predict(
        self,
        task: str = Input(
            description="Choose a task",
            choices=[
                "speech_editing-substitution",
                "speech_editing-insertion",
                "speech_editing-deletion",
                "zero-shot text-to-speech",
            ],
            default="zero-shot text-to-speech",
        ),
        voicecraft_model: str = Input(
            description="Choose a model",
            choices=["giga830M.pth", "giga330M.pth", "giga330M_TTSEnhanced.pth"],
            default="giga330M_TTSEnhanced.pth",
        ),
        orig_audio: Path = Input(
            description="Original audio file. WhisperX small.en model will be used for transcription"
        ),
        orig_transcript: str = Input(
            description="Optionally provide the transcript of the input audio. Leave it blank to use the whisper model below to generate the transcript. Inaccurate transcription may lead to error TTS or speech editing",
            default="",
        ),
        whisper_model: str = Input(
            description="If orig_transcript is not provided above, choose a Whisper model. Inaccurate transcription may lead to error TTS or speech editing. You can modify the generated transcript and provide it directly to ",
            choices=["base.en", "small.en", "medium.en"],
            default="base.en",
        ),
        target_transcript: str = Input(
            description="Transcript of the target audio file",
        ),
        cut_off_sec: float = Input(
            description="Only used for for zero-shot text-to-speech task. The first seconds of the original audio that are used for zero-shot text-to-speech. 3 sec of reference is generally enough for high quality voice cloning, but longer is generally better, try e.g. 3~6 sec",
            default=3.01,
        ),
        kvcache: int = Input(
            description="Set to 0 to use less VRAM, but with slower inference",
            default=1,
        ),
        left_margin: float = Input(
            description="Margin to the left of the editing segment",
            default=0.08,
        ),
        right_margin: float = Input(
            description="Margin to the right of the editing segment",
            default=0.08,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic. Do not recommend to change",
            default=1,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        stop_repetition: int = Input(
            default=-1,
            description=" -1 means do not adjust prob of silence tokens. if there are long silence or unnaturally stretched words, increase sample_batch_size to 2, 3 or even 4",
        ),
        sample_batch_size: int = Input(
            description="The higher the number, the faster the output will be. Under the hood, the model will generate this many samples and choose the shortest one",
            default=4,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        seed_everything(seed)

        segments = self.transcribe_models[whisper_model].transcribe(str(orig_audio))
        state = get_transcribe_state(segments)
        whisper_transcript = state["transcript"].strip()

        if len(orig_transcript.strip()) == 0:
            orig_transcript = whisper_transcript

        print(f"The transcript from the Whisper model: {whisper_transcript}")

        temp_folder = "exp_dir"
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

        os.makedirs(temp_folder)

        filename = "orig_audio"
        shutil.copy(orig_audio, f"{temp_folder}/{filename}.wav")

        with open(f"{temp_folder}/{filename}.txt", "w") as f:
            f.write(orig_transcript)

        # run MFA to get the alignment
        align_temp = f"{temp_folder}/mfa_alignments"

        command = f'/bin/bash -c "source /cog/miniconda/bin/activate && conda activate {ENV_NAME} && mfa align -v --clean -j 1 --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}"'
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Error:", e)
            raise RuntimeError("Error running Alignment")

        print("Alignment done!")

        align_fn = f"{align_temp}/{filename}.csv"
        audio_fn = f"{temp_folder}/{filename}.wav"
        info = torchaudio.info(audio_fn)
        audio_dur = info.num_frames / info.sample_rate

        # hyperparameters for inference
        codec_audio_sr = 16000
        codec_sr = 50
        top_k = 0
        silence_tokens = [1388, 1898, 131]

        if voicecraft_model == "giga330M_TTSEnhanced.pth":
            voicecraft_model = "gigaHalfLibri330M_TTSEnhanced_max16s.pth"

        if task == "zero-shot text-to-speech":
            assert (
                cut_off_sec < audio_dur
            ), f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
            prompt_end_frame = int(cut_off_sec * info.sample_rate)

            idx = find_closest_cut_off_word(state["word_bounds"], cut_off_sec)
            orig_transcript_until_cutoff_time = "".join(
                [word_bound["word"] for word_bound in state["word_bounds"][:idx]]
            )
        else:
            edit_type = task.split("-")[-1]
            orig_span, new_span = get_span(
                orig_transcript, target_transcript, edit_type
            )
            if orig_span[0] > orig_span[1]:
                RuntimeError(f"example {audio_fn} failed")
            if orig_span[0] == orig_span[1]:
                orig_span_save = [orig_span[0]]
            else:
                orig_span_save = orig_span
            if new_span[0] == new_span[1]:
                new_span_save = [new_span[0]]
            else:
                new_span_save = new_span
            orig_span_save = ",".join([str(item) for item in orig_span_save])
            new_span_save = ",".join([str(item) for item in new_span_save])
            start, end = get_mask_interval(align_fn, orig_span_save, edit_type)

            # span in codec frames
            morphed_span = (
                max(start - left_margin, 1 / codec_sr),
                min(end + right_margin, audio_dur),
            )  # in seconds
            mask_interval = [
                [round(morphed_span[0] * codec_sr), round(morphed_span[1] * codec_sr)]
            ]
            mask_interval = torch.LongTensor(mask_interval)  # [M,2], M==1 for now

        decode_config = {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "stop_repetition": stop_repetition,
            "kvcache": kvcache,
            "codec_audio_sr": codec_audio_sr,
            "codec_sr": codec_sr,
            "silence_tokens": silence_tokens,
        }

        if task == "zero-shot text-to-speech":
            decode_config["sample_batch_size"] = sample_batch_size
            _, gen_audio = inference_one_sample(
                self.models[voicecraft_model],
                self.ckpt[voicecraft_model]["config"],
                self.phn2num[voicecraft_model],
                self.text_tokenizer,
                self.audio_tokenizer,
                audio_fn,
                orig_transcript_until_cutoff_time.strip()
                + ""
                + target_transcript.strip(),
                self.device,
                decode_config,
                prompt_end_frame,
            )
        else:
            _, gen_audio = inference_one_sample_editing(
                self.models[voicecraft_model],
                self.ckpt[voicecraft_model]["config"],
                self.phn2num[voicecraft_model],
                self.text_tokenizer,
                self.audio_tokenizer,
                audio_fn,
                target_transcript,
                mask_interval,
                self.device,
                decode_config,
            )

        # save segments for comparison
        gen_audio = gen_audio[0].cpu()

        out = "/tmp/out.wav"
        torchaudio.save(out, gen_audio, codec_audio_sr)
        return ModelOutput(
            generated_audio=Path(out), whisper_transcript_orig_audio=whisper_transcript
        )


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_transcribe_state(segments):
    words_info = [word_info for segment in segments for word_info in segment["words"]]
    return {
        "transcript": " ".join([segment["text"].strip() for segment in segments]),
        "word_bounds": [
            {"word": word["word"], "start": word["start"], "end": word["end"]}
            for word in words_info
        ],
    }


def find_closest_cut_off_word(word_bounds, cut_off_sec):
    min_distance = float("inf")

    for i, word_bound in enumerate(word_bounds):
        distance = abs(word_bound["start"] - cut_off_sec)

        if distance < min_distance:
            min_distance = distance

        if word_bound["end"] > cut_off_sec:
            break

    return i
