# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import numpy as np
import random
import getpass
import torch
import torchaudio
import shutil
import subprocess
import sys

os.environ["USER"] = getpass.getuser()

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from cog import BasePredictor, Input, Path
from models import voicecraft
from inference_tts_scale import inference_one_sample

ENV_NAME = "myenv"
# sys.path.append(f"/cog/miniconda/envs/{ENV_NAME}/lib/python3.10/site-packages")

MODEL_URL = "https://weights.replicate.delivery/default/VoiceCraft.tar"
MODEL_CACHE = "model_cache"


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

        voicecraft_name = "giga830M.pth"  # or giga330M.pth

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        encodec_fn = f"{MODEL_CACHE}/encodec_4cb2048_giga.th"
        ckpt_fn = f"{MODEL_CACHE}/{voicecraft_name}"

        self.ckpt = torch.load(ckpt_fn, map_location="cpu")
        self.model = voicecraft.VoiceCraft(self.ckpt["config"])
        self.model.load_state_dict(self.ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        self.phn2num = self.ckpt["phn2num"]

        self.text_tokenizer = TextTokenizer(backend="espeak")
        self.audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=self.device)

    def predict(
        self,
        orig_audio: Path = Input(description="Original audio file"),
        orig_transcript: str = Input(
            description="Transcript of the original audio file. You can use models such as https://replicate.com/openai/whisper and https://replicate.com/vaibhavs10/incredibly-fast-whisper to get the transcript (and modify it if it's not accurate)",
        ),
        cut_off_sec: float = Input(
            description="The first seconds of the original audio that are used for zero-shot text-to-speech (TTS).  3 sec of reference is generally enough for high quality voice cloning, but longer is generally better, try e.g. 3~6 sec",
            default=3.01,
        ),
        orig_transcript_until_cutoff_time: str = Input(
            description="Transcript of the original audio file until the cut_off_sec specified above. This process will be improved and made automatically later",
        ),
        target_transcript: str = Input(
            description="Transcript of the target audio file",
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic,
            ge=0.01,
            le=5,
            default=1,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        sampling_rate: int = Input(
            description="Specify the sampling rate of the audio codec", default=16000
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        seed_everything(seed)

        temp_folder = "exp_temp"
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

        os.makedirs(temp_folder)
        os.system(f"cp {str(orig_audio)} {temp_folder}")
        # filename = os.path.splitext(orig_audio.split("/")[-1])[0]
        with open(f"{temp_folder}/orig_audio_file.txt", "w") as f:
            f.write(orig_transcript)

        # run MFA to get the alignment
        align_temp = f"{temp_folder}/mfa_alignments"

        command = f'/bin/bash -c "source /cog/miniconda/bin/activate && conda activate {ENV_NAME} && mfa align -v --clean -j 1 --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}"'
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Error:", e)
        print("Alignment done!")

        audio_fn = str(orig_audio)  # f"{temp_folder}/{filename}.wav"
        info = torchaudio.info(audio_fn)
        audio_dur = info.num_frames / info.sample_rate

        assert (
            cut_off_sec < audio_dur
        ), f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
        prompt_end_frame = int(cut_off_sec * info.sample_rate)

        codec_sr = 50
        top_k = 0
        silence_tokens = [1388, 1898, 131]
        kvcache = 1  # NOTE if OOM, change this to 0, or try the 330M model

        # NOTE adjust the below three arguments if the generation is not as good
        stop_repetition = 3  # NOTE if the model generate long silence, reduce the stop_repetition to 3, 2 or even 1
        sample_batch_size = 4  # NOTE: if the if there are long silence or unnaturally strecthed words, increase sample_batch_size to 5 or higher. What this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest. So if the speech rate of the generated is too fast change it to a smaller number.

        decode_config = {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "stop_repetition": stop_repetition,
            "kvcache": kvcache,
            "codec_audio_sr": sampling_rate,
            "codec_sr": codec_sr,
            "silence_tokens": silence_tokens,
            "sample_batch_size": sample_batch_size,
        }
        concated_audio, gen_audio = inference_one_sample(
            self.model,
            self.ckpt["config"],
            self.phn2num,
            self.text_tokenizer,
            self.audio_tokenizer,
            audio_fn,
            orig_transcript_until_cutoff_time.strip() + "" + target_transcript.strip(),
            self.device,
            decode_config,
            prompt_end_frame,
        )

        # save segments for comparison
        concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()

        out = "/tmp/out.wav"
        torchaudio.save(out, gen_audio, sampling_rate)
        torchaudio.save("out.wav", gen_audio, sampling_rate)
        return Path(out)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
