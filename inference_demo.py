"""
This script will allow you to run TTS inference with Voicecraft
Before getting started, be sure to follow the environment setup.
"""

from inference_tts_scale import inference_one_sample
from models import voicecraft
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from IPython.display import display, Audio
import argparse
import random
import numpy as np
import torchaudio
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "me"  # TODO change this to your username

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="VoiceCraft Inference: see the script for more information on the options")

    parser.add_argument("--model_name", type=str, default="giga330M.pth", choices=[
                        "giga330M.pth", "gigaHalfLibri330M_TTSEnhanced_max16s.pth", "giga830M.pth"],
                        help="VoiceCraft model to use")
    parser.add_argument("--codec_audio_sr", type=int,
                        default=16000, help="Audio sampling rate for the codec")
    parser.add_argument("--codec_sr", type=int, default=50,
                        help="Sampling rate for the codec")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling value")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling value")
    parser.add_argument("--temperature", type=float,
                        default=1.0, help="Temperature for sampling")
    parser.add_argument("--silence_tokens", type=int, nargs="*",
                        default=[1388, 1898, 131], help="Silence token IDs")
    parser.add_argument("--kvcache", type=int, default=1,
                        help="Key-value cache flag (0 or 1)")
    parser.add_argument("--stop_repetition", type=int,
                        default=3, help="Stop repetition for generation")
    parser.add_argument("--sample_batch_size", type=int,
                        default=3, help="Batch size for sampling")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./generated_tts",
                        help="directory to save generated audio")
    parser.add_argument("--original_audio", type=str,
                        default="./demo/84_121550_000074_000000.wav", help="location of target audio file")
    parser.add_argument("--original_transcript", type=str,
                        default="But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,",
                        help="original audio transcript")
    parser.add_argument("--target_transcript", type=str,
                        default="Gwynplaine had, besides, for his work and for his feats of strength, I cannot believe that the same model can also do text to speech synthesis too!",
                        help="target audio transcript")
    parser.add_argument("--cut_off_sec", type=float, default=3.6,
                        help="cut off point in seconds for input prompt")
    args = parser.parse_args()
    return args


args = parse_arguments()

voicecraft_name = args.model_name
# hyperparameters for inference
codec_audio_sr = args.codec_audio_sr
codec_sr = args.codec_sr
top_k = args.top_k
top_p = args.top_p  # defaults to 0.9 can also try 0.8, but 0.9 seems to work better
temperature = args.temperature
silence_tokens = args.silence_tokens
kvcache = args.kvcache  # NOTE if OOM, change this to 0, or try the 330M model

# NOTE adjust the below three arguments if the generation is not as good
# NOTE if the model generate long silence, reduce the stop_repetition to 3, 2 or even 1
stop_repetition = args.stop_repetition

# NOTE: if the if there are long silence or unnaturally strecthed words,
# increase sample_batch_size to 4 or higher. What this will do to the model is that the
# model will run sample_batch_size examples of the same audio, and pick the one that's the shortest.
# So if the speech rate of the generated is too fast change it to a smaller number.
sample_batch_size = args.sample_batch_size
seed = args.seed  # change seed if you are still unhappy with the result

# load the model
model = voicecraft.VoiceCraft.from_pretrained(
    f"pyp1/VoiceCraft_{voicecraft_name.replace('.pth', '')}")
phn2num = model.args.phn2num
config = vars(model.args)
model.to(device)

encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
if not os.path.exists(encodec_fn):
    os.system(
        f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
    os.system(
        f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")
# will also put the neural codec model on gpu
audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)

text_tokenizer = TextTokenizer(backend="espeak")

# Prepare your audio
# point to the original audio whose speech you want to clone
# write down the transcript for the file, or run whisper to get the transcript (and you can modify it if it's not accurate), save it as a .txt file
orig_audio = args.original_audio
orig_transcript = args.original_transcript

# move the audio and transcript to temp folder
temp_folder = "./demo/temp"
os.makedirs(temp_folder, exist_ok=True)
os.system(f"cp {orig_audio} {temp_folder}")
filename = os.path.splitext(orig_audio.split("/")[-1])[0]
with open(f"{temp_folder}/{filename}.txt", "w") as f:
    f.write(orig_transcript)
# run MFA to get the alignment
align_temp = f"{temp_folder}/mfa_alignments"

os.system("source ~/.bashrc && \
    conda activate voicecraft && \
    mfa align -v --clean -j 1 --output_format csv {temp_folder} \
        english_us_arpa english_us_arpa {align_temp}"
          )

# # if the above fails, it could be because the audio is too hard for the alignment model,
# increasing the beam size usually solves the issue
# os.system("source ~/.bashrc && \
#     conda activate voicecraft && \
#     mfa align -v --clean -j 1 --output_format csv {temp_folder} \
#         english_us_arpa english_us_arpa {align_temp} --beam 1000 --retry_beam 2000")

# take a look at demo/temp/mfa_alignment, decide which part of the audio to use as prompt
cut_off_sec = args.cut_off_sec  # NOTE: according to forced-alignment file demo/temp/mfa_alignments/5895_34622_000026_000002.wav, the word "strength" stop as 3.561 sec, so we use first 3.6 sec as the prompt. this should be different for different audio
target_transcript = args.target_transcript
# NOTE: 3 sec of reference is generally enough for high quality voice cloning, but longer is generally better, try e.g. 3~6 sec.
audio_fn = f"{temp_folder}/{filename}.wav"
info = torchaudio.info(audio_fn)
audio_dur = info.num_frames / info.sample_rate

assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
prompt_end_frame = int(cut_off_sec * info.sample_rate)

# run the model to get the output


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache,
                 "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}
concated_audio, gen_audio = inference_one_sample(model, argparse.Namespace(
    **config), phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_transcript, device, decode_config, prompt_end_frame)

# save segments for comparison
concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
# logging.info(f"length of the resynthesize orig audio: {orig_audio.shape}")

# save the audio
# output_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
seg_save_fn_gen = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_gen_seed{seed}.wav"
seg_save_fn_concat = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_concat_seed{seed}.wav"

torchaudio.save(seg_save_fn_gen, gen_audio, codec_audio_sr)
torchaudio.save(seg_save_fn_concat, concated_audio, codec_audio_sr)

# you might get warnings like WARNING:phonemizer:words count mismatch on 300.0% of the lines (3/1), this can be safely ignored
