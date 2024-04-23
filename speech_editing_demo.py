"""
This script will allow you to run Speech Editing inference with Voicecraft
Before getting started, be sure to follow the environment setup.
"""

from inference_speech_editing_scale import inference_one_sample, get_mask_interval
from edit_utils import get_span
from models import voicecraft
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
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
        description="VoiceCraft Speech Editing: see the script for more information on the options")

    parser.add_argument("--model_name", type=str, default="giga330M.pth", choices=[
                        "giga330M.pth", "gigaHalfLibri330M_TTSEnhanced_max16s.pth", "giga830M.pth"],
                        help="VoiceCraft model to use")
    parser.add_argument("--silence_tokens", type=int, nargs="*",
                        default=[1388, 1898, 131], help="Silence token IDs")
    parser.add_argument("--left_margin", type=float,
                        default=0.08, help="Left margin value.")
    parser.add_argument("--right_margin", type=float,
                        default=0.08, help="Right margin value.")
    parser.add_argument("--codec_audio_sr", type=int,
                        default=16000, help="Codec audio sample rate.")
    parser.add_argument("--codec_sr", type=int, default=50,
                        help="Codec sample rate.")
    parser.add_argument("--top_k", type=float, default=0, help="Top k value.")
    parser.add_argument("--top_p", type=float,
                        default=0.8, help="Top p value.")
    parser.add_argument("--temperature", type=float,
                        default=1, help="Temperature value.")
    parser.add_argument("--kvcache", type=float,
                        default=0, help="Kvcache value.")
    parser.add_argument("--seed", type=int, default=1, help="Seed value.")
    parser.add_argument("--beam_size", type=int, default=10,
                        help="beam size for MFA alignment")
    parser.add_argument("--retry_beam_size", type=int, default=40,
                        help="retry beam size for MFA alignment")
    parser.add_argument("--original_audio", type=str,
                        default="./demo/84_121550_000074_000000.wav", help="location of audio file")
    parser.add_argument("--stop_repetition", type=int,
                        default=-1, help="Stop repetition for generation")
    parser.add_argument("--original_transcript", type=str,
                        default="But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,",
                        help="original transcript")
    parser.add_argument("--target_transcript", type=str,
                        default="But when I saw the mirage of the lake in the distance, which the sense deceives, Lost not by distance any of its marks,",
                        help="target transcript")
    parser.add_argument("--edit_type", type=str,
                        default="substitution",
                        choices=["insertion", "substitution", "deletion"],
                        help="type of specified edit")
    parser.add_argument("--output_dir", type=str,
                        default="./demo/generated_se", help="output directory")
    args = parser.parse_args()
    return args


args = parse_arguments()

voicecraft_name = args.model_name

# hyperparameters for inference
left_margin = args.left_margin
right_margin = args.right_margin
codec_audio_sr = args.codec_audio_sr
codec_sr = args.codec_sr
top_k = args.top_k
top_p = args.top_p
temperature = args.temperature
kvcache = args.kvcache
# NOTE: adjust the below three arguments if the generation is not as good
seed = args.seed  # random seed magic
silence_tokens = args.silence_tokens
# if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1
stop_repetition = args.stop_repetition
# what this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
# or gigaHalfLibri330M_TTSEnhanced_max16s.pth, giga830M.pth
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
audio_tokenizer = AudioTokenizer(signature=encodec_fn)

text_tokenizer = TextTokenizer(backend="espeak")

# point to the original file or record the file
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
os.makedirs(align_temp, exist_ok=True)
beam_size = args.beam_size
retry_beam_size = args.retry_beam_size

os.system("source ~/.bashrc && \
    conda activate voicecraft && \
    mfa align -v --clean -j 1 --output_format csv {temp_folder} \
        english_us_arpa english_us_arpa {align_temp} --beam {beam_size} --retry_beam {retry_beam_size}"
          )
# if it fail, it could be because the audio is too hard for the alignment model, increasing the beam size usually solves the issue
# os.system(f"mfa align -j 1 --clean --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp} --beam 1000 --retry_beam 2000")
audio_fn = f"{temp_folder}/{filename}.wav"
transcript_fn = f"{temp_folder}/{filename}.txt"
align_fn = f"{align_temp}/{filename}.csv"

# propose what do you want the target modified transcript to be
target_transcript = args.target_transcript
edit_type = args.edit_type

# if you want to do a second modification on top of the first one, write down the second modification (target_transcript2, type_of_modification2)
# make sure the two modification do not overlap, if they do, you need to combine them into one modification

# run the script to turn user input to the format that the model can take
orig_span, new_span = get_span(orig_transcript, target_transcript, edit_type)
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
info = torchaudio.info(audio_fn)
audio_dur = info.num_frames / info.sample_rate
morphed_span = (max(start - left_margin, 1/codec_sr),
                min(end + right_margin, audio_dur))  # in seconds

# span in codec frames
mask_interval = [[round(morphed_span[0]*codec_sr),
                  round(morphed_span[1]*codec_sr)]]
mask_interval = torch.LongTensor(mask_interval)  # [M,2], M==1 for now

# run the model to get the output

decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition,
                 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens}
orig_audio, new_audio = inference_one_sample(model, argparse.Namespace(
    **config), phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_transcript, mask_interval, device, decode_config)

# save segments for comparison
orig_audio, new_audio = orig_audio[0].cpu(), new_audio[0].cpu()
# logging.info(f"length of the resynthesize orig audio: {orig_audio.shape}")

# save the audio
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

save_fn_new = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_new_seed{seed}.wav"

torchaudio.save(save_fn_new, new_audio, codec_audio_sr)

save_fn_orig = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_orig.wav"
if not os.path.isfile(save_fn_orig):
    orig_audio, orig_sr = torchaudio.load(audio_fn)
    if orig_sr != codec_audio_sr:
        orig_audio = torchaudio.transforms.Resample(
            orig_sr, codec_audio_sr)(orig_audio)
    torchaudio.save(save_fn_orig, orig_audio, codec_audio_sr)

# # if you get error importing T5 in transformers
# # try
# # pip uninstall Pillow
# # pip install Pillow
# # you are likely to get warning looks like WARNING:phonemizer:words count mismatch on 300.0% of the lines (3/1), this can be safely ignored
