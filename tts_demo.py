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
import argparse
import random
import numpy as np
import torchaudio
import torch
import os
os.environ["USER"] = "me"  # TODO change this to your username

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="VoiceCraft TTS Inference: see the script for more information on the options")

    parser.add_argument("-m", "--model_name", type=str, default="giga830M", choices=[
                        "giga330M", "giga830M", "giga330M_TTSEnhanced", "giga830M_TTSEnhanced"],
                        help="VoiceCraft model to use")
    parser.add_argument("-st", "--silence_tokens", type=int, nargs="*",
                        default=[1388, 1898, 131], help="Silence token IDs")
    parser.add_argument("-casr", "--codec_audio_sr", type=int,
                        default=16000, help="Codec audio sample rate.")
    parser.add_argument("-csr", "--codec_sr", type=int, default=50,
                        help="Codec sample rate.")

    parser.add_argument("-k", "--top_k", type=float,
                        default=0, help="Top k value.")
    parser.add_argument("-p", "--top_p", type=float,
                        default=0.8, help="Top p value.")
    parser.add_argument("-t", "--temperature", type=float,
                        default=1, help="Temperature value.")
    parser.add_argument("-kv", "--kvcache", type=float, choices=[0, 1],
                        default=0, help="Kvcache value.")
    parser.add_argument("-sr", "--stop_repetition", type=int,
                        default=-1, help="Stop repetition for generation")
    parser.add_argument("--sample_batch_size", type=int,
                        default=3, help="Batch size for sampling")
    parser.add_argument("-s", "--seed", type=int,
                        default=1, help="Seed value.")
    parser.add_argument("-bs", "--beam_size", type=int, default=50,
                        help="beam size for MFA alignment")
    parser.add_argument("-rbs", "--retry_beam_size", type=int, default=200,
                        help="retry beam size for MFA alignment")
    parser.add_argument("--output_dir", type=str, default="./generated_tts",
                        help="directory to save generated audio")
    parser.add_argument("-oa", "--original_audio", type=str,
                        default="./demo/5895_34622_000026_000002.wav", help="location of audio file")
    parser.add_argument("-ot", "--original_transcript", type=str,
                        default="Gwynplaine had, besides, for his work and for his feats of strength, round his neck and over his shoulders, an esclavine of leather.",
                        help="original transcript")
    parser.add_argument("-tt", "--target_transcript", type=str,
                        default="I cannot believe that the same model can also do text to speech synthesis too!",
                        help="target transcript")
    parser.add_argument("-co", "--cut_off_sec", type=float, default=3.6,
                        help="cut off point in seconds for input prompt")
    parser.add_argument("-ma", "--margin", type=float, default=0.04,
                    help="margin in seconds between the end of the cutoff words and the start of the next word. If the next word is not immediately following the cutoff word, the algorithm is more tolerant to word alignment errors")
    parser.add_argument("-cuttol", "--cutoff_tolerance", type=float, default=1, help="tolerance in seconds for the cutoff time, if given cut_off_sec plus the tolerance, we still are not able to find the next word, we will use the best cutoff time found, i.e. likely no margin or very small margin between the end of the cutoff word and the start of the next word")

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
if voicecraft_name == "330M":
    voicecraft_name = "giga330M"
elif voicecraft_name == "830M":
    voicecraft_name = "giga830M"
elif voicecraft_name == "330M_TTSEnhanced":
    voicecraft_name = "330M_TTSEnhanced"
elif voicecraft_name == "830M_TTSEnhanced":
    voicecraft_name = "830M_TTSEnhanced"
model = voicecraft.VoiceCraft.from_pretrained(
    f"pyp1/VoiceCraft_{voicecraft_name.replace('.pth', '')}")
phn2num = model.args.phn2num
config = vars(model.args)
model.to(device)

encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
if not os.path.exists(encodec_fn):
    os.system(
        f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O ./pretrained_models/encodec_4cb2048_giga.th")
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
beam_size = args.beam_size
retry_beam_size = args.retry_beam_size
alignments = f"{temp_folder}/mfa_alignments/{filename}.csv"
if not os.path.isfile(alignments):
    os.system(f"mfa align -v --clean -j 1 --output_format csv {temp_folder} \
            english_us_arpa english_us_arpa {align_temp} --beam {beam_size} --retry_beam {retry_beam_size}")
# if the above fails, it could be because the audio is too hard for the alignment model,
# increasing the beam_size and retry_beam_size usually solves the issue

def find_closest_word_boundary(alignments, cut_off_sec, margin, cutoff_tolerance = 1):
    with open(alignments, 'r') as file:
        # skip header
        next(file)
        cutoff_time = None
        cutoff_index = None
        cutoff_time_best = None
        cutoff_index_best = None
        lines = [l for l in file.readlines()]
        for i, line in enumerate(lines):
            end = float(line.strip().split(',')[1])
            if end >= cut_off_sec and cutoff_time == None:
                cutoff_time = end
                cutoff_index = i
            if end >= cut_off_sec and end < cut_off_sec + cutoff_tolerance and float(lines[i+1].strip().split(',')[0]) - end >= margin:
                    cutoff_time_best = end + margin * 2 / 3
                    cutoff_index_best = i
                    break
        if cutoff_time_best != None:
            cutoff_time = cutoff_time_best
            cutoff_index = cutoff_index_best
        return cutoff_time, cutoff_index

# take a look at demo/temp/mfa_alignment, decide which part of the audio to use as prompt
# NOTE: according to forced-alignment file demo/temp/mfa_alignments/5895_34622_000026_000002.wav, the word "strength" stop as 3.561 sec, so we use first 3.6 sec as the prompt. this should be different for different audio
cut_off_sec = args.cut_off_sec
margin = args.margin
audio_fn = f"{temp_folder}/{filename}.wav"

cut_off_sec, cut_off_word_idx = find_closest_word_boundary(alignments, cut_off_sec, margin, args.cutoff_tolerance)
target_transcript = " ".join(orig_transcript.split(" ")[:cut_off_word_idx+1]) + " " + args.target_transcript
# NOTE: 3 sec of reference is generally enough for high quality voice cloning, but longer is generally better, try e.g. 3~6 sec.
info = torchaudio.info(audio_fn)
audio_dur = info.num_frames / info.sample_rate

assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
prompt_end_frame = int(cut_off_sec * info.sample_rate)


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

# inference
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
