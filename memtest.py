import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "neow"  # TODO change this to your username

import torch
import torchaudio
import numpy as np
import random

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)

import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)

get_gpu_memory()

if __name__ == "__main__":
    # load model, encodec, and phn2num
    # # load model, tokenizer, and other necessary files
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from models import voicecraft

    # import models.voicecraft as voicecraft
    voicecraft_name = "giga330M.pth"  # or giga330M.pth
    ckpt_fn = f"./pretrained_models/{voicecraft_name}"
    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(ckpt_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\?download\=true")
        os.system(f"mv {voicecraft_name}\?download\=true ./pretrained_models/{voicecraft_name}")
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")

    ckpt = torch.load(ckpt_fn, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    # model.half()
    model.eval()

    print("loaded model")
    get_gpu_memory()

    phn2num = ckpt['phn2num']

    text_tokenizer = TextTokenizer(backend="espeak")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)  # will also put the neural codec model on gpu

    # %%

    # Prepare your audio
    # point to the original audio whose speech you want to clone
    # write down the transcript for the file, or run whisper to get the transcript (and you can modify it if it's not accurate), save it as a .txt file
    orig_audio = "./demo/84_121550_000074_000000.wav"
    orig_transcript = "But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,"

    # move the audio and transcript to temp folder
    temp_folder = "./demo/temp"
    os.makedirs(temp_folder, exist_ok=True)
    os.system(f"cp {orig_audio} {temp_folder}")
    filename = os.path.splitext(orig_audio.split("/")[-1])[0]
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(orig_transcript)
    # run MFA to get the alignment
    align_temp = f"{temp_folder}/mfa_alignments"

    # # if the above fails, it could be because the audio is too hard for the alignment model, increasing the beam size usually solves the issue
    # !source ~/.bashrc && \
    #     conda activate voicecraft && \
    #     mfa align -v --clean -j 1 --output_format csv {temp_folder} \
    #         english_us_arpa english_us_arpa {align_temp} --beam 1000 --retry_beam 2000


    # take a look at demo/temp/mfa_alignment, decide which part of the audio to use as prompt
    cut_off_sec = 7.0  # NOTE: according to forced-alignment file demo/temp/mfa_alignments/84_121550_000074_000000.csv, the word "common" stop as 3.01 sec, this should be different for different audio
    target_transcript = "But when I had approached so near to them The common I cannot believe that the same model can also do text to speech synthesis as well! I love shuffle 512 and janise"
    # NOTE: 3 sec of reference is generally enough for high quality voice cloning, but longer is generally better, try e.g. 3~6 sec.
    audio_fn = f"{temp_folder}/{filename}.wav"
    info = torchaudio.info(audio_fn)
    audio_dur = info.num_frames / info.sample_rate

    assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    # run the model to get the output
    # hyperparameters for inference
    codec_audio_sr = 16000
    codec_sr = 50
    top_k = 0
    top_p = 0.8
    temperature = 1
    silence_tokens = [1388, 1898, 131]
    kvcache = 0  # NOTE if OOM, change this to 0, or try the 330M model

    # NOTE adjust the below three arguments if the generation is not as good
    stop_repetition = 3  # NOTE if the model generate long silence, reduce the stop_repetition to 3, 2 or even 1
    sample_batch_size = 1  # NOTE: if the if there are long silence or unnaturally strecthed words, increase sample_batch_size to 5 or higher. What this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest. So if the speech rate of the generated is too fast change it to a smaller number.
    seed = 1  # change seed if you are still unhappy with the result


    def seed_everything(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    seed_everything(seed)

    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition,
                     'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
                     "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}
    from inference_tts_scale import inference_one_sample

    print("before inference")
    get_gpu_memory()

    concated_audio, gen_audio = inference_one_sample(model, ckpt["config"], phn2num, text_tokenizer, audio_tokenizer,
                                                     audio_fn, target_transcript, device, decode_config, prompt_end_frame, False)
    print("after inference")
    get_gpu_memory()

    # save segments for comparison
    concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
    # logging.info(f"length of the resynthesize orig audio: {orig_audio.shape}")


    # display the audio
    # from IPython.display import Audio
    #
    # print("concatenate prompt and generated:")
    # display(Audio(concated_audio, rate=codec_audio_sr))
    #
    # print("generated:")
    # display(Audio(gen_audio, rate=codec_audio_sr))

    # # save the audio
    # # output_dir
    output_dir = "/home/pyp/VoiceCraft/demo/generated_tts"
    os.makedirs(output_dir, exist_ok=True)
    seg_save_fn_gen = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_gen_seed{seed}.wav"
    seg_save_fn_concat = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_concat_seed{seed}.wav"

    torchaudio.save(seg_save_fn_gen, gen_audio, codec_audio_sr)
    torchaudio.save(seg_save_fn_concat, concated_audio, codec_audio_sr)

    print("finished running")

    # if you get error importing T5 in transformers
    # try
    # pip uninstall Pillow
    # pip install Pillow
    # you are might get warnings like WARNING:phonemizer:words count mismatch on 300.0% of the lines (3/1), this can be safely ignored

    # %%


