#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import argparse
import importlib

from data.tokenizer import TextTokenizer, AudioTokenizer

# The following requirements are for VoiceCraft inside inference_tts_scale.py
try:
    import torch
    import torchaudio
    import torchmetrics
    import numpy
    import tqdm
    import phonemizer
    import audiocraft
except ImportError:
    print(
        "Pre-reqs not found. Installing numpy, torch, and audio dependencies.")
    subprocess.run(
        ["pip", "install", "numpy", "torch==2.0.1", "torchaudio",
         "torchmetrics", "tqdm", "phonemizer"])

    subprocess.run(["pip", "install", "-e",
                    "git+https://github.com/facebookresearch/audiocraft.git"
                    "@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft"])

from inference_tts_scale import inference_one_sample
from models import voicecraft

description = """
VoiceCraft Inference Text-to-Speech Demo
This script demonstrates how to use the VoiceCraft model for text-to-speech synthesis.

Pre-Requirements:
- Python 3.9.16
- Conda (https://docs.conda.io/en/latest/miniconda.html)
- FFmpeg
- eSpeak NG

Usage:
1. Prepare an audio file and its corresponding transcript.
2. Run the script with the required command-line arguments:
   python voicecraft_tts_demo.py --audio <path_to_audio_file> --transcript <path_to_transcript_file>
3. The generated audio files will be saved in the `./demo/generated_tts` directory.

Notes:
- The script will download the required models automatically if they are not found in the `./pretrained_models` directory.
- You can adjust the hyperparameters using command-line arguments to fine-tune the text-to-speech synthesis.
"""


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None


def run_command(command, error_message):
    if command[0] == "source":
        # Handle the 'source' command separately using os.system()
        status = os.system(" ".join(command))
        if status != 0:
            print(error_message)
            sys.exit(1)
    else:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(error_message)
            sys.exit(1)


def install_linux_dependencies():
    if is_tool("apt-get"):
        # Debian, Ubuntu, and derivatives
        run_command(["sudo", "apt-get", "update"],
                    "Failed to update package lists.")
        run_command(["sudo", "apt-get", "install", "-y", "git-core", "ffmpeg",
                     "espeak-ng"],
                    "Failed to install Linux dependencies.")
    elif is_tool("pacman"):
        # Arch Linux and derivatives
        run_command(["sudo", "pacman", "-Syu", "--noconfirm", "git", "ffmpeg",
                     "espeak-ng"],
                    "Failed to install Linux dependencies.")
    elif is_tool("dnf"):
        # Fedora and derivatives
        run_command(
            ["sudo", "dnf", "install", "-y", "git", "ffmpeg", "espeak-ng"],
            "Failed to install Linux dependencies.")
    elif is_tool("yum"):
        # CentOS and derivatives
        run_command(
            ["sudo", "yum", "install", "-y", "git", "ffmpeg", "espeak-ng"],
            "Failed to install Linux dependencies.")
    else:
        print(
            "Error: Unsupported Linux distribution. Please install the dependencies manually.")
        sys.exit(1)


def install_macos_dependencies():
    if is_tool("brew"):
        packages = ["git", "ffmpeg", "espeak", "anaconda"]
        missing_packages = [package for package in packages if
                            not is_tool(package)]

        if missing_packages:
            run_command(["brew", "install"] + missing_packages,
                        "Failed to install missing macOS dependencies.")
        else:
            print("All required packages are already installed.")

        # Add Anaconda bin directory to PATH
        anaconda_bin_path = "/opt/homebrew/anaconda3/bin"
        os.environ["PATH"] = f"{anaconda_bin_path}:{os.environ['PATH']}"

        # Update the shell configuration file (e.g., .bash_profile or .zshrc)
        shell_config_file = os.path.expanduser(
            "~/.bash_profile")  # or "~/.zshrc" for zsh
        with open(shell_config_file, "a") as file:
            file.write(f'\nexport PATH="{anaconda_bin_path}:$PATH"\n')

    else:
        print(
            "Error: Homebrew not found. Please install Homebrew and try again.")
        sys.exit(1)


def install_dependencies():
    if sys.platform == "win32":
        print(description)
        print("Please install the required dependencies manually on Windows.")
        sys.exit(1)
    elif sys.platform == "darwin":
        install_macos_dependencies()
    elif sys.platform.startswith("linux"):
        install_linux_dependencies()
    else:
        print(f"Unsupported platform: {sys.platform}")
        sys.exit(1)


def install_conda_dependencies():
    conda_packages = [
        "montreal-forced-aligner=2.2.17",
        "openfst=1.8.2",
        "kaldi=5.5.1068"
    ]

    run_command(
        ["conda", "install", "-y", "-c", "conda-forge", "--solver",
         "classic"] + conda_packages,
        "Failed to install Conda packages.")


def create_conda_environment():
    run_command(["conda", "create", "-y", "-n", "voicecraft", "python=3.9.16",
                 "--solver", "classic"],
                "Failed to create Conda environment.")

    # Initialize Conda for the current shell session
    conda_init_command = 'eval "$(conda shell.bash hook)"'
    os.system(conda_init_command)

    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        run_command(["source", bashrc_path],
                    "Failed to source .bashrc.")
    else:
        print("Warning: ~/.bashrc not found. Skipping sourcing.")

    # Activate the Conda environment
    activate_command = f"conda activate voicecraft"
    os.system(activate_command)

    # Install any required dependencies in Conda env
    install_conda_dependencies()


def install_python_dependencies():
    pip_packages = [
        "torch==2.0.1",
        "tensorboard==2.16.2",
        "phonemizer==3.2.1",
        "torchaudio==2.0.2",
        "datasets==2.16.0",
        "torchmetrics==0.11.1"
    ]

    run_command(["pip", "install"] + pip_packages,
                "Failed to install Python packages.")

    run_command(["pip", "install", "-e",
                 "git+https://github.com/facebookresearch/audiocraft.git"
                 "@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft"],
                "Failed to install audiocraft package.")


def download_models(ckpt_fn, encodec_fn):
    if not os.path.exists(ckpt_fn):
        run_command(["wget",
                     f"https://huggingface.co/pyp1/VoiceCraft/resolve/main/{os.path.basename(ckpt_fn)}?download=true"],
                    f"Failed to download {ckpt_fn}.")
        run_command(
            ["mv", f"{os.path.basename(ckpt_fn)}?download=true", ckpt_fn],
            f"Failed to move {ckpt_fn}.")

    if not os.path.exists(encodec_fn):
        run_command(["wget",
                     "https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th"],
                    f"Failed to download {encodec_fn}.")
        run_command(["mv", "encodec_4cb2048_giga.th", encodec_fn],
                    f"Failed to move {encodec_fn}.")


def check_python_dependencies():
    dependencies = [
        "torch",
        "torchaudio",
        "data.tokenizer",
        "models.voicecraft",
        "inference_tts_scale",
        "audiocraft",
        "phonemizer",
        "tensorboard"
    ]

    missing_dependencies = []
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except ImportError:
            missing_dependencies.append(dependency)

    if missing_dependencies:
        print("Missing Python dependencies:", missing_dependencies)
        install_python_dependencies()


def parse_arguments():
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-a", "--audio", required=True,
                        help="Path to the input audio file used as a "
                             "reference for the voice.")
    parser.add_argument("-t", "--transcript", required=True,
                        help="Path to the text file containing the transcript "
                             "to be synthesized.")
    parser.add_argument("--skip-install", "-s", action="store_true",
                        help="Skip the installation of prerequisites.")
    parser.add_argument("--output_dir", default="./demo/generated_tts",
                        help="Output directory where the generated audio "
                             "files will be saved. Default: "
                             "'./demo/generated_tts'")
    parser.add_argument("--cut-off-sec", type=float, default=3.0,
                        help="Cut-off time in seconds for the audio prompt ("
                             "hundredths of a second are acceptable). "
                             "Default: 3.0")
    parser.add_argument("--left_margin", type=float, default=0.08,
                        help="Left margin of the audio segment used for "
                             "speech editing. This is not used for "
                             "text-to-speech synthesis. Default: 0.08")
    parser.add_argument("--right_margin", type=float, default=0.08,
                        help="Right margin of the audio segment used for "
                             "speech editing. This is not used for "
                             "text-to-speech synthesis. Default: 0.08")
    parser.add_argument("--codec_audio_sr", type=int, default=16000,
                        help="Sample rate of the audio codec used for "
                             "encoding and decoding. Default: 16000")
    parser.add_argument("--codec_sr", type=int, default=50,
                        help="Sample rate of the codec used for encoding and "
                             "decoding. Default: 50")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling parameter. It limits the number "
                             "of highest probability tokens to consider "
                             "during generation. A higher value (e.g., "
                             "50) will result in more diverse but potentially "
                             "less coherent speech, while a lower value ("
                             "e.g., 1) will result in more conservative and "
                             "repetitive speech. Setting it to 0 disables "
                             "top-k sampling. Default: 0")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="Top-p sampling parameter. It controls the "
                             "diversity of the generated audio by truncating "
                             "the least likely tokens whose cumulative "
                             "probability exceeds 'p'. Lower values (e.g., "
                             "0.5) will result in more conservative and "
                             "repetitive speech, while higher values (e.g., "
                             "0.9) will result in more diverse speech. "
                             "Default: 0.8")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. It controls the "
                             "randomness of the generated speech. Higher "
                             "values (e.g., 1.5) will result in more "
                             "expressive and varied speech, while lower "
                             "values (e.g., 0.5) will result in more "
                             "monotonous and conservative speech. Default: 1.0")
    parser.add_argument("--kvcache", type=int, default=1,
                        help="Key-value cache size used for caching "
                             "intermediate results. A larger cache size may "
                             "improve performance but consume more memory. "
                             "Default: 1")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility. Use the same "
                             "seed value to generate the same output for a "
                             "given input. Default: 1")
    parser.add_argument("--stop_repetition", type=int, default=3,
                        help="Stop repetition threshold. It controls the "
                             "number of consecutive repetitions allowed in "
                             "the generated speech. Lower values (e.g., "
                             "1 or 2) will result in less repetitive speech "
                             "but may also lead to abrupt stopping. Higher "
                             "values (e.g., 4 or 5) will allow more "
                             "repetitions. Default: 3")
    parser.add_argument("--sample_batch_size", type=int, default=4,
                        help="Number of audio samples generated in parallel. "
                             "Increasing this value may improve the quality "
                             "of the generated speech by reducing long "
                             "silences or unnaturally stretched words, "
                             "but it will also increase memory usage. "
                             "Default: 4")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.skip_install:
        install_dependencies()
        create_conda_environment()
        check_python_dependencies()

    orig_audio = args.audio
    orig_transcript = args.transcript
    output_dir = args.output_dir

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameters for inference
    left_margin = args.left_margin
    right_margin = args.right_margin
    codec_audio_sr = args.codec_audio_sr
    codec_sr = args.codec_sr
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    kvcache = args.kvcache
    silence_tokens = [1388, 1898, 131]
    seed = args.seed
    stop_repetition = args.stop_repetition
    sample_batch_size = args.sample_batch_size

    # Set the device based on available hardware
    if torch.cuda.is_available():
        device = "cuda"
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Move audio and transcript to temp folder
    temp_folder = "./demo/temp"
    os.makedirs(temp_folder, exist_ok=True)
    subprocess.run(["cp", orig_audio, temp_folder])
    filename = os.path.splitext(os.path.basename(orig_audio))[0]
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(orig_transcript)

    # Run MFA to get the alignment
    align_temp = f"{temp_folder}/mfa_alignments"
    os.makedirs(align_temp, exist_ok=True)
    subprocess.run(
        ["mfa", "model", "download", "dictionary", "english_us_arpa"])
    subprocess.run(["mfa", "model", "download", "acoustic", "english_us_arpa"])
    subprocess.run(
        ["mfa", "align", "-v", "--clean", "-j", "1", "--output_format", "csv",
         temp_folder, "english_us_arpa", "english_us_arpa", align_temp])

    audio_fn = f"{temp_folder}/{os.path.basename(orig_audio)}"
    transcript_fn = f"{temp_folder}/{filename}.txt"
    align_fn = f"{align_temp}/{filename}.csv"

    # Decide which part of the audio to use as prompt based on forced alignment
    cut_off_sec = args.cut_off_sec

    info = torchaudio.info(audio_fn)
    audio_dur = info.num_frames / info.sample_rate
    assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    # Load model, tokenizer, and other necessary files
    voicecraft_name = "giga830M.pth"
    ckpt_fn = f"./pretrained_models/{voicecraft_name}"
    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"

    if not os.path.exists(ckpt_fn):
        subprocess.run(["wget",
                        f"https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}?download=true"])
        subprocess.run(["mv", f"{voicecraft_name}?download=true",
                        f"./pretrained_models/{voicecraft_name}"])

    if not os.path.exists(encodec_fn):
        subprocess.run(["wget",
                        "https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th"])
        subprocess.run(["mv", "encodec_4cb2048_giga.th",
                        "./pretrained_models/encodec_4cb2048_giga.th"])

    ckpt = torch.load(ckpt_fn, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    phn2num = ckpt['phn2num']
    text_tokenizer = TextTokenizer(backend="espeak")
    audio_tokenizer = AudioTokenizer(
        signature=encodec_fn)  # will also put the neural codec model on gpu

    # Run the model to get the output
    decode_config = {
        'top_k': top_k, 'top_p': top_p, 'temperature': temperature,
        'stop_repetition': stop_repetition, 'kvcache': kvcache,
        "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
        "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size
    }
    concated_audio, gen_audio = inference_one_sample(
        model, ckpt["config"], phn2num, text_tokenizer, audio_tokenizer,
        audio_fn, transcript_fn, device, decode_config, prompt_end_frame
    )

    # Save segments for comparison
    concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()

    # Save the audio
    seg_save_fn_gen = os.path.join(output_dir,
                                   f"{os.path.basename(orig_audio)[:-4]}_gen_seed{seed}.wav")
    seg_save_fn_concat = os.path.join(output_dir,
                                      f"{os.path.basename(orig_audio)[:-4]}_concat_seed{seed}.wav")

    torchaudio.save(seg_save_fn_gen, gen_audio, codec_audio_sr)
    torchaudio.save(seg_save_fn_concat, concated_audio, codec_audio_sr)


if __name__ == "__main__":
    main()
