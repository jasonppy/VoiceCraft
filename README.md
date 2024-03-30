# VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild
VoiceCraft is a token infilling neural codec language model, delivering cutting-edge performance in **speech editing** and **zero-shot text-to-speech (TTS)** across diverse real-world data like audiobooks, internet videos, and podcasts. With just a few seconds of reference, VoiceCraft can clone or modify unseen voices effortlessly.

[Demo](https://jasonppy.github.io/VoiceCraft_web) | [Paper](https://jasonppy.github.io/assets/pdfs/VoiceCraft.pdf)

## News
:star: **03/28/2024:** Model weights are now available on HuggingFace🤗 [here](https://huggingface.co/pyp1/VoiceCraft/tree/main)!

## TODO
Remaining tasks will be completed by the end of March 2024.
- [x] Codebase upload
- [x] Environment setup
- [x] Inference demo for speech editing and TTS
- [x] Training guidance
- [x] RealEdit dataset and training manifest
- [x] Model weights (both 330M and 830M, the former seems to be just as good)
- [ ] Author comprehensive Colab notebooks for better hands-on experience
- [ ] Conduct HuggingFace Spaces demo
- [ ] Enhance guidance on training procedures

## Environment Setup
```bash
conda create -n voicecraft python=3.9.16
conda activate voicecraft

pip install torch==2.0.1 # Assuming CUDA 11.7 compatibility; adjust as needed
apt-get install ffmpeg # Install ffmpeg if not already present
pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
apt-get install espeak-ng # Phonemizer backend
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install torchaudio==2.0.2
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
# Install MFA for forced-alignment; might take a few minutes
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
```

For precise version matching, refer to [environment.yml](./environment.yml).

## Inference Examples
Refer to [`inference_speech_editing.ipynb`](./inference_speech_editing.ipynb) and [`inference_tts.ipynb`](./inference_tts.ipynb).

## Training
To train a VoiceCraft model, prepare:
1. Utterances and their transcripts
2. Encode the utterances into codes using Encodec
3. Convert transcripts into phoneme sequences and a phoneme set (vocab.txt)
4. Manifest (metadata)

Handling steps 1-3 is done in [./data/phonemize_encodec_encode_hf.py](./data/phonemize_encodec_encode_hf.py), where:
- Gigaspeech is downloaded via HuggingFace. An agreement is needed for dataset access.
- Phoneme sequences and Encodec codes are extracted using the script.

For example:

```bash
conda activate voicecraft
export CUDA_VISIBLE_DEVICES=0
cd ./data
python phonemize_encodec_encode_hf.py \
--dataset_size xs \
--download_to path/to/store_huggingface_downloads \
--save_dir path/to/store_extracted_codes_and_phonemes \
--encodec_model_path path/to/encodec_model \
--mega_batch_size 120 \
--batch_size 32 \
--max_len 30000
```

Ensure the `encodec_model_path` is available [here](https://huggingface.co/pyp1/VoiceCraft). This model, trained on Gigaspeech XL, features 56M parameters, 4 codebooks with 2048 codes each. Further details are in our [paper](https://jasonppy.github.io/assets/pdfs/VoiceCraft.pdf). If encountering OOM errors during extraction, decrease batch_size and/or max_len.
Extracted codes, phonemes, and vocab.txt will be stored at `path/to/store_extracted_codes_and_phonemes/${dataset_size}/{encodec_16khz_4codebooks,phonemes,vocab.txt}`.

For the manifest, download train.txt and validation.txt from [here](https://huggingface.co/datasets/pyp1/VoiceCraft_RealEdit/tree/main), placing them under `path/to/store_extracted_codes_and_phonemes/manifest/`. Also, download vocab.txt from the same location if using our pretrained VoiceCraft model for consistent phoneme-to-token mapping.

Now, start training:

```bash
conda activate voicecraft
cd ./z_scripts
bash e830M.sh
```

## License
The codebase is under CC BY-NC-SA 4.0 ([LICENSE-CODE](./LICENSE-CODE)), while model weights are under Coqui Public Model License 1.0.0 ([LICENSE-MODEL](./LICENSE-MODEL)). Note usage of code from other repositories under varying licenses:
- `./models/codebooks_patterns.py`: MIT license
- `./models/modules`, `./steps/optim.py`, `data/tokenizer.py`: Apache License, Version 2.0
- Phonemizer: GNU 3.0 License

For phonemizer replacement (text to IPA phoneme mapping), consider [g2p](https://github.com/roedoejet/g2p) (MIT License) or [OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer) (BSD-3-Clause Clear), though untested.

## Acknowledgement
Special thanks to Feiteng for his [VALL-E reproduction](https://github.com/lifeiteng/vall-e), and to the audiocraft team for open-sourcing [encodec](https://github.com/facebookresearch/audiocraft).

## Citation
```
@article{peng2024voicecraft,
  author    = {Peng, Puyuan and Huang, Po-Yao and Li, Daniel and Mohamed, Abdelrahman and Harwath, David},
  title     = {VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild},
  journal   = {arXiv},
  year      = {2024},
}
```

## Disclaimer
Using technologies mentioned in this paper to generate or edit someone's speech without consent (e.g., government leaders, political figures, celebrities) is prohibited. Violation may infringe copyright laws.
