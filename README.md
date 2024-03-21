# VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild
[Demo](https://jasonppy.github.io/VoiceCraft_web) [Paper](https://jasonppy.github.io/assets/pdfs/VoiceCraft.pdf)

TL;DR:
VoiceCraft is a token infilling neural codec language model, that achieves state-of-the-art performance on both **speech editing** and **zero-shot text-to-speech (TTS)** on in-the-wild data including audiobooks, internet videos, and podcasts.

To clone or edit an unseen voice, VoiceCraft needs only a few seconds of reference.


## TODO
The TODOs left will be completed by the end of March 2024.
- [x] Codebase upload
- [x] Environment setup
- [x] Inference demo for speech editing and TTS
- [] Upload model weights
- [] Training guidance
- [] Upload the RealEdit dataset

## Environment setup
```bash
conda create -n voicecraft python=3.9.16
conda activate voicecraft

pip install torch==2.0.1 torchaudio==2.0.2 # this assumes your system is compatible with CUDA 11.7, otherwise checkout https://pytorch.org/get-started/previous-versions/#v201
apt-get install ffmpeg # if you don't already have ffmpeg installed
pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
apt-get install espeak-ng # backend for the phonemizer installed below
pip install phonemizer==3.2.1
pip install tensorboard
pip install datasets==2.12.0
# install MFA for getting forced-alignment, this could take a few minutes
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# conda install pocl # above gives an warning for installing pocl, not sure if really need this

# to run ipynb
conda install -n voicecraft ipykernel --update-deps --force-reinstall
```

## Inference Examples
Checkout [`inference_speech_editing.ipynb`](./inference_speech_editing.ipynb) and [`inference_tts.ipynb`](./inference_tts.ipynb)

## License
The codebase is under CC BY-NC-SA 4.0 ([LICENSE-CODE](./LICENSE-CODE)), and the model weights are under Coqui Public Model License 1.0.0 ([LICENSE-MODEL](./LICENSE-MODEL)). Note that we use some of the code from other repository that are under different licenses: `./models/codebooks_patterns.py` is under MIT license; `./models/modules`, `./steps/optim.py`, `data/tokenizer.py` are under Apache License, Version 2.0; the phonemizer we used is under GNU 3.0 License. For drop-in replacement of the phonemizer (i.e. text to IPA phoneme mapping), try [g2p](https://github.com/roedoejet/g2p) (MIT License) or [OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer) (BSD-3-Clause Clear), although these are not tested.

<!-- How to use g2p to convert english text into IPA phoneme sequence
first install it with `pip install g2p`
```python
from g2p import make_g2p
transducer = make_g2p('eng', 'eng-ipa')
transducer("hello").output_string 
# it will output: 'hʌloʊ'
``` -->

## Acknowledgement
We thank Feiteng for his [VALL-E reproduction](https://github.com/lifeiteng/vall-e), and we thank audiocraft team for open-sourcing [encodec](https://github.com/facebookresearch/audiocraft).

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
Any organization or individual is prohibited from using any technology mentioned in this paper to generate or edit someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

