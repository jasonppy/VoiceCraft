FROM jupyter/base-notebook:python-3.9.13

USER root

# Install OS dependencies
RUN apt-get update && apt-get install -y git-core ffmpeg espeak-ng && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update Conda, create the voicecraft environment, and install dependencies
RUN conda update -y -n base -c conda-forge conda && \
    conda create -y -n voicecraft python=3.9.16 && \
    conda run -n voicecraft conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068 && \
    conda run -n voicecraft mfa model download dictionary english_us_arpa && \
    conda run -n voicecraft mfa model download acoustic english_us_arpa && \
    conda run -n voicecraft pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft && \
    conda run -n voicecraft pip install xformers==0.0.22 && \
    conda run -n voicecraft pip install torch==2.0.1 && \
    conda run -n voicecraft pip install torchaudio==2.0.2 && \
    conda run -n voicecraft pip install tensorboard==2.16.2 && \
    conda run -n voicecraft pip install phonemizer==3.2.1 && \
    conda run -n voicecraft pip install datasets==2.16.0 && \
    conda run -n voicecraft pip install torchmetrics==0.11.1 && \
    conda run -n voicecraft pip install huggingface_hub==0.22.2
    

# Install the Jupyter kernel
RUN conda install -n voicecraft ipykernel --update-deps --force-reinstall -y && \
    conda run -n voicecraft python -m ipykernel install --name=voicecraft