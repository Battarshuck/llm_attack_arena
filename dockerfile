FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git

ENV DATA_DIRECTORY=/workspace/ \
    JUPYTER_DIR=/


WORKDIR ${DATA_DIRECTORY}

# dependencies
RUN pip install --no-cache-dir \
    pytz==2023.3.post1 \
    urllib3==2.1.0 \
    protobuf==4.25.0 \
    python-dotenv \
    google-generativeai \
    sentencepiece \
    accelerate \
    fastparquet \
    pyarrow \
    openpyxl \
    ml_collections \
    absl-py \
    "fschat[model_worker,webui]" \
    anthropic \
    openai==1.10.0 \
    vllm==v0.8.4


RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    nltk \
    google.generativeai \
    tenacity \
    && pip uninstall nvidia-nccl-cu11 nvidia-nccl-cu12 -y \
    && pip install nvidia-nccl-cu12==2.18.1

