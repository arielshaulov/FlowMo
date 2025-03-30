FROM --platform=linux/amd64 pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN pip install --upgrade pip

WORKDIR /storage/itaytuviah/video-motion

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTORCH_KERNEL_CACHE_PATH="/storage/itaytuviah/cache"
ENV TORCH_KERNEL_CACHE_DIR="/storage/itaytuviah/cache"
ENV HF_HOME="/storage/itaytuviah/cache"
ENV TRANSFORMERS_CACHE="/storage/itaytuviah/cache"
ENV WANDB_CACHE_DIR="/storage/itaytuviah/cache"
ENV WANDB_DIR="/storage/itaytuviah/cache/wandb/"
ENV WANDB_CONFIG_DIR="/storage/itaytuviah/cache/wandb/"
ENV WANDB_TEMP_DIR="/storage/itaytuviah/cache/wandb/"
ENV WANDB_DATA_DIR="/storage/itaytuviah/cache/wandb/"

CMD [ "bash" , "run.sh" ]