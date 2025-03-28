FROM --platform=linux/amd64 python:3.9

RUN pip install --upgrade pip

WORKDIR /storage/itaytuviah/video-motion

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTORCH_KERNEL_CACHE_PATH="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
ENV TORCH_KERNEL_CACHE_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
ENV HF_HOME="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
ENV TRANSFORMERS_CACHE="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
ENV WANDB_CACHE_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache"
ENV WANDB_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache/wandb/"
ENV WANDB_CONFIG_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache/wandb/"
ENV WANDB_TEMP_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache/wandb/"
ENV WANDB_DATA_DIR="/home/ai_center/ai_users/itaytuviah/video-motion/cache/wandb/"

CMD [ "bash" , "run.sh" ]