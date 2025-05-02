#! /bin/sh

#SBATCH --job-name=itayenv
#SBATCH --output=/home/ai_center/ai_users/itaytuviah/video-motion/jobs-out-err/%x-%j.out
#SBATCH --error=/home/ai_center/ai_users/itaytuviah/video-motion/jobs-out-err/%x-%j.err
#SBATCH --open-mode=append      # Append instead of overwriting logs
#SBATCH --time=90 # max time (minutes)
#SBATCH --partition=killable


cd /home/ai_center/ai_users/itaytuviah/video-motion
echo "========== Done CD" 

# conda activate videoenv
# echo "========== Done activate env"

# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# pip install --no-cache-dir -r requirements.txt
# pip install --no-cache_dir torch=2.4.1+cu118
# pip install --no-cache-dir --no-build-isolation tokenizers>=0.20.3
# pip install --no-cache-dir --no-build-isolation flash-attn>=2.7.0
# -r requirements.txt
# echo "========== Done!"

# cp -r /home/ai_center/ai_users/arielshaulov/Wan2.1/wan .
