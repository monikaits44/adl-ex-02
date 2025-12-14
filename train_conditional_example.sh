#!/bin/bash
# Example script for training a conditional diffusion model with classifier-free guidance

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /proj/aimi-adl/envs/adl23_2

# Train conditional model for 30 epochs
python ex02_main.py \
    --conditional \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.003 \
    --p_uncond 0.1 \
    --guidance_scale 3.0 \
    --save_model \
    --run_name DDPM_conditional_30epochs \
    --save_dir outputs_conditional_30ep

echo "Training complete! Model saved to models/DDPM_conditional_30epochs/"
