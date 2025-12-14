#!/bin/bash
# Example script for generating images with a conditional model

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /proj/aimi-adl/envs/adl23_2

# Generate samples from specific CIFAR-10 classes
# CIFAR-10 classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 
#                   5=dog, 6=frog, 7=horse, 8=ship, 9=truck

echo "Generating images from conditional model..."
python ex02_main.py \
    --inference \
    --conditional \
    --model_path models/DDPM_conditional_30epochs/ckpt.pt \
    --num_samples 40 \
    --guidance_scale 5.0 \
    --sample_classes "0,1,2,3,4,5,6,7,8,9" \
    --save_dir outputs_conditional_inference

echo "Samples saved to outputs_conditional_inference/inference_samples/"
echo "Try different guidance scales: 0 (no guidance), 3 (moderate), 7 (strong)"
