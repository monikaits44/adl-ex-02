# CIFAR-10 Diffusion Model (DDPM)

## Project Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** for generating CIFAR-10 images. The implementation follows the paper ["Denoising Diffusion Probabilistic Models" by Ho et al.](https://arxiv.org/abs/2006.11239) and includes:
- Multiple noise schedule variants (linear, cosine, sigmoid)
- **Classifier-Free Guidance** for class-conditional generation
- Full training and inference pipelines
- Comprehensive documentation and testing

## Features

- ✅ **Complete DDPM Implementation**: Forward and reverse diffusion processes
- ✅ **Multiple Noise Schedules**: Linear, cosine, and sigmoid beta schedules
- ✅ **U-Net Architecture**: Time-conditioned denoising model with attention mechanisms
- ✅ **Classifier-Free Guidance**: Conditional generation with controllable guidance strength
- ✅ **Training & Inference**: Full training loop with validation and test evaluation
- ✅ **Loss Tracking**: Automatic saving of training, validation, and test losses
- ✅ **Sample Generation**: Generate high-quality 32×32 CIFAR-10 images
- ✅ **Class-Conditional Generation**: Generate specific CIFAR-10 classes on demand

## Project Structure

```
ex02_code_skeleton/
├── ex02_main.py           # Main training and inference script
├── ex02_model.py          # U-Net model architecture
├── ex02_diffusion.py      # Diffusion process implementation
├── ex02_helpers.py        # Helper functions
├── ex02_tests.py          # Unit tests
├── models/                # Saved model checkpoints
│   ├── DDPM_10epochs_GPU/
│   └── DDPM_30epochs_cosine/
├── outputs_30ep/          # Generated samples (30 epochs)
├── outputs_50ep/          # Generated samples (50 epochs)
└── outputs_inference/     # Inference samples
```

## Implementation Details

### Core Components

#### 1. Diffusion Process (`ex02_diffusion.py`)
- **Forward Process (q_sample)**: Adds noise to images according to a beta schedule
- **Reverse Process (p_sample)**: Iteratively denoises images using the trained model
- **Loss Computation (p_losses)**: MSE/L1 loss between predicted and actual noise

#### 2. Model Architecture (`ex02_model.py`)
- **U-Net with Time Embeddings**: Sinusoidal position embeddings for timestep conditioning
- **Attention Mechanisms**: Self-attention layers for better feature learning
- **Residual Blocks**: Deep residual connections with group normalization

#### 3. Noise Schedules (`ex02_diffusion.py`)
- **Linear Schedule**: Standard linear interpolation between beta_start and beta_end
- **Cosine Schedule**: Improved schedule from [Improved DDPM paper](https://arxiv.org/abs/2102.09672)
- **Sigmoid Schedule**: Custom sigmoidal beta schedule

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 32×32 | CIFAR-10 native resolution |
| Timesteps | 100 | Number of diffusion steps |
| Batch Size | 64 | Training batch size |
| Learning Rate | 0.003 | AdamW optimizer learning rate |
| Epochs | 30-50 | Training duration |
| Schedule | Cosine | Beta schedule for noise |

## Usage

### Training from Scratch

#### Unconditional Training
```bash
# Activate conda environment
conda activate /proj/aimi-adl/envs/adl23_2

# Train for 30 epochs (unconditional)
python ex02_main.py --epochs 30 --save_model --run_name DDPM_30epochs --save_dir outputs_30ep

# Train for 50 epochs with loss tracking
python ex02_main.py --epochs 50 --save_model --run_name DDPM_50epochs_cosine --save_dir outputs_50ep --batch_size 64 --lr 0.003
```

#### Conditional Training with Classifier-Free Guidance
```bash
# Train conditional model (can control which class to generate)
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

# Or use the provided script
bash train_conditional_example.sh
```

### Inference (Generate Samples)

#### Unconditional Generation
```bash
# Generate 64 samples from a trained model
python ex02_main.py --inference --model_path models/DDPM_30epochs_cosine/ckpt.pt --num_samples 64 --save_dir outputs_inference
```

#### Conditional Generation (Specific Classes)
```bash
# Generate images from specific CIFAR-10 classes
# Classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 
#          5=dog, 6=frog, 7=horse, 8=ship, 9=truck

# Generate 40 samples with guidance scale 5.0
python ex02_main.py \
    --inference \
    --conditional \
    --model_path models/DDPM_conditional_30epochs/ckpt.pt \
    --num_samples 40 \
    --guidance_scale 5.0 \
    --sample_classes "0,1,2,3,4,5,6,7,8,9" \
    --save_dir outputs_conditional_inference

# Or use the provided script
bash inference_conditional_example.sh

# Try different guidance scales for varying strength:
# --guidance_scale 0.0   # No guidance (like unconditional)
# --guidance_scale 3.0   # Moderate guidance (good balance)
# --guidance_scale 7.0   # Strong guidance (more class-specific)
```

### Command-Line Arguments

```bash
python ex02_main.py [OPTIONS]

Training Options:
  --epochs EPOCHS              Number of training epochs (default: 5)
  --batch_size BATCH_SIZE      Batch size for training (default: 64)
  --lr LR                      Learning rate (default: 0.003)
  --timesteps TIMESTEPS        Number of diffusion timesteps (default: 100)
  --save_model                 Save the trained model checkpoint
  --run_name RUN_NAME         Name for the training run (default: DDPM)
  --save_dir SAVE_DIR         Directory to save outputs (default: outputs)
  --log_interval INT          Logging interval in batches (default: 100)
  --no_cuda                   Disable CUDA (use CPU)

Inference Options:
  --inference                  Run inference mode (no training)
  --model_path PATH           Path to model checkpoint
  --num_samples NUM           Number of samples to generate (default: 64)

Conditional Generation Options:
  --conditional                Enable classifier-free guidance for conditional generation
  --p_uncond FLOAT            Probability of unconditional training (default: 0.1)
                              During training, randomly drops class labels to enable CFG
  --guidance_scale FLOAT      Guidance scale for inference (default: 3.0)
                              0 = no guidance, 3-5 = moderate, 7+ = strong
  --sample_classes CLASSES    Comma-separated class indices (e.g., "0,1,2,3")
                              CIFAR-10: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
                                        5=dog, 6=frog, 7=horse, 8=ship, 9=truck
```

## Results

### Trained Models

| Model | Epochs | Schedule | Location |
|-------|--------|----------|----------|
| DDPM_10epochs_GPU | 10 | Cosine | `models/DDPM_10epochs_GPU/` |
| DDPM_30epochs_cosine | 30 | Cosine | `models/DDPM_30epochs_cosine/` |
| DDPM_50epochs_cosine | 50 | Cosine | `models/DDPM_50epochs_cosine/` (in progress) |

### Generated Samples

- **30-epoch samples**: Available in `outputs_30ep/`
- **Inference samples**: Available in `outputs_inference/inference_samples/`
- **Loss history**: Saved as JSON files (e.g., `loss_history_DDPM_50epochs_cosine.json`)

### Image Quality

The generated images are 32×32 pixels (CIFAR-10 native resolution). While they may appear pixelated when zoomed in, this is expected for this image size. The model learns to generate recognizable CIFAR-10 objects (airplanes, cars, birds, cats, etc.) at this resolution.

## Task Completion Status

### ✅ Completed Tasks

1. **Task 2.2**: Forward and reverse diffusion process implementation
   - ✅ `q_sample()`: Forward diffusion with noise addition
   - ✅ `p_sample()`: Single-step reverse diffusion
   - ✅ `sample()`: Full sampling loop from noise to image
   - ✅ `p_losses()`: Training loss computation

2. **Task 2.3**: Multiple noise schedules
   - ✅ Cosine beta schedule implementation
   - ✅ Sigmoid beta schedule implementation
   - ✅ Integrated into training pipeline

3. **Task 2.4**: Conditional generation with classifier-free guidance ⭐ **NEW**
   - ✅ Class embeddings in U-Net architecture
   - ✅ Conditional training with random label dropping (p_uncond)
   - ✅ Classifier-free guidance sampling
   - ✅ Guidance scale control during inference
   - ✅ Class-specific image generation
   - ✅ Full integration with training and inference pipelines

4. **Training Infrastructure**:
   - ✅ Complete training loop with validation
   - ✅ Loss tracking and JSON export
   - ✅ Model checkpointing
   - ✅ Sample generation during training
   - ✅ Inference mode for pre-trained models
   - ✅ Support for both conditional and unconditional models

### 🔄 Optional/Future Tasks

- **Task 2.5**: Adapt to different datasets (currently optimized for CIFAR-10)

## Loss Tracking

The training script automatically saves loss history in JSON format:

```json
{
  "train_loss": [0.245, 0.198, 0.156, ...],
  "val_loss": [0.267, 0.215, 0.178, ...],
  "test_loss": [0.271, 0.219, 0.182, ...],
  "epochs": [0, 1, 2, ...]
}
```

This allows for easy plotting and analysis of training progress.

## Classifier-Free Guidance Explained

### What is Classifier-Free Guidance?

Classifier-free guidance (CFG) is a technique that allows controlling the generation process by conditioning on class labels, without requiring a separate classifier. It works by:

1. **Training**: Randomly dropping class labels during training (probability `p_uncond`, default 0.1)
   - This teaches the model both conditional and unconditional generation
   
2. **Inference**: Combining conditional and unconditional predictions
   ```
   eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
   ```
   - `guidance_scale = 0`: Pure unconditional (ignores class)
   - `guidance_scale = 1`: Pure conditional (standard prediction)
   - `guidance_scale > 1`: Amplified conditioning (more class-specific)

### Guidance Scale Effects

- **0.0**: No guidance - generates random CIFAR-10 images
- **1.0-3.0**: Subtle guidance - class-consistent with variation
- **3.0-5.0**: Moderate guidance - clear class identity (recommended)
- **5.0-7.0**: Strong guidance - very class-specific
- **>7.0**: Very strong - may reduce diversity or quality

### CIFAR-10 Classes

The model can generate these 10 classes:
- 0: Airplane ✈️
- 1: Automobile 🚗
- 2: Bird 🐦
- 3: Cat 🐱
- 4: Deer 🦌
- 5: Dog 🐕
- 6: Frog 🐸
- 7: Horse 🐴
- 8: Ship 🚢
- 9: Truck 🚚

## Technical Details

### Model Architecture
- **Base Dimension**: 32
- **Dimension Multipliers**: (1, 2, 4)
- **Channels**: 3 (RGB)
- **Attention**: Included in later stages
- **Normalization**: Group normalization
- **Activation**: SiLU (Swish)
- **Class Embeddings**: 128-dim embeddings + MLP (for conditional models)
- **Conditioning**: FiLM-style modulation in ResNet blocks

### Data Preprocessing
```python
transform = Compose([
    RandomHorizontalFlip(),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
])
```

### Training Split
- **Training**: 90% of CIFAR-10 train set (45,000 images)
- **Validation**: 10% of CIFAR-10 train set (5,000 images)
- **Test**: Full CIFAR-10 test set (10,000 images)

## Dependencies

Main requirements:
- PyTorch
- torchvision
- einops
- tqdm
- numpy
- Pillow

Install via conda environment: `/proj/aimi-adl/envs/adl23_2`

## Performance Notes

### Training Time
- **Per Epoch**: ~10-15 minutes on GPU
- **30 Epochs**: ~5-7 hours
- **50 Epochs**: ~8-12 hours

### Sample Generation
- **Single Sample**: ~8 seconds (100 diffusion steps)
- **64 Samples**: ~30-40 seconds on GPU

## Example Workflows

### 1. Quick Start - Unconditional Generation
```bash
# Train a basic unconditional model
python ex02_main.py --epochs 10 --save_model --run_name quick_test

# Generate samples
python ex02_main.py --inference --model_path models/quick_test/ckpt.pt --num_samples 16
```

### 2. Full Unconditional Training
```bash
# Train for 50 epochs with full tracking
python ex02_main.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.003 \
    --save_model \
    --run_name DDPM_50epochs \
    --save_dir outputs_50ep
```

### 3. Conditional Model - Full Pipeline
```bash
# Step 1: Train conditional model
python ex02_main.py \
    --conditional \
    --epochs 30 \
    --p_uncond 0.1 \
    --save_model \
    --run_name DDPM_conditional \
    --save_dir outputs_conditional

# Step 2: Generate specific classes (e.g., all airplanes)
python ex02_main.py \
    --inference \
    --conditional \
    --model_path models/DDPM_conditional/ckpt.pt \
    --num_samples 16 \
    --guidance_scale 5.0 \
    --sample_classes "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" \
    --save_dir airplane_samples

# Step 3: Compare different guidance scales
for scale in 0.0 3.0 7.0; do
    python ex02_main.py \
        --inference \
        --conditional \
        --model_path models/DDPM_conditional/ckpt.pt \
        --num_samples 10 \
        --guidance_scale $scale \
        --sample_classes "3,3,3,3,3,3,3,3,3,3" \
        --save_dir cats_guidance_${scale}
done
```

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
2. Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. ICML.
3. Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. NeurIPS Workshop.
4. [Hugging Face Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion)
5. [Lucidrains DDPM Implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)
6. [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics.](https://arxiv.org/abs/1503.03585)

## Notes

### Why Are Images Pixelated?
The images are 32×32 pixels because that's the native CIFAR-10 resolution. This is **not** a training issue but rather the inherent resolution of the dataset. Training for more epochs improves the quality of the generated content (better object recognition), not the resolution.

### Resolution vs. Quality
- **Resolution**: Fixed at 32×32 (determined by dataset/architecture)
- **Quality**: Improves with training (better object shapes, colors, coherence)

To generate higher resolution images, you would need to:
1. Train on a higher resolution dataset (e.g., CelebA, ImageNet)
2. Adjust the model architecture accordingly
3. Implement super-resolution techniques

## Author

Exercise 2 - Advanced Deep Learning Course
