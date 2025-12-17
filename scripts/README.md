# Scripts Directory

This directory contains shell scripts for training, testing, and sanity checking the DDPM models.

## Training Scripts

### `train_conditional.sh`
Train a conditional DDPM model with classifier-free guidance.

**Usage:**
```bash
# From scripts directory
cd scripts
./train_conditional.sh [EPOCHS]

# Default: 2 epochs
./train_conditional.sh

# Custom: 10 epochs
./train_conditional.sh 10
```

**Configuration:**
- Batch size: 128
- Timesteps: 1000
- Learning rate: 0.0002
- P_uncond: 0.1
- Guidance scale: 3.0

**Outputs:**
- Model: `models/DDPM_conditional_[N]epochs_cosine/ckpt.pt`
- Samples: `sanity_test_outputs/conditional_[N]ep/`
- Log: `sanity_test_outputs/training_conditional_[N]ep.log`

---

### `train_unconditional.sh`
Train an unconditional DDPM model.

**Usage:**
```bash
cd scripts
./train_unconditional.sh [EPOCHS]

# Default: 2 epochs
./train_unconditional.sh

# Custom: 50 epochs
./train_unconditional.sh 50
```

**Configuration:**
- Batch size: 128
- Timesteps: 1000
- Learning rate: 0.0002

**Outputs:**
- Model: `models/DDPM_[N]epochs_cosine/ckpt.pt`
- Samples: `sanity_test_outputs/unconditional_[N]ep/`
- Log: `sanity_test_outputs/training_[N]ep.log`

---

## Inference Testing Scripts

### `test_inference_unconditional.sh`
Test unconditional model inference.

**Usage:**
```bash
cd scripts
./test_inference_unconditional.sh [MODEL_PATH] [NUM_SAMPLES]

# Default: 64 samples from DDPM_30epochs_cosine
./test_inference_unconditional.sh

# Custom model and sample count
./test_inference_unconditional.sh models/DDPM_50epochs_cosine/ckpt.pt 32
```

**Outputs:**
- Images: `sanity_test_outputs/inference_unconditional/inference_samples/`

---

### `test_inference_conditional.sh`
Test conditional model inference with multiple scenarios.

**Usage:**
```bash
cd scripts
./test_inference_conditional.sh [MODEL_PATH] [NUM_SAMPLES] [GUIDANCE_SCALE]

# Default: 64 samples, guidance_scale=3.0
./test_inference_conditional.sh

# Custom parameters
./test_inference_conditional.sh models/DDPM_conditional_10epochs_cosine/ckpt.pt 32 5.0
```

**Tests:**
1. Random classes: Generates N samples with random CIFAR-10 classes
2. Per-class: Generates 2 samples per class (20 total)

**Outputs:**
- Random: `sanity_test_outputs/inference_conditional_random/inference_samples/`
- Per-class: `sanity_test_outputs/inference_conditional_per_class/inference_samples/`

---

## Sanity Check Script

### `run_sanity_check.sh`
Comprehensive test suite for all functionality.

**Usage:**
```bash
cd scripts
./run_sanity_check.sh
```

**Tests:**
1. Plot Beta Schedules
2. Inference Mode (Unconditional)
3. Create Animation (Forward Process)
4. Create Animation (Reverse Process)
5. Run Unit Tests

**Note:** Requires existing trained models at:
- `models/DDPM_30epochs_cosine/ckpt.pt`

---

## Output Organization

All test outputs are saved to `sanity_test_outputs/`:

```
sanity_test_outputs/
├── conditional_2ep/              # Conditional training outputs
├── unconditional_2ep/            # Unconditional training outputs
├── inference_conditional_random/ # Conditional inference (random)
├── inference_conditional_per_class/ # Conditional inference (per class)
├── inference_unconditional/      # Unconditional inference
├── test_inference/               # Sanity check inference
├── training_conditional_2ep.log  # Training logs
└── training_2ep.log             # Training logs
```

---

## Quick Start Examples

```bash
# Navigate to scripts directory
cd scripts

# Train conditional model (2 epochs for testing)
./train_conditional.sh 2

# Test the trained model
./test_inference_conditional.sh models/DDPM_conditional_2epochs_cosine/ckpt.pt 16

# Run all sanity checks
./run_sanity_check.sh
```
