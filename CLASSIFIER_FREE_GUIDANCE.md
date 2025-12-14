# Classifier-Free Guidance Implementation Guide

## Overview

This document provides detailed information about the classifier-free guidance (CFG) implementation in the DDPM model.

## What Was Implemented

### 1. Model Architecture Changes (`ex02_model.py`)

#### Class Embeddings
```python
# Added to Unet.__init__()
if self.class_free_guidance:
    classes_dim = dim * 4
    self.class_emb = nn.Embedding(num_classes + 1, classes_dim)
    self.null_class = num_classes  # Last index as null token
    self.classes_mlp = nn.Sequential(...)
```

- **Embedding Layer**: Maps class indices to dense embeddings
- **Null Token**: Extra embedding (index 10 for CIFAR-10) for unconditional generation
- **MLP**: Processes class embeddings similar to time embeddings

#### ResNet Block Modifications
```python
# Updated ResnetBlock to accept class embeddings
def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
    full_emb_dim = int(default(time_emb_dim, 0)) + int(default(classes_emb_dim, 0))
    self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(full_emb_dim, dim_out * 2))
```

- Concatenates time and class embeddings
- Uses FiLM conditioning (Feature-wise Linear Modulation)
- Applied to all ResNet blocks in the U-Net

### 2. Diffusion Process Changes (`ex02_diffusion.py`)

#### Conditional Sampling
```python
def p_sample(self, model, x, t, t_index, classes=None, guidance_scale=0.0):
    if classes is not None and guidance_scale > 0:
        # Conditional prediction
        predicted_noise_cond = model(x, t, classes)
        # Unconditional prediction
        predicted_noise_uncond = model(x, t, null_classes)
        # Apply guidance
        predicted_noise = predicted_noise_uncond + guidance_scale * (
            predicted_noise_cond - predicted_noise_uncond
        )
```

The guidance formula:
```
ε = ε_uncond + w * (ε_cond - ε_uncond)
```
where:
- `ε`: Final noise prediction
- `ε_uncond`: Unconditional prediction (null token)
- `ε_cond`: Conditional prediction (with class)
- `w`: Guidance scale (weight)

#### Training with Random Dropping
```python
# In model forward pass during training
if self.training:
    uncond_mask = torch.rand(classes.shape[0], device=classes.device) < self.p_uncond
    classes = torch.where(uncond_mask, 
                         torch.tensor(self.null_class, device=classes.device), 
                         classes)
```

- With probability `p_uncond` (default 0.1), replaces class labels with null token
- Teaches model both conditional and unconditional generation
- Essential for classifier-free guidance to work

### 3. Training Script Updates (`ex02_main.py`)

#### New Arguments
- `--conditional`: Enable CFG
- `--p_uncond`: Probability of unconditional training (default 0.1)
- `--guidance_scale`: Guidance strength (default 3.0)
- `--sample_classes`: Specific classes to generate

#### Updated Functions
- `train()`: Passes class labels during training
- `test()`: Evaluates with class conditioning
- `sample_and_save_images()`: Generates with specific classes
- `run_inference_test()`: Supports conditional inference

## Usage Examples

### Training

#### Unconditional (Original)
```bash
python ex02_main.py \
    --epochs 30 \
    --save_model \
    --run_name DDPM_uncond
```

#### Conditional with CFG
```bash
python ex02_main.py \
    --conditional \
    --epochs 30 \
    --p_uncond 0.1 \
    --save_model \
    --run_name DDPM_cond
```

### Inference

#### Generate Specific Classes
```bash
# Generate 10 airplanes with strong guidance
python ex02_main.py \
    --inference \
    --conditional \
    --model_path models/DDPM_cond/ckpt.pt \
    --num_samples 10 \
    --guidance_scale 5.0 \
    --sample_classes "0,0,0,0,0,0,0,0,0,0"
```

#### Generate All Classes
```bash
# Generate 2 samples of each CIFAR-10 class
python ex02_main.py \
    --inference \
    --conditional \
    --model_path models/DDPM_cond/ckpt.pt \
    --num_samples 20 \
    --guidance_scale 3.0 \
    --sample_classes "0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9"
```

## Understanding Guidance Scale

### Visual Guide

```
w = 0.0  →  Random CIFAR-10 images (unconditional)
w = 1.0  →  Standard conditional generation
w = 3.0  →  Moderate guidance (RECOMMENDED)
w = 5.0  →  Strong guidance
w = 7.0  →  Very strong guidance (may reduce diversity)
```

### Effect on Generation

| Scale | Class Consistency | Diversity | Quality |
|-------|------------------|-----------|---------|
| 0.0   | None             | High      | Varies  |
| 1.0   | Moderate         | High      | Good    |
| 3.0   | Good             | Medium    | Good    |
| 5.0   | Very Good        | Medium    | Good    |
| 7.0   | Excellent        | Low       | May degrade |

## Mathematical Details

### Forward Process (Same for both)
```
q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1 - ᾱ_t) I)
```

### Reverse Process (Unconditional)
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Reverse Process (Conditional)
```
p_θ(x_{t-1} | x_t, y) = N(x_{t-1}; μ_θ(x_t, t, y), Σ_θ(x_t, t, y))
```

### Classifier-Free Guidance
```
ε̃_θ(x_t, t, y) = ε_θ(x_t, t, ∅) + w · (ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))
```

where:
- `ε_θ(x_t, t, y)`: Conditional noise prediction
- `ε_θ(x_t, t, ∅)`: Unconditional noise prediction (null token)
- `w`: Guidance scale

## Implementation Notes

### Why p_uncond = 0.1?
- 10% unconditional training ensures model learns both modes
- Too low: Model may not learn unconditional generation well
- Too high: Model may not learn conditional generation well
- 0.1-0.2 is typical in literature

### Why Use Null Token?
- Avoids needing separate unconditional model
- Single model handles both conditional and unconditional
- More efficient than training two models

### Memory Considerations
- Conditional model is slightly larger (class embeddings)
- During CFG inference, model is run twice per step
- Approximately 2x slower inference with guidance

## Troubleshooting

### Issue: Model ignores class labels
**Solution**: Check that `--conditional` flag is set during both training and inference

### Issue: Poor quality with high guidance
**Solution**: Reduce guidance scale (try 3.0 instead of 7.0)

### Issue: Generated images don't match class
**Solutions**:
1. Train longer (30+ epochs)
2. Increase guidance scale
3. Check that p_uncond isn't too high

### Issue: Low diversity in generated images
**Solutions**:
1. Reduce guidance scale
2. Sample different random seeds
3. Check model hasn't overfitted

## Testing

Run the test suite to verify implementation:
```bash
python test_conditional.py
```

Should output:
```
✅ Unconditional model: ALL TESTS PASSED
✅ Conditional model: ALL TESTS PASSED
✅ Backward compatibility: ALL TESTS PASSED
```

## Code Structure

```
ex02_model.py
├── Unet
│   ├── __init__()           # Added class_free_guidance, class embeddings
│   └── forward()            # Added classes parameter, null token handling
├── ResnetBlock
│   ├── __init__()           # Added classes_emb_dim
│   └── forward()            # Added class_emb parameter

ex02_diffusion.py
├── Diffusion
│   ├── p_sample()           # Added classes, guidance_scale parameters
│   ├── sample()             # Added classes, guidance_scale parameters
│   ├── q_sample()           # Added classes parameter (for API consistency)
│   └── p_losses()           # Added classes parameter

ex02_main.py
├── parse_args()             # Added conditional arguments
├── train()                  # Passes class labels
├── test()                   # Evaluates with classes
├── sample_and_save_images() # Generates with classes
├── run_inference_test()     # Conditional inference
└── run()                    # Creates conditional model
```

## References

1. **Classifier-Free Diffusion Guidance** (Ho & Salimans, 2022)
   - https://arxiv.org/abs/2207.12598
   - Introduced CFG technique

2. **GLIDE** (Nichol et al., 2021)
   - https://arxiv.org/abs/2112.10741
   - First major application of CFG

3. **Imagen** (Saharia et al., 2022)
   - https://arxiv.org/abs/2205.11487
   - High-quality CFG results

## Future Enhancements

Possible extensions:
1. **Text Conditioning**: Replace class embeddings with text encoders
2. **Multi-label**: Support multiple simultaneous conditions
3. **Continuous Labels**: Age, size, etc. instead of discrete classes
4. **Learned Guidance**: Optimize guidance scale per timestep
5. **Negative Prompts**: Explicitly avoid certain classes

## License

Same as main project.
