#!/bin/bash
# Test conditional DDPM inference

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing Conditional DDPM Inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

MODEL_PATH=${1:-"models/DDPM_conditional_10epochs_cosine/ckpt.pt"}
NUM_SAMPLES=${2:-64}
GUIDANCE_SCALE=${3:-3.0}
OUTPUT_DIR="sanity_test_outputs/inference_conditional"

echo "Configuration:"
echo "  - Model: $MODEL_PATH"
echo "  - Number of samples: $NUM_SAMPLES"
echo "  - Guidance scale: $GUIDANCE_SCALE"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

cd ..

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# Test 1: Random classes
echo "Test 1: Generating $NUM_SAMPLES samples with random classes..."
python ex02_main.py \
    --inference \
    --conditional \
    --model_path "$MODEL_PATH" \
    --num_samples $NUM_SAMPLES \
    --guidance_scale $GUIDANCE_SCALE \
    --save_dir "${OUTPUT_DIR}_random" \
    --timesteps 100

echo ""

# Test 2: Specific classes (2 samples per CIFAR-10 class)
echo "Test 2: Generating 20 samples (2 per class) from specific classes..."
python ex02_main.py \
    --inference \
    --conditional \
    --model_path "$MODEL_PATH" \
    --num_samples 20 \
    --guidance_scale $GUIDANCE_SCALE \
    --sample_classes "0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9" \
    --save_dir "${OUTPUT_DIR}_per_class" \
    --timesteps 100

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Inference completed!"
echo "  - Random samples saved to: ${OUTPUT_DIR}_random/inference_samples/"
echo "  - Per-class samples saved to: ${OUTPUT_DIR}_per_class/inference_samples/"
IMG_COUNT_RANDOM=$(find "${OUTPUT_DIR}_random" -name "*.png" 2>/dev/null | wc -l)
IMG_COUNT_CLASS=$(find "${OUTPUT_DIR}_per_class" -name "*.png" 2>/dev/null | wc -l)
echo "  - Total random images: $IMG_COUNT_RANDOM"
echo "  - Total per-class images: $IMG_COUNT_CLASS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
