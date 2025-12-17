#!/bin/bash
# Test unconditional DDPM inference

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing Unconditional DDPM Inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

MODEL_PATH=${1:-"models/DDPM_30epochs_cosine/ckpt.pt"}
NUM_SAMPLES=${2:-64}
OUTPUT_DIR="sanity_test_outputs/inference_unconditional"

echo "Configuration:"
echo "  - Model: $MODEL_PATH"
echo "  - Number of samples: $NUM_SAMPLES"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

cd ..

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

python ex02_main.py \
    --inference \
    --model_path "$MODEL_PATH" \
    --num_samples $NUM_SAMPLES \
    --save_dir $OUTPUT_DIR \
    --timesteps 100

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Inference completed!"
echo "  - Generated images saved to: $OUTPUT_DIR/inference_samples/"
IMG_COUNT=$(find $OUTPUT_DIR -name "*.png" 2>/dev/null | wc -l)
echo "  - Total images generated: $IMG_COUNT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
