#!/bin/bash
# Train unconditional DDPM

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training Unconditional DDPM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  - Epochs: ${1:-2}"
echo "  - Batch size: 128"
echo "  - Timesteps: 100"
echo "  - Learning rate: 0.0002"
echo "  - Conditional: No"
echo ""

EPOCHS=${1:-2}
OUTPUT_DIR="sanity_test_outputs/unconditional_${EPOCHS}ep"
RUN_NAME="DDPM_${EPOCHS}epochs_cosine"
LOG_FILE="sanity_test_outputs/training_${EPOCHS}ep.log"

cd ..
python ex02_main.py \
    --epochs $EPOCHS \
    --batch_size 128 \
    --timesteps 100 \
    --lr 0.0002 \
    --log_interval 100 \
    --save_dir $OUTPUT_DIR \
    --save_model \
    --run_name "$RUN_NAME" \
    2>&1 | tee $LOG_FILE

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training completed!"
echo "  - Model saved to: models/$RUN_NAME/ckpt.pt"
echo "  - Outputs saved to: $OUTPUT_DIR"
echo "  - Log saved to: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
