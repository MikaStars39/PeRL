#!/bin/bash

# Automatic evaluation script for all checkpoints in a directory
# Usage: bash scripts/eval/eval.sh <checkpoints_dir>
# Example: bash scripts/eval/eval.sh outputs/train/my_experiment

set -e

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <checkpoints_dir>"
    echo "  checkpoints_dir: Directory containing checkpoint-* subdirectories"
    exit 1
fi

CHECKPOINTS_DIR="$1"
PORT=8001
GPU_ID=0
TENSOR_PARALLEL_SIZE=8
EVAL_CONFIG="scripts/eval/eval_test.yaml"
EVAL_TASKS="math_500_avg_4@n=4,minerva_pass_4@k=4,amc23_pass_32@k=32,aime24_pass_32@k=32,aime25_pass_32@k=32,gpqa_diamond_pass_4@k=4,olympiadbench_pass_4@k=4"
CUSTOM_TASKS="config/math.py"

# Check if checkpoints directory exists
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Error: Directory $CHECKPOINTS_DIR does not exist"
    exit 1
fi

# Cleanup function for Ctrl+C interruption
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo ">>> Killing vLLM server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    exit
}
trap cleanup SIGINT SIGTERM

# Function to wait for vLLM server to be ready
wait_for_vllm() {
    local max_retries=120  # Wait up to 10 minutes (120 * 5s)
    local count=0
    
    echo ">>> Waiting for vLLM server to be ready..."
    while [ $count -lt $max_retries ]; do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo ">>> vLLM server is ready!"
            return 0
        fi
        
        # Check if process is still alive
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ">>> Error: vLLM process died unexpectedly!"
            return 1
        fi
        
        sleep 5
        ((count++))
        echo -n "."
    done
    
    echo ""
    echo ">>> Error: vLLM server failed to start within timeout"
    return 1
}

# Main loop: iterate over all checkpoint directories
for CHECKPOINT_DIR in "$CHECKPOINTS_DIR"/checkpoint-*; do
    # Skip if not a directory
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        continue
    fi
    
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
    echo "=========================================="
    echo "Evaluating: $CHECKPOINT_NAME"
    echo "=========================================="
    
    # Load checkpoint as full model
    MODEL_PATH="$CHECKPOINT_DIR"
    
    # Start vLLM server in background
    echo ">>> Starting vLLM server..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "local-model" \
        --max-model-len 32768 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.95 \
        --port $PORT \
        --trust-remote-code \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE &
    
    SERVER_PID=$!
    echo ">>> vLLM server started (PID: $SERVER_PID)"
    
    # Wait for server to be ready
    if ! wait_for_vllm; then
        echo ">>> Skipping $CHECKPOINT_NAME due to vLLM startup failure"
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        sleep 5
        continue
    fi
    
    # Run evaluation
    echo ">>> Running evaluation..."
    OUTPUT_LOG="$CHECKPOINT_DIR/eval_output.log"

    lighteval endpoint litellm \
        "$EVAL_CONFIG" \
        "$EVAL_TASKS" \
        --custom-tasks "$CUSTOM_TASKS" \
        --output_dir "$CHECKPOINT_DIR/eval_results" \
        &> "$OUTPUT_LOG"
    
    if [ $? -eq 0 ]; then
        echo ">>> Evaluation completed successfully"
        echo ">>> Results saved to: $CHECKPOINT_DIR/eval_results"
        echo ">>> Log saved to: $OUTPUT_LOG"
    else
        echo ">>> Error: Evaluation failed. Check log at $OUTPUT_LOG"
    fi
    
    # Stop vLLM server
    echo ">>> Stopping vLLM server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    # Wait for GPU memory to be released
    echo ">>> Waiting for GPU memory to be released..."
    sleep 10
    
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
