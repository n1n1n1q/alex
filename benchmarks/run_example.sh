#!/bin/bash
# Example script for running benchmarks
# Modify paths to match your setup

# Configuration
# Note: ALEX uses the same STEVE-1 checkpoint (no separate ALEX checkpoint needed)

# Option 1: Use HuggingFace models (recommended - automatic download)
VPT_MODEL="${VPT_MODEL:-CraftJarvis/MineStudio_VPT.bc_early_game_3x}"
VPT_WEIGHTS="${VPT_WEIGHTS:-}"  # Leave empty for HuggingFace models
STEVE_MODEL="${STEVE_MODEL:-CraftJarvis/MineStudio_STEVE-1.official}"

# Option 2: Use local files (uncomment and set paths)
# VPT_MODEL="/path/to/vpt/foundation-model-2x.model"
# VPT_WEIGHTS="/path/to/vpt/foundation-model-2x.weights"
# STEVE_MODEL="/path/to/steve1/steve1.ckpt"

OUTPUT_DIR="./benchmark_results"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Minecraft Agent Benchmark Suite${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if using local files or HuggingFace models
if [[ "$VPT_MODEL" == *"/"* ]] && [[ "$VPT_MODEL" != *"CraftJarvis"* ]]; then
    # Local file path
    echo -e "${GREEN}Using local model files...${NC}"
    if [ ! -f "$VPT_MODEL" ]; then
        echo -e "${RED}ERROR: VPT model file not found: $VPT_MODEL${NC}"
        exit 1
    fi
    if [ -n "$VPT_WEIGHTS" ] && [ ! -f "$VPT_WEIGHTS" ]; then
        echo -e "${RED}ERROR: VPT weights file not found: $VPT_WEIGHTS${NC}"
        exit 1
    fi
    if [[ "$STEVE_MODEL" == *"/"* ]] && [[ "$STEVE_MODEL" != *"CraftJarvis"* ]] && [ ! -f "$STEVE_MODEL" ]; then
        echo -e "${RED}ERROR: STEVE model file not found: $STEVE_MODEL${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Using HuggingFace models (will download if needed)...${NC}"
    echo "  VPT:   $VPT_MODEL"
    echo "  STEVE: $STEVE_MODEL"
fi
echo ""

echo -e "${BLUE}Option 1: Running complete benchmark suite${NC}"
echo -e "${YELLOW}Note: ALEX uses the same STEVE-1 checkpoint${NC}"

# Build command with optional weights argument
CMD="python run_benchmarks.py \
    --vpt-model \"$VPT_MODEL\" \
    --steve-model \"$STEVE_MODEL\" \
    --task-trials 10 \
    --task-max-steps 6000 \
    --dirt-trials 5 \
    --dirt-max-steps 3000 \
    --output-dir \"$OUTPUT_DIR\" \
    --device cuda"

# Add VPT weights if specified
if [ -n "$VPT_WEIGHTS" ]; then
    CMD="$CMD --vpt-weights \"$VPT_WEIGHTS\""
fi

eval $CMD "$VPT_WEIGHTS" \
    --steve-model "$STEVE_MODEL" \
    --task-trials 10 \
    --task-max-steps 6000 \
    --dirt-trials 5 \
    --dirt-max-steps 3000 \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

# Option 2: Run benchmarks individually (uncomment to use)
# This is useful for debugging or running specific tests

# echo -e "${BLUE}Option 2: Running individual benchmarks${NC}"

# # Task success benchmarks
# for MODEL in vpt steve alex; do
#     for TASK in crafting_table stone_axe iron_ore; do
#         echo -e "${GREEN}Running $MODEL on $TASK${NC}"
#         
#         if [ "$MODEL" = "vpt" ]; then
#             python task_success_benchmark.py \
#                 --model "$MODEL" \
#                 --task "$TASK" \
#                 --model-path "$VPT_MODEL" \
#                 --weights-path "$VPT_WEIGHTS" \
#                 --trials 10 \
#                 --max-steps 6000 \
#                 --output-dir "$OUTPUT_DIR"
#         else
#             python task_success_benchmark.py \
#                 --model "$MODEL" \
#                 --task "$TASK" \
#                 --model-path "$STEVE_MODEL" \
#                 --trials 10 \
#                 --max-steps 6000 \
#                 --output-dir "$OUTPUT_DIR"
#         fi
#     done
# done

# # Dirt mining benchmarks
# for MODEL in vpt steve alex; do
#     echo -e "${GREEN}Running dirt mining for $MODEL${NC}"
#     
#     if [ "$MODEL" = "vpt" ]; then
#         python dirt_mining_benchmark.py \
#             --model "$MODEL" \
#             --model-path "$VPT_MODEL" \
#             --weights-path "$VPT_WEIGHTS" \
#             --trials 5 \
#             --max-steps 3000 \
#             --output-dir "$OUTPUT_DIR"
#     else
#         python dirt_mining_benchmark.py \
#             --model "$MODEL" \
#             --model-path "$STEVE_MODEL" \
#             --trials 5 \
#             --max-steps 3000 \
#             --output-dir "$OUTPUT_DIR"
#     fi
# done

# Find the most recent results directory
LATEST_RESULTS=$(ls -td "$OUTPUT_DIR"/benchmark_suite_* 2>/dev/null | head -1)

if [ -z "$LATEST_RESULTS" ]; then
    echo -e "${YELLOW}No results found to analyze${NC}"
else
    echo ""
    echo -e "${GREEN}Analyzing results...${NC}"
    python analyze_results.py "$LATEST_RESULTS"
    
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${GREEN}Benchmarks complete!${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    echo -e "Results saved to: ${YELLOW}$LATEST_RESULTS${NC}"
    echo ""
    echo "View the summary:"
    echo "  - Text report: $LATEST_RESULTS/comparison_report.txt"
    echo "  - HTML summary: $LATEST_RESULTS/benchmark_summary.html"
    echo "  - Charts: $LATEST_RESULTS/*.png"
    echo ""
fi
