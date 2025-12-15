#!/bin/bash
# Script to run each benchmark individually for each model
# This allows you to run specific combinations or resume interrupted benchmarks

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"  # Change to benchmarks directory

# Default configuration file
DEFAULT_CONFIG="benchmark_config.yaml"
CONFIG_FILE=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a task success benchmark
run_task_benchmark() {
    local model=$1
    local task=$2
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running: $model - $task${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    python task_success_benchmark.py \
        --model "$model" \
        --task "$task" \
        --config "$CONFIG_FILE"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Completed: $model - $task${NC}"
    else
        echo -e "${RED}✗ Failed: $model - $task (exit code: $exit_code)${NC}"
        return $exit_code
    fi
    echo ""
}

# Function to run dirt mining benchmark
run_dirt_benchmark() {
    local model=$1
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running: $model - dirt mining${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    python dirt_mining_benchmark.py \
        --model "$model" \
        --config "$CONFIG_FILE"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Completed: $model - dirt mining${NC}"
    else
        echo -e "${RED}✗ Failed: $model - dirt mining (exit code: $exit_code)${NC}"
        return $exit_code
    fi
    echo ""
}

# Parse command line arguments
SELECTED_MODELS=("${MODELS[@]}")
SELECTED_TASKS=("${TASKS[@]}")
RUN_DIRT=true
CONTINUE_ON_ERROR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --models)
            shift
            SELECTED_MODELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_MODELS+=("$1")
                shift
            done
            ;;
        --tasks)
            shift
            SELECTED_TASKS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_TASKS+=("$1")
                shift
            done
            ;;
        --no-dirt)
            RUN_DIRT=false
            shift
            ;;
        --only-dirt)
            SELECTED_TASKS=()
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE               Path to config YAML file (default: benchmark_config.yaml)"
            echo "  --models MODEL1 MODEL2 ...  Run only specified models (vpt, steve, alex)"
            echo "  --tasks TASK1 TASK2 ...     Run only specified tasks (crafting_table, stone_axe, iron_ore)"
            echo "  --no-dirt                   Skip dirt mining benchmarks"
            echo "  --only-dirt                 Run only dirt mining benchmarks"
            echo "  --continue-on-error         Continue even if a benchmark fails"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run only ALEX on crafting table"
            echo "  $0 --models alex --tasks crafting_table"
            echo ""
            echo "  # Run with custom config"
            echo "  $0 --config my_config.yaml --models steve alex"
            echo ""
            echo "  # Run only dirt mining for all models"
            echo "  $0 --only-dirt"
            echo ""
            echo "  # Run VPT and STEVE on all tasks"
            echo "  $0 --models vpt steve"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Use default config if not specified
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file '$CONFIG_FILE' not found${NC}"
    exit 1
fi

# Load configuration using Python
CONFIG_INFO=$(python3 -c "
from benchmark_config import BenchmarkConfig
import yaml
config = BenchmarkConfig('$CONFIG_FILE')
print(f\"{config.get_output_dir()}\")
print(f\"{config.get_device()}\")
")

# Parse config info
OUTPUT_DIR=$(echo "$CONFIG_INFO" | sed -n '1p')
DEVICE=$(echo "$CONFIG_INFO" | sed -n '2p')

# Task list (will be filtered by command-line args)
TASKS=("crafting_table" "stone_axe" "iron_ore")
MODELS=("vpt" "steve" "alex")

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Individual Benchmark Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration File: $CONFIG_FILE"
echo "Output Directory:   $OUTPUT_DIR"
echo "Device:             $DEVICE"
echo ""
echo "Note: Task-specific trials and max_steps are defined in $CONFIG_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}Will run benchmarks for:${NC}"
echo "  Models: ${SELECTED_MODELS[@]}"
if [ ${#SELECTED_TASKS[@]} -gt 0 ]; then
    echo "  Tasks:  ${SELECTED_TASKS[@]}"
fi
if [ "$RUN_DIRT" = true ]; then
    echo "  + Dirt Mining"
fi
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Track progress
TOTAL_BENCHMARKS=$((${#SELECTED_MODELS[@]} * ${#SELECTED_TASKS[@]}))
if [ "$RUN_DIRT" = true ]; then
    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + ${#SELECTED_MODELS[@]}))
fi

COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Run task success benchmarks
if [ ${#SELECTED_TASKS[@]} -gt 0 ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}TASK SUCCESS BENCHMARKS${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    for model in "${SELECTED_MODELS[@]}"; do
        for task in "${SELECTED_TASKS[@]}"; do
            if run_task_benchmark "$model" "$task"; then
                COMPLETED=$((COMPLETED + 1))
            else
                FAILED=$((FAILED + 1))
                if [ "$CONTINUE_ON_ERROR" = false ]; then
                    echo -e "${RED}Stopping due to error. Use --continue-on-error to continue on failures.${NC}"
                    exit 1
                fi
            fi
        done
    done
fi

# Run dirt mining benchmarks
if [ "$RUN_DIRT" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}DIRT MINING BENCHMARKS${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    for model in "${SELECTED_MODELS[@]}"; do
        if run_dirt_benchmark "$model"; then
            COMPLETED=$((COMPLETED + 1))
        else
            FAILED=$((FAILED + 1))
            if [ "$CONTINUE_ON_ERROR" = false ]; then
                echo -e "${RED}Stopping due to error. Use --continue-on-error to continue on failures.${NC}"
                exit 1
            fi
        fi
    done
fi

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}ALL BENCHMARKS COMPLETE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary:"
echo "  Completed: $COMPLETED"
echo "  Failed:    $FAILED"
echo "  Duration:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}Some benchmarks failed. Check the logs above for details.${NC}"
    exit 1
fi

echo -e "${GREEN}All benchmarks completed successfully!${NC}"
