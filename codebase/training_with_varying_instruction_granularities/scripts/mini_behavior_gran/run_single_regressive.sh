#!/bin/bash

# =============================
# OpenVLA Finetuning Script with argparse-like interface
# =============================
# example
# bash scripts/mini_behavior/run_single.sh --server_name xxxorgan --data_root_dir default --dataset_name mini_behavior_single_instr_pure_H --gpus 2,3
set -euo pipefail

# === Default values ===
# DEFAULT_DATA_ROOT_DIR_xxxorgan="/data/projects/xxxplace0478/xxxnameh/xxxname_Project/vla/SimpleVLA-RL/data/01_raw/modified_nsai_rlds"
# DEFAULT_DATA_ROOT_DIR_xxxorganx="/nfsdata/data/xxxnameh/SimpleVLA-RL/data/01_raw/modified_nsai_rlds"  # Change this if needed

# single sentence version
DEFAULT_DATA_ROOT_DIR_xxxorgan="/data/gpfs/projects/xxxplace2219/xxxname_temp_storage/single_sentence_modified_nsai_rlds"
DEFAULT_DATA_ROOT_DIR_xxxorganx="/nfsdata/data/xxxnameh/SimpleVLA-RL/data/01_raw/single_sentence_modified_nsai_rlds"  # Change this if needed
MASTERPORT=13501
MAX_STEPS=100000
SAVE_FREQ=50000
NUM_STEPS_BEFORE_DECAY=80000
# === Configurable parameters ===
USE_INSTRUCTION_SENTENCE_ENCODER=false    # or false
USE_TEXT_WORLD_STATE_MODEL=true

# if USE_TEXT_WORLD_STATE_MODEL is true, set MAX_STEPS to 200000
if [ "$USE_TEXT_WORLD_STATE_MODEL" = true ]; then
  MAX_STEPS=200000
  SAVE_FREQ=100000
  NUM_STEPS_BEFORE_DECAY=160000
fi

USE_DISCRETE_DIFFUSION=false
USE_L1_REGRESSION=false
# if USE_DISCRETE_DIFFUSION is true should be false
# if [ "$USE_DISCRETE_DIFFUSION" = true ]; then
#   MAX_STEPS=200000
#   SAVE_FREQ=100000
#   NUM_STEPS_BEFORE_DECAY=160000
# fi

# === Valid dataset names ===
VALID_DATASETS=(
  "mini_behavior_full_instr_pure_L"
  "mini_behavior_full_instr_pure_M"
  "mini_behavior_full_instr_pure_H"
  "mini_behavior_full_instr_centroid"
  "mini_behavior_full_instr_axial_L"
  "mini_behavior_full_instr_axial_M"
  "mini_behavior_full_instr_axial_H"
  "mini_behavior_full_instr_edge_1"
  "mini_behavior_full_instr_edge_2"
  "mini_behavior_full_instr_edge_3"
  "mini_behavior_mix_instr_pure_L"
  "mini_behavior_mix_instr_pure_M"
  "mini_behavior_mix_instr_pure_H"
  "mini_behavior_mix_instr_centroid"
  "mini_behavior_mix_instr_axial_L"
  "mini_behavior_mix_instr_axial_M"
  "mini_behavior_mix_instr_axial_H"
  "mini_behavior_mix_instr_edge_1"
  "mini_behavior_mix_instr_edge_2"
  "mini_behavior_mix_instr_edge_3"
  "mini_behavior_single_instr_pure_L"
  "mini_behavior_single_instr_pure_M"
  "mini_behavior_single_instr_pure_H"
  "mini_behavior_single_instr_centroid"
  "mini_behavior_single_instr_axial_L"
  "mini_behavior_single_instr_axial_M"
  "mini_behavior_single_instr_axial_H"
  "mini_behavior_single_instr_edge_1"
  "mini_behavior_single_instr_edge_2"
  "mini_behavior_single_instr_edge_3"
)

# pure H (abstract) can have batch size 5 
# pure M (middle) can have batch size 3 
# pure L (detailed) can have batch size 2?


# === Help message ===
print_help() {
  cat << EOF
Usage: $0 --server_name <mon | xxxorgan> --data_root_dir <path> --dataset_name <name> --gpus <0,1,2,3>

Required Arguments:
  --server_name       Server: 'xxxorganx' or 'xxxorgan'
  --data_root_dir     Root directory of dataset
  --dataset_name      Dataset name (see valid list below)
  --gpus              Comma-separated GPU IDs (e.g., 0,1 or 0,1,2,3)

Optional:
  --help, -h          Show this help message and exit

Valid dataset_name values:
$(printf '  - %s\n' "${VALID_DATASETS[@]}" | head -10)
  ... (and more)

Example:
  $0 --server_name xxxorganx \\
     --data_root_dir /data/projects/... \\
     --dataset_name mini_behavior_full_instr_pure_L \\
     --gpus 0,1

  # Or with spaces (also accepted):
  $0 --gpus "0 1 2"
EOF
  exit 1
}

# === Parse arguments ===
SERVER_NAME=""
DATA_ROOT_DIR=""
DATASET_NAME=""
GPU_IDS_INPUT=""
VLA_PATH="openvla/openvla-7b"
NUM_CHUNKS_FOR_TEXT=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --server_name)
      SERVER_NAME="$2"
      shift 2
      ;;
    --data_root_dir)
      DATA_ROOT_DIR="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --gpus)
      GPU_IDS_INPUT="$2"
      shift 2
      ;;
    --num_chunks_for_text)
      NUM_CHUNKS_FOR_TEXT="$2"
      shift 2
      ;;
    --masterport)
      MASTERPORT="$2"
      shift 2
      ;;
    --vla_path)
      VLA_PATH="$2"
      shift 2
      ;;
    --help|-h)
      print_help
      ;;
    *)
      echo "Error: Unknown argument: $1"
      print_help
      ;;
  esac
done

# === Validate required arguments ===
if [[ -z "$SERVER_NAME" || -z "$DATA_ROOT_DIR" || -z "$DATASET_NAME" || -z "$GPU_IDS_INPUT" ]]; then
  echo "Error: Missing required arguments."
  print_help
fi

# === Validate server_name ===
if [[ "$SERVER_NAME" != "xxxorganx" && "$SERVER_NAME" != "xxxorgan" ]]; then
  echo "Error: --server_name must be 'xxxorganx' or 'xxxorgan'"
  exit 1
fi

# === Load environment ===
if [[ "$SERVER_NAME" == "xxxorganx" ]]; then
  source env_xxxorganx.sh
  echo "Sourced xxxorganx environment"
elif [[ "$SERVER_NAME" == "xxxorgan" ]]; then
  source env_xxxorgan.sh
  echo "Sourced xxxorgan environment"
fi


# === Validate dataset_name ===
valid_dataset=false
for ds in "${VALID_DATASETS[@]}"; do
  if [[ "$ds" == "$DATASET_NAME" ]]; then
    valid_dataset=true
    break
  fi
done
if [[ "$valid_dataset" == false ]]; then
  echo "Error: Invalid --dataset_name: '$DATASET_NAME'"
  echo "Valid options:"
  printf '  - %s\n' "${VALID_DATASETS[@]}"
  exit 1
fi

# set bach size if the selected $DATASET_NAME contain uppercase H, M, L 
# if USE_DISCRETE_DIFFUSION is true, reduce batch size by 1
if [ "$USE_DISCRETE_DIFFUSION" = true ]; then
  if [[ "$DATASET_NAME" == *"_pure_H"* || "$DATASET_NAME" == *"_axial_H"* ]]; then
    BATCH_SIZE=8
  elif [[ "$DATASET_NAME" == *"_pure_M"* ]]; then
    BATCH_SIZE=8
  elif [[ "$DATASET_NAME" == *"_pure_L"* || "$DATASET_NAME" == *"_axial_L"* ]]; then
    BATCH_SIZE=8
  else
    BATCH_SIZE=8  # default
  fi
else
  if [[ "$DATASET_NAME" == *"_pure_H"* || "$DATASET_NAME" == *"_axial_H"* ]]; then
    BATCH_SIZE=8
  elif [[ "$DATASET_NAME" == *"_pure_M"* ]]; then
    BATCH_SIZE=8
  elif [[ "$DATASET_NAME" == *"_pure_L"* || "$DATASET_NAME" == *"_axial_L"* ]]; then
    BATCH_SIZE=8
  else
    BATCH_SIZE=8  # default
  fi
fi





# === Parse and validate GPU IDs ===
# Replace commas and spaces with comma for uniform processing
GPU_IDS_CLEAN=$(echo "$GPU_IDS_INPUT" | tr ', ' ',' | sed 's/,*$//')
echo "Parsed GPU IDs: $GPU_IDS_CLEAN"
if [[ -z "$GPU_IDS_CLEAN" ]]; then
  echo "Error: --gpus cannot be empty"
  exit 1
fi

# Split into array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS_CLEAN"

# Validate each is a number
for gpu in "${GPU_ARRAY[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid GPU ID: '$gpu'. Must be integer."
    exit 1
  fi
done

# Set environment and torchrun args
export CUDA_VISIBLE_DEVICES="$GPU_IDS_CLEAN"
NPROC_PER_NODE=${#GPU_ARRAY[@]}

# === Set data_root_dir default if needed ===
if [[ "$DATA_ROOT_DIR" == "default" ]]; then
  if [[ "$SERVER_NAME" == "xxxorganx" ]]; then
    DATA_ROOT_DIR="$DEFAULT_DATA_ROOT_DIR_xxxorganx"
  else
    DATA_ROOT_DIR="$DEFAULT_DATA_ROOT_DIR_xxxorgan"
  fi
fi

# === Check data directory exists (optional warning) ===
if [[ ! -d "$DATA_ROOT_DIR" ]]; then
  echo "Warning: data_root_dir does not exist: $DATA_ROOT_DIR"
  read -p "Continue anyway? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi


# === Set TORCHRUN_MASTER_IP ===
if [[ "$SERVER_NAME" == "xxxorganx" ]]; then
  TORCHRUN_MASTER_IP='172.26.93.134'
elif [[ "$SERVER_NAME" == "xxxorgan" ]]; then
  TORCHRUN_MASTER_IP='127.0.0.1'
fi

# === Echo configuration ===
echo "=================================="
echo "Configuration:"
echo "  Server: $SERVER_NAME"
echo "  Data Root: $DATA_ROOT_DIR"
echo "  Dataset: $DATASET_NAME"
echo "  GPUs: $GPU_IDS_CLEAN (nproc_per_node=$NPROC_PER_NODE)"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Master IP: $TORCHRUN_MASTER_IP"
echo "  Batch Size: $BATCH_SIZE"
echo "=================================="



# Automatically build the note based on the values above
ENCODER_PART=""
[ "$USE_INSTRUCTION_SENTENCE_ENCODER" = true ] && ENCODER_PART="use_instruction_se" || ENCODER_PART="no_instruction_se"

WSM_PART=""
[ "$USE_TEXT_WORLD_STATE_MODEL" = true ] && WSM_PART="use_text_wsm" || WSM_PART="no_text_wsm"

DIFFUSION_PART=""
[ "$USE_DISCRETE_DIFFUSION" = true ] && DIFFUSION_PART="-with_discrete_diffusion" || DIFFUSION_PART=""

REGRESSIVE_PART=""
[ "$USE_L1_REGRESSION" = false ] && REGRESSIVE_PART="-regressive" || REGRESSIVE_PART=""

RUN_ID_NOTE="finetune-${ENCODER_PART}-chunks${NUM_CHUNKS_FOR_TEXT}-${WSM_PART}${DIFFUSION_PART}${REGRESSIVE_PART}"

# === Launch training ===
torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port=$MASTERPORT \
  vla-scripts/finetune.py \
  --vla_path "$VLA_PATH" \
  --data_root_dir "$DATA_ROOT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$PYTHONPATH" \
  --use_l1_regression $USE_L1_REGRESSION \
  --use_diffusion False \
  --use_discrete_diffusion $USE_DISCRETE_DIFFUSION \
  --use_film False \
  --use_class_balance True \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size $BATCH_SIZE \
  --learning_rate 2.5e-4 \
  --num_steps_before_decay $NUM_STEPS_BEFORE_DECAY \
  --max_steps $MAX_STEPS \
  --save_freq $SAVE_FREQ \
  --resume True \
  --resume_step 100000 \
  --save_latest_checkpoint_only False \
  --image_aug False \
  --lora_rank 32 \
  --wandb_entity "xxx" \
  --wandb_project "OpenVLA-OFT-mini_behavior" \
  --use_instruction_sentence_encoder $USE_INSTRUCTION_SENTENCE_ENCODER \
  --use_text_world_state_model $USE_TEXT_WORLD_STATE_MODEL \
  --num_chunks_for_text $NUM_CHUNKS_FOR_TEXT \
  --run_id_note "$RUN_ID_NOTE"

  # --learning_rate 2.5e-5 or 2.5e-4... smaller batch seems to need smaller lr
  # --use_instruction_sentence_encoder True \
