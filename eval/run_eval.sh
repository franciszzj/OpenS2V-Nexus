#!/bin/bash

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

export https_proxy=http://fwdproxy:8080
export http_proxy=http://fwdproxy:8080

EXP_DIR=$1

EVAL_TYPE="Open-Domain"  # [Human-Domain, Single-Domain, Open-Domain]
INPUT_VIDEO_FOLDER="/home/zijianzhou/s2v_results/${EXP_DIR}/epoch0"
OUTPUT_JSON_FOLDER="/home/zijianzhou/s2v_results/${EXP_DIR}"
OPENS2V_EVAL_PATH="/home/zijianzhou/OpenS2V-Eval"
OPENS2V_WEIGHT_PATH="/home/zijianzhou/OpenS2V-Weight"
OPENAI_API_KEY=""

conda activate opens2v
echo "current conda env: $CONDA_DEFAULT_ENV"

echo "run get_aesscore.py"
python get_aesscore.py \
  --input_video_folder $INPUT_VIDEO_FOLDER \
  --output_json_folder $OUTPUT_JSON_FOLDER \
  --aes_main_path "${OPENS2V_WEIGHT_PATH}/aesthetic-model.pth"

echo "run get_motion_amplitude.py"
python get_motion_amplitude.py \
  --input_video_folder $INPUT_VIDEO_FOLDER \
  --output_json_folder $OUTPUT_JSON_FOLDER

echo "run get_facesim.py"
python get_facesim.py \
  --input_video_folder $INPUT_VIDEO_FOLDER \
  --input_image_folder $OPENS2V_EVAL_PATH \
  --input_json_file "${OPENS2V_EVAL_PATH}/${EVAL_TYPE}_Eval.json" \
  --output_json_folder $OUTPUT_JSON_FOLDER \
  --model_path $OPENS2V_WEIGHT_PATH

echo "run get_gmescore.py"
python get_gmescore.py \
  --input_video_folder $INPUT_VIDEO_FOLDER \
  --input_json_file "${OPENS2V_EVAL_PATH}/${EVAL_TYPE}_Eval.json" \
  --output_json_folder $OUTPUT_JSON_FOLDER

echo "run get_naturalscore.py"
python get_naturalscore.py \
  --input_video_folder $INPUT_VIDEO_FOLDER \
  --output_json_folder $OUTPUT_JSON_FOLDER \
  --api_key $OPENAI_API_KEY

conda deactivate

conda activate opens2v_yoloworld
echo "current conda env: $CONDA_DEFAULT_ENV"

echo "run get_nexusscore.py"
python get_nexusscore.py \
  --input_video_folder $INPUT_VIDEO_FOLDER \
  --input_image_folder $OPENS2V_EVAL_PATH \
  --input_json_file "${OPENS2V_EVAL_PATH}/${EVAL_TYPE}_Eval.json" \
  --output_json_folder $OUTPUT_JSON_FOLDER \
  --yolo_model_path "${OPENS2V_WEIGHT_PATH}/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth"

conda deactivate

conda activate opens2v_qalign
echo "current conda env: $CONDA_DEFAULT_ENV"

echo "run get_qalignscore.py"
python get_motion_smoothness.py \
  --input_video_folder $INPUT_VIDEO_FOLDER \
  --output_json_folder $OUTPUT_JSON_FOLDER

conda deactivate

echo "run merge_result.py"
python merge_result.py \
  $EVAL_TYPE \
  $OUTPUT_JSON_FOLDER \
  "${OUTPUT_JSON_FOLDER}/${EVAL_TYPE}.json"

cat "${OUTPUT_JSON_FOLDER}/${EVAL_TYPE}.json"
