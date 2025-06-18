#!/bin/bash

files=(
    "bfm_noneck_v3.pkl"
    "bfm_noneck_v3_slim.pkl"
    "mb1_120x120.pth"
    "model_ir_se50.pth"
    "modnet_photographic_portrait_matting.ckpt"
    "param_mean_std_62d_120x120.pkl"
    "shape_predictor_68_face_landmarks.dat"
)

base_url="https://huggingface.co/Fubei/splatviz_inversion_checkpoints/resolve/main/"
mkdir -p models
for file in "${files[@]}"; do
    curl -L -o "models/$file" "${base_url}${file}"
done

cd ./preprocess/3DDFA_V2/
./build.sh
