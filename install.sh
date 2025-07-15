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
conda_bin=$(which conda)
conda_base=$(dirname $(dirname $conda_bin))
conda_sh="${conda_base}/etc/profile.d/conda.sh"
if [ ! -f "$conda_sh" ]; then
    echo "Could not find conda installation"
    exit 1
fi
source "$conda_sh"
if [ -n "$(conda env list | grep 'ganinv ')" ]; then
    echo "A conda env with the name 'ganinv' already exists"
    exit 1
fi

conda create -n ganinv python=3.10
conda activate ganinv
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
rm ./diff-gaussian-rasterization
cd ./preprocess/3DDFA_V2/
./build.sh

