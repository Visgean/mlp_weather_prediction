#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:2
#SBATCH --mem=16000  # memory in Mb
#SBATCH --time=0-12:00:00
#SBATCH -c 4

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/


mkdir -p ${DATASET_DIR}/temperature_850/
mkdir -p ${DATASET_DIR}/geopotential_500/

cp -n /home/${STUDENT_ID}/temperature_850/*.nc ${DATASET_DIR}/temperature_850/
cp -n /home/${STUDENT_ID}/geopotential_500/*.nc ${DATASET_DIR}/geopotential_500/


export SAVE_DIR=/home/${STUDENT_ID}/output_ltsm_5days/
mkdir -p ${SAVE_DIR}
mkdir -p ${SAVE_DIR}/models



# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd /home/${STUDENT_ID}/mlp_weather_prediction/src

python train_two_layer_ltsm_5days.py
