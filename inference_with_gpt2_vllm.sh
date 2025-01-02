#!/bin/bash -l
#SBATCH -A lxp
#SBATCH --job-name=MMvllmTEST
#SBATCH -q short
#SBATCH -p gpu
#SBATCH --time=00:45:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --error="vllm-%j.err"
#SBATCH --output="vllm-%j.out"

module --force purge
module load env/release/2023.1
module load Apptainer/1.3.1-GCCcore-12.3.0

# Fix pmix error (munge)
export PMIX_MCA_psec=native

# to remove warnings such as:
# WARNING: Environment variable HF_HOME already has value [/root/.cache/huggingface],...

unset $HF_HOME

# https://docs.vllm.ai/en/latest/design/huggingface_integration.html
# this is the argument expected by vllm
export HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN}

# Choose a directory for the cache
export LOCAL_HF_CACHE="/mnt/tier1/project/lxp/mmagliulo/huggingFaceCache/new_cache_dir/"
# export LOCAL_HF_CACHE="${PWD}/HF_cache"
mkdir -p ${LOCAL_HF_CACHE}

# uncomment if necessary:
# Apptainer has built-in support for pulling Docker images and converting them into .sif files.
# apptainer pull vllm-openai_latest.sif docker://vllm/vllm-openai:latest

export SIF_IMAGE="vllm-openai_latest.sif"
# export SIF_IMAGE="transformers-pytorch-gpu_latest.sif"

# Bind and environment variables
# important --> -B <host_directory>:<container_directory>.
# It maps a directory from the host filesystem into the container filesystem.
# export APPTAINER_ARGS="--nvccli -B ${PWD}:/workspace -B ${LOCAL_HF_CACHE}:/root/.cache/huggingface --env HF_HOME=/root/.cache/huggingface"
export APPTAINER_ARGS="--nvccli -B ${PWD}:/workspace -B ${LOCAL_HF_CACHE}:/root/.cache/huggingface"

# Define model path and other variables
export HF_MODEL="gpt2"
export HEAD_HOSTNAME="$(hostname)"
export HEAD_IPADDRESS="$(hostname --ip-address)"
export RANDOM_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Commands for Ray setup
export RAY_CMD_HEAD="ray start --block --head --port=${RANDOM_PORT}"
export RAY_CMD_WORKER="ray start --block --address=${HEAD_IPADDRESS}:${RANDOM_PORT}"

# Set parallelism for Pixtral
export TENSOR_PARALLEL_SIZE=4                 # GPUs per node
export PIPELINE_PARALLEL_SIZE=${SLURM_NNODES} # Number of nodes

# Instructions for connecting
echo "HEAD NODE: ${HEAD_HOSTNAME}"
echo "IP ADDRESS: ${HEAD_IPADDRESS}"

export LOCAL_PORT=8080
# [local_port]:[destination_host]:[destination_port]
echo "SSH TUNNEL (Execute on your local machine): ssh -p 8822 ${USER}@login.lxp.lu -NL 8000:${HEAD_IPADDRESS}:${LOCAL_PORT}"

# Start head node
echo "Starting head node"
srun -J "head_ray_node-%J" -N 1 --ntasks-per-node=1 -c $((SLURM_CPUS_PER_TASK / 2)) -w ${HEAD_HOSTNAME} apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} ${RAY_CMD_HEAD} &
sleep 10

# Start worker nodes
echo "Starting worker nodes"
srun -J "worker_ray_node-%J" -N $((SLURM_NNODES - 1)) --ntasks-per-node=1 -c ${SLURM_CPUS_PER_TASK} -x ${HEAD_HOSTNAME} apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} ${RAY_CMD_WORKER} &

sleep 10

# Start the VLLM server on the head node
# do not forget authentification !

echo "Starting server"

#useful for debugging
export VLLM_TRACE_FUNCTION=1

# IMPORTANT
# when running the following command I realized that I got:
# By default, the Hugging Face library tries to use the container's root filesystem (/), which is restricted to 64 MB (overlay). This is why the download fails.
# overlay                                                                                                                     64M   72K   64M   1% /
# tmpfs
# apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} df -h

# because of the comment above, it is really important to Point the Hugging Face cache to a directory with sufficient space

# WORKS do not erase
apptainer exec --nvccli \
        -B ${LOCAL_HF_CACHE}:/workspace/cache \
            --env HF_HOME=/workspace/cache \
                vllm-openai_latest.sif vllm serve ${HF_MODEL} --uvicorn-log-level debug --port ${LOCAL_PORT}

# apptainer exec --nvccli -B ${PWD}:/workspace -B ${LOCAL_HF_CACHE}:/root/.cache/huggingface vllm-openai_latest.sif vllm serve gpt2 --uvicorn-log-level debug

# to be run on the local machine after the port forwarding
# curl -X POST -H "Content-Type: application/json" http://localhost:8000/v1/completions -d '{
# "model": "gpt2",
# "prompt": "San Francisco is a"
# }'

