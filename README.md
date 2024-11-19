# Transformer Model Inference Experiments on Meluxina
## Overview

This repository contains a collection of scripts and files for running experiments on the inference of transformer pre-trained models. These experiments are designed to be executed on Meluxina, a high-performance computing (HPC) platform, utilizing one or multiple GPUs for optimal performance. The aim is to explore, benchmark, and refine the inference capabilities of transformer models in different configurations.

To launch inference with `Mixtral7B` I use `accelerate` like this from an interactive session like 'salloc -A lxp -p gpu --qos default -N 1 -t 8:00:0':

```
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --tee 3 --config_file=config_1node_4GPUs.yaml Mixtral7BFullPrecisionTests.py```
