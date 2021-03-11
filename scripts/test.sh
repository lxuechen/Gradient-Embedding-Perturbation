#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
  --private --rgp --clip0 5 --clip1 2 \
  --num_bases 1000 --aux_dataset imagenet --aux_data_size 100 --sess cifar10_GEP_default \
  --batchsize 32
