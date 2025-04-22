#!/bin/bash

# Meta
dataset="MUTAG"
aug_ratio=0.1
CUDA_VISIBLE_DEVICES=0

# HPO
fgw_alphas=("0.05" "0.5" "0.95")
rhos=("0.1" "1" "10")
mixup_alphas=("0.1" "0.3" "0.5" "1.0" "5.0")

for fgw_alpha in "${fgw_alphas[@]}"; do
  for rho in "${rhos[@]}"; do
    for mixup_alpha in "${mixup_alphas[@]}"; do

      echo FGW-Alpha: $fgw_alpha, Rho: $rho, Mixup-Alpha: $mixup_alpha

      python ./src/gmixup_dgl.py \
        --data_path . \
        --backbone GCN \
        --dataset $dataset \
        --lr=1e-3 \
        --gmixup True \
        --seed=0 \
        --num_layers=6 \
        --log_screen True \
        --batch_size 128 \
        --num_hidden 64 \
        --metric adj \
        --agg=sum \
        --pooling=mean \
        --gpu \
        --measure=uniform \
        --kfold \
        --symmetry \
        --epoch=400 \
        --beta_k=$mixup_alpha \
        --alpha=$fgw_alpha \
        --act=relu \
        --bapg \
        --rho=$rho \
        --aug_ratio=$aug_ratio

    done
  done
done


