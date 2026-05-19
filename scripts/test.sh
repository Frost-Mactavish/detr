#!/bin/bash

alias exp="python -u main_open_world.py --dataset owod --num_queries 900 --eval_every 5 --data_root dataset/OWDETR \
                                        --test_set test --num_classes 25 --unmatched_boxes --top_unk 5 --featdim 1024 \
                                        --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 --with_box_refine --two_stage \
                                        --backbone dino_resnet50 --eval"


EXP_DIR=vis/OWDETR_t4
exp --output_dir ${EXP_DIR} --PREV_INTRODUCED_CLS 18 --CUR_INTRODUCED_CLS 6 \
    --train_set task4_ft --epochs 60 \
    --resume exps/5e-5/500/OWDETR_t4_ft/checkpoint.pth