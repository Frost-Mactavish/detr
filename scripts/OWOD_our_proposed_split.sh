#!/usr/bin/env bash

alias exp="python -u main_open_world.py --dataset owod --num_queries 900 --eval_every 5 --data_root dataset/OWDETR \
                                        --test_set test --num_classes 26 --unmatched_boxes --top_unk 5 --featdim 1024 \
                                        --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 --with_box_refine --two_stage --backbone dino_resnet50"

EXP_DIR=exps/OWDETR_t1
exp --output_dir ${EXP_DIR} --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 6 \
    --train_set 'task1_train' --epochs 12

EXP_DIR=exps/OWDETR_t2
exp --output_dir ${EXP_DIR} --PREV_INTRODUCED_CLS 6 --CUR_INTRODUCED_CLS 12 \
    --train_set 'task2_train' --epochs 16 --lr 1e-5 \
    --pretrain 'exps/OWDETR_t1/checkpoint.pth'

EXP_DIR=exps/OWDETR_t2_ft
exp --output_dir ${EXP_DIR} --PREV_INTRODUCED_CLS 6 --CUR_INTRODUCED_CLS 12 \
    --train_set 'task2_ft' --epochs 28 \
    --pretrain 'exps/OWDETR_t2/checkpoint.pth'

EXP_DIR=exps/OWDETR_t3
exp --output_dir ${EXP_DIR} --PREV_INTRODUCED_CLS 12 --CUR_INTRODUCED_CLS 18 \
    --train_set 'task3_train' --epochs 32 --lr 1e-5 \
    --pretrain 'exps/OWDETR_t2_ft/checkpoint.pth'

EXP_DIR=exps/OWDETR_t3_ft
exp --output_dir ${EXP_DIR} --PREV_INTRODUCED_CLS 12 --CUR_INTRODUCED_CLS 18 \
    --train_set 'task3_ft' --epochs 44 \
    --pretrain 'exps/OWDETR_t3/checkpoint.pth'

EXP_DIR=exps/OWDETR_t4
exp --output_dir ${EXP_DIR} --PREV_INTRODUCED_CLS 18 --CUR_INTRODUCED_CLS 24 \
    --train_set 'task4_train' --epochs 48 --lr 1e-5 \
    --pretrain 'exps/OWDETR_t3_ft/checkpoint.pth'

EXP_DIR=exps/OWDETR_t4_ft
exp --output_dir ${EXP_DIR} --PREV_INTRODUCED_CLS 18 --CUR_INTRODUCED_CLS 24 \
    --train_set 'task4_ft' --epochs 60 \
    --pretrain 'exps/OWDETR_t4/checkpoint.pth'