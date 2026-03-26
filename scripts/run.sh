#!/usr/bin/env bash

alias exp="python main_open_world.py --data_root dataset/OWDETR --test_set test --model_type prob --num_queries 900 \
                                     --num_classes 25 --obj_loss_coef 1e-3 --obj_temp 1.3 --with_box_refine --two_stage \
                                     --backbone dino_resnet50 --lr_drop 9 --num_inst_per_class 50"

EXP_DIR=exps/PROB
exp --output_dir ${EXP_DIR}/t1 --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 6 \
    --train_set task1_train --epochs 12 \
    --exemplar_replay_selection --exemplar_replay_max_length 850 --exemplar_replay_cur_file learned_task1_ft.txt
    
exp --output_dir ${EXP_DIR}/t2 --PREV_INTRODUCED_CLS 6 --CUR_INTRODUCED_CLS 6 \
    --train_set task2_train --epochs 16 --lr 1e-5 --freeze_prob_model \
    --exemplar_replay_selection --exemplar_replay_max_length 1679 \
    --exemplar_replay_prev_file learned_task1_ft.txt --exemplar_replay_cur_file learned_task2_ft.txt \
    --pretrain ${EXP_DIR}/t1/checkpoint.pth 
    
exp --output_dir ${EXP_DIR}/t2_ft --PREV_INTRODUCED_CLS 6 --CUR_INTRODUCED_CLS 6 \
    --train_set learned_task2_ft --epochs 28 \
    --pretrain ${EXP_DIR}/t2/checkpoint.pth
    
exp --output_dir ${EXP_DIR}/t3 --PREV_INTRODUCED_CLS 12 --CUR_INTRODUCED_CLS 6 \
    --train_set task3_train --epochs 32 --lr 1e-5 --freeze_prob_model \
    --exemplar_replay_selection --exemplar_replay_max_length 2345 \
    --exemplar_replay_prev_file learned_task2_ft.txt --exemplar_replay_cur_file learned_task3_ft.txt \
    --pretrain ${EXP_DIR}/t2_ft/checkpoint.pth
    
exp --output_dir ${EXP_DIR}/t3_ft --PREV_INTRODUCED_CLS 12 --CUR_INTRODUCED_CLS 6 \
    --train_set learned_task3_ft --epochs 44 \
    --pretrain ${EXP_DIR}/t3/checkpoint.pth
    
exp --output_dir ${EXP_DIR}/t4 --PREV_INTRODUCED_CLS 18 --CUR_INTRODUCED_CLS 6 \
    --train_set task4_train --epochs 48 --lr 1e-5 --freeze_prob_model \
    --exemplar_replay_selection --exemplar_replay_max_length 2664 \
    --exemplar_replay_prev_file learned_task3_ft.txt --exemplar_replay_cur_file learned_task4_ft.txt \
    --pretrain ${EXP_DIR}/t3_ft/checkpoint.pth
    
exp --output_dir ${EXP_DIR}/t4_ft --PREV_INTRODUCED_CLS 18 --CUR_INTRODUCED_CLS 6 \
    --train_set learned_task4_ft --epochs 60 \
    --pretrain ${EXP_DIR}/t4/checkpoint.pth