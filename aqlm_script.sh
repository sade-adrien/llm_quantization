export CUDA_VISIBLE_DEVICES=2   # or e.g. 0,1,2,3 #include multiple for multi-gpu
export MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.2
export DATASET_PATH=./data/red_pajama_n=4096_4096_context_length_mistral.pth
export SAVE_PATH=./models/Mistral-7B-Instruct-v0.2-AQLM-2bits-1x16/
export WANDB_PROJECT=AQLM
export WANDB_NAME=AQLM_MISTRAL_2bits_1x16

python ./AQLM/main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=1 \
 --val_size=0 \
 --num_codebooks=1 \
 --nbits_per_codebook=16 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=1 \
 --offload_activations \
 --wandb \
 --resume \
 --save $SAVE_PATH
 #--finetune_batch_size=128 \
 #--attn_implementation='flash_attention_2' \