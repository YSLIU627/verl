

EXPECTILE=0.5
export WANDB_API_KEY=84f03efa3815c8727157b1951519ce4b0f2a190a
wandb login --relogin $WANDB_API_KEY

TASK_NAMES=("orz_aime2024" "orz_gpqa_diamond" "orz_math500")
pip install math-verify[antlr4_13_2]
# comment START_IDX and END_IDX if you want to use the whole dataset for the training

SAVE_LOCAL_DIR_PREFIX='checkpoints/'
PROJECT_NAME=Exploration-Orz-Qwen2.5-7B
MODEL_NAME=Qwen/Qwen2.5-7B
EXPERIMENT_NAME=ppo_expectile_${EXPECTILE}_gen4_test4
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}


### preprocess the dataset
DATA_PATHS=()
for TASK_NAME in "${TASK_NAMES[@]}"; do
    echo "Processing task: $TASK_NAME"
    
    if [ -z "${START_IDX:-}" ]; then
        DATA_PATH_SUFF=${TASK_NAME}
        python3 data_preprocess/${TASK_NAME}.py --local_dir ./data/$DATA_PATH_SUFF --data_remote_dir $REMOTE_DATA_PATH
    else
        DATA_PATH_SUFF=${TASK_NAME}_${START_IDX}_${END_IDX}
        python3 data_preprocess/${TASK_NAME}.py --local_dir ./data/$DATA_PATH_SUFF --sample_start_idx $START_IDX --sample_end_idx $END_IDX --data_remote_dir $REMOTE_DATA_PATH
    fi
    DATA_PATHS+=("./data/$DATA_PATH_SUFF")
done
echo "Combined tasks: ${TASK_NAMES[@]}"
python3 data_preprocess/combine_parquet.py --data_dirs ${DATA_PATHS[@]} --output_dir ./data/combined

python3 data_preprocess/orz_dataset.py --local_dir ./data/orz_dataset

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.expectile=${EXPECTILE} \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=prime \
    data.custom_temp_dir=$HOME/tmp/ray/  \
    data.train_files=./data/orz_dataset/train.parquet \
    data.val_files=./data/combined/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=8000 \
    actor_rollout_ref.model.path=${MODEL_NAME} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.warmup_style="constant" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48000 \
    critic.optim.lr=5e-6 \
    critic.use_dynamic_bsz=True \
    critic.optim.lr_warmup_steps_ratio=0.03 \
    critic.optim.warmup_style="constant" \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_NAME} \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.ppo_max_token_len_per_gpu=72000 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_LOCAL_DIR} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.test_sample_n=4 $@