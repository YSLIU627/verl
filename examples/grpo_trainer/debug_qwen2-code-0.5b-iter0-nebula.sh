#set -x
#export CUDA_VISIBLE_DEVICES=2,3,4,5

### task name can be selected from [gsm8k, math_dataset, opencoder]
TASK_NAME=opencoder
# comment START_IDX and END_IDX if you want to use the whole dataset for the training
#START_IDX=0
#END_IDX=2000
KL_CORRECTION=0.01
REMOTE_DATA_PATH=ZHLiu627/dataset_qwen2.5_code_1.5b_grpo_iter0_full_data_miao_0212_2_global_step_70filtered_v1
SAVE_LOCAL_DIR_PREFIX='checkpoints/'
PROJECT_NAME=qwen2.5_code_0.5b_grpo
EXPERIMENT_NAME=debug
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}

### preprocess the dataset
if [ -z "${START_IDX:-}" ]; then
    DATA_PATH_SUFF=${TASK_NAME}
    python3 examples/data_preprocess/${TASK_NAME}.py --local_dir $HOME/data/$DATA_PATH_SUFF --data_remote_dir $REMOTE_DATA_PATH
else
    DATA_PATH_SUFF=${TASK_NAME}_${START_IDX}_${END_IDX}
    python3 examples/data_preprocess/${TASK_NAME}.py --local_dir $HOME/data/$DATA_PATH_SUFF --sample_start_idx $START_IDX --sample_end_idx $END_IDX --data_remote_dir $REMOTE_DATA_PATH
fi

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=84f03efa3815c8727157b1951519ce4b0f2a190a
python3 -m verl.trainer.main_ppo_correct \
    algorithm.kl_ctrl.kl_coef_correction=${KL_CORRECTION} \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=prime \
    data.custom_temp_dir=$HOME/tmp/ray/ \
    data.save_ppo_rollouts_path=rollouts/qwen2.5_code_0.5b_grpo/ \
    data.train_files=$HOME/data/$DATA_PATH_SUFF/train.parquet \
    data.val_files=$HOME/data/$DATA_PATH_SUFF/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1024 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_LOCAL_DIR} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 $@