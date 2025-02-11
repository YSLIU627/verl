#set -x
export CUDA_VISIBLE_DEVICE=6,7,8,9

### task name can be selected from [gsm8k, math_dataset, opencoder]
TASK_NAME=opencoder
# comment START_IDX and END_IDX if you want to use the whole dataset for the training
START_IDX=0
END_IDX=2000


### preprocess the dataset
if [ -z "${START_IDX:-}" ]; then
    DATA_PATH_SUFF=${TASK_NAME}
    python3 examples/data_preprocess/${TASK_NAME}.py --local_dir $HOME/data/$DATA_PATH_SUFF 
else
    DATA_PATH_SUFF=${TASK_NAME}_${START_IDX}_${END_IDX}
    python3 examples/data_preprocess/${TASK_NAME}.py --local_dir $HOME/data/$DATA_PATH_SUFF --sample_start_idx $START_IDX --sample_end_idx $END_IDX
fi

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=84f03efa3815c8727157b1951519ce4b0f2a190a
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.custom_temp_dir=$HOME/tmp/ray/ \
    data.save_ppo_rollouts_path=rollouts/qwen2.5_code_1.5b_grpo/ \
    data.train_files=$HOME/data/$DATA_PATH_SUFF/train.parquet \
    data.val_files=$HOME/data/$DATA_PATH_SUFF/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1312 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='qwen2.5_code_1.5b' \
    trainer.experiment_name='qwen2.5_code_1.5b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 $@