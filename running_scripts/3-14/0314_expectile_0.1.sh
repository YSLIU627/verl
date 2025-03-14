set -x


# task name can be selected from [gsm8k, math_dataset, opencoder]
TASK_NAME=math_dataset


EXPECTILE=0.1
LOCAL_DATA_PATH=data
MODEL_PATH=Qwen/Qwen2.5-Math-7B
SAVE_LOCAL_DIR_PREFIX=/scratch/m000069/miaolu/checkpoints
CUSTOM_TEMP_DIR=/projects/m000069/miaolu/tmp/ray
PROJECT_NAME=Qwen2.5-Math-7B-PPO
EXPERIMENT_NAME=ppo_math_dataset_expectile_0.1_0314_1
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}/${PROJECT_NAME}/${EXPERIMENT_NAME}


# preprocess the dataset
python3 examples/data_preprocess/${TASK_NAME}.py --local_dir $LOCAL_DATA_PATH/$TASK_NAME


export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=4418d996107a448b1bc6c52e433d2dd864b0a016


python3 -m verl.trainer.main_ppo \
    algorithm.expectile=${EXPECTILE} \
    data.train_files=$LOCAL_DATA_PATH/$TASK_NAME/train.parquet \
    data.val_files=$LOCAL_DATA_PATH/$TASK_NAME/test.parquet \
    data.custom_temp_dir=${CUSTOM_TEMP_DIR} \
    data.train_batch_size=1024 \
    data.val_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=3584 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24000 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=98304 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    +trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.default_local_dir=${SAVE_LOCAL_DIR} \
    trainer.save_freq=30 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 $@
