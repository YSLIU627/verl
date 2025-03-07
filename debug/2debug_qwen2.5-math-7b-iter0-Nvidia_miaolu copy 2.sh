set -x

#export CUDA_VISIBLE_DEVICES=6,7,8,9
export CUDA_VISIBLE_DEVICES=6,7,8,9
# task name can be selected from [gsm8k, math_dataset, opencoder]
TASK_NAME=math_dataset


# comment START_IDX and END_IDX if you want to use the whole dataset for the training
#START_IDX=0
#END_IDX=2000
optimism_coeff=0
optimistic_actor=False
LOCAL_DATA_PATH=data
#MODEL_NAME=Qwen/Qwen2.5-Math-7B
MODEL_NAME=extrop/Qwen2.5-Math-7B-Instruct-alpha0.5
#Qwen/Qwen2.5-Math-0.5B
SAVE_LOCAL_DIR_PREFIX=checkpoints
PROJECT_NAME=Exploration
EXPERIMENT_NAME=debug
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}/${PROJECT_NAME}/${EXPERIMENT_NAME}


# preprocess the dataset
python3 examples/data_preprocess/${TASK_NAME}.py --local_dir $LOCAL_DATA_PATH/$TASK_NAME

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=4418d996107a448b1bc6c52e433d2dd864b0a016


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.optimism_coef=${optimism_coeff} \
    algorithm.optimistic_actor=${optimistic_actor} \
    data.train_files=$LOCAL_DATA_PATH/$TASK_NAME/train.parquet \
    data.val_files=$LOCAL_DATA_PATH/$TASK_NAME/test.parquet \
    data.custom_temp_dir=$HOME/tmp/ray \
    reward_model.reward_manager=prime \
    data.train_batch_size=1024 \
    data.val_batch_size=1024 \
    data.max_prompt_length=256 \
    data.max_response_length=3840 \
    actor_rollout_ref.model.path=${MODEL_NAME} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${SAVE_LOCAL_DIR} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@
    