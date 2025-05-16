LOCAL_DIR=/home/cmu/math-curriculum/checkpoints/Math/easy-4096/global_step_200/actor
LOCAL_DIR=/scr-ssd/akchak/exploration_rl/ppo/grpo_deepscaler_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/global_step_200/actor
TARGET_DIR=/scr-ssd/akchak/exploration_rl/ppo/grpo_deepscaler_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/global_step_200/hf

python scripts/model_merger.py --backend fsdp \
    --hf_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR

python scripts/save_tokenizer.py \
    --hf_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --target_dir $TARGET_DIR

