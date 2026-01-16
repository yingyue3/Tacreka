python scripts/test_policy.py \
    --checkpoint /path/to/checkpoint.pt \
    --task Isaac-Cartpole-Direct-v0 \
    --num_episodes 1000 \
    --rl_library rsl_rl \
    --headless

python scripts/record_policy.py \
    --checkpoint /path/to/checkpoint.pt \
    --task Isaac-Cartpole-Direct-v0 \
    --num_episodes 1 \
    --rl_library rsl_rl \
    --headless \
    --output_dir ./recordings \
    --output_filename policy_execution