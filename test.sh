# not  complete

python scripts/test_policy.py \
    --checkpoint /path/to/checkpoint.pt \
    --task Isaac-Cartpole-Direct-v0 \
    --num_episodes 1000 \
    --rl_library rsl_rl \
    --headless

# Record the video of the policy

./isaaclab.sh -p scripts/play_with_recording.py \
    --checkpoint checkpoint.pt \
    --task Isaac-Cartpole-Direct-v0 \
    --rl_library rsl_rl \
    --headless \
    --record_video \
    --output_dir ./recordings