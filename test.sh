# not  complete

# python scripts/test_policy.py \
#     --checkpoint /path/to/checkpoint.pt \
#     --task Isaac-Cartpole-Direct-v0 \
#     --num_episodes 1000 \
#     --rl_library rsl_rl \
#     --headless

# Record the video of the policy

# ./isaaclab.sh -p scripts/record_cartpole_fallback.py \
#     --checkpoint /scratch/somjit77/Tacreka/logs/rl_runs/rsl_rl_revolve_full/cartpole_direct/2026-02-23_07-42-27_Run-0/model_99.pt \
#     --task Isaac-Cartpole-Direct-v0 \
#     --rl_library rsl_rl \
#     --headless \
#     --num_envs 1 \
#     --num_episodes 1 \
#     --max_frames 600 \
#     --output_file ./recordings/cartpole_fallback_revolve_full.mp4

# Quadcopter fallback recorder (headless-safe, no Isaac camera rendering).
# Replace --checkpoint with your quadcopter checkpoint path.
./isaaclab.sh -p scripts/record_quadcopter_fallback.py \
    --checkpoint /scratch/somjit77/Tacreka/logs/rl_runs/rsl_rl_revolve_full/quadcopter_direct/2026-02-24_13-06-28_Run-0/model_99.pt \
    --task Isaac-Quadcopter-Direct-v0 \
    --rl_library rsl_rl \
    --headless \
    --num_envs 1 \
    --num_episodes 1 \
    --max_frames 900 \
    --output_file ./recordings/quadcopter_fallback_revolve_full.mp4
