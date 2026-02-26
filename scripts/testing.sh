python scripts/eval_cartpole_success_rate.py --checkpoint /path/to/model.pt --num_episodes 100
python scripts/eval_quadcopter_success_rate.py --checkpoint /path/to/model_99.pt --num_episodes 100

python scripts/eval_cartpole_success_rate.py \
    --checkpoint logs/rl_runs/rsl_rl_eureka/cartpole_direct/2026-02-10_15-23-46_Run-1/model_99.pt \
    --num_episodes 100