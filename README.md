# optimal_quad_control_RL
Reinforcement learning for time optimal end-to-end quadcopter control

https://arxiv.org/abs/2504.21586

## Setup instructions
1. clone repo
2. install `uv` (https://docs.astral.sh/uv/getting-started/installation/)
3. set up a venv and download all the packages
4. (optional) train a new model: `uv run python train.py general_session general_model --pi 64 64 64 --vf 64 64 64 --randomization randomized`
5. (if you ran step 4) copy the weights to the inference package (quad_controller_package/quad_controller), overriding the ones already there (100000000.zip)
6. Run inference in drake: `uv run python test_controller_drake.py`
