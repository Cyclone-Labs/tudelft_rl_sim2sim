# optimal_quad_control_RL
Reinforcement learning for time optimal end-to-end quadcopter control

https://arxiv.org/abs/2504.21586

## Overview
This repo is a fork of the original TU Delft one. It adds capabilites to do sim2sim transfer. Basically, we train the RL agent in one simulator, in this case the simple one that TU Delft wrote, and then see how it performs in a different simulator, in this case Drake. In theory this should work well because we have precise control of Drake's physics model and the TU Delft simulator is really simple so we can base our physics exactly on theirs by reading the source code.

The main idea of the original TU Delft paper is that if you vary the parameters of your simulation while training the agent's policy network, it makes it more robust to changes in the environment, which means you can use the same policy in real life. This is called domain randomization and is a really common tactic for sim2real transfer nowadays.

Right now this project is at the point where I've trained a model and verified it can successfully complete the figure-8 trajectory in the TU Delft simulated environment, but when I run it in Drake it starts taking off fine but then flips over and the episode terminates after 58 timesteps every time. I haven't debugged too much but it's weird because I removed most of the termination conditions so it should at least be crashing into the ground before the sim gets reset. Also I haven't instrumented the drake environment with enough logging yet. Ideally we can have plots of every state variable and policy action vs. timestep to visualize in something like tensorboard or rerun. I think this project should be very possible, I'm guessing there's just a bunch of bugs in the drake environment that make the policy fail, or I vibe coded too hard and the command I ran for training is not actually what they used in the paper and the network overfit or there isn't enough randomization. 

## Setup instructions
1. clone repo
2. install `uv` (https://docs.astral.sh/uv/getting-started/installation/)
3. set up a venv and download all the packages
4. (optional) train a new model: `uv run python train.py general_session general_model --pi 64 64 64 --vf 64 64 64 --randomization randomized`
5. (if you ran step 4) copy the weights to the inference package (quad_controller_package/quad_controller), overriding the ones already there (100000000.zip)
6. Run inference in drake: `uv run python test_controller_drake.py`
7. Open localhost:7000 in a web browser to look at the visualization. Once you're ready to start it, go back to your terminal window and press enter to start the experiments

## Setup instructions (Delft Env)
1. clone repo
2. install `uv` (https://docs.astral.sh/uv/getting-started/installation/)
3. set up a venv and download all the packages
4. (optional) train a new model: `uv run python train.py general_session general_model --pi 64 64 64 --vf 64 64 64 --randomization randomized`
5. (optional) add `--load_model model_relative_path` to continue training from a saved model
7. run src/delft/validate.py by default the validation is run on the last file added to src/delft/models/general_session/general_model
8. once the validation has run a rerun window will open automatically