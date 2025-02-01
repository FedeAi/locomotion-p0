# To create virt env where to install all the necessary (must be in locomotion-po)
python3.12 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt (just this)

# To use tensorboard
~/Scrivania/locomotion-p0$ tensorboard --logdir=logs/

# To run scripts
python GO1_train.py --run train
python GO1_train.py --run test --model_path ../models/rew_700_len_450_net_64_128_64_delta_action_joystick/best_model.zip