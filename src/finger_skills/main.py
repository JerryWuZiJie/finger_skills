'''
This file use to train/test the model using PPO
it only works for environment with continuous observation and action space
'''

import sys
import os
import pickle
import time

import gym
import torch

import ppo
import policy_network
import eval_policy
import env_finger


DT = 0.01
MAX_TIMESTEPS_PER_EPISODE = int(2/DT)  # 2s simulation
TIMESTEPS_PER_BATCH = MAX_TIMESTEPS_PER_EPISODE * 10  # 10 game in each iteration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ITERATION = 100
MODE = 0

RENDER = False
if MODE == 0:
    MODE = 'restart'
elif MODE == 1:
    MODE = 'train'
elif MODE == 2:
    MODE = 'test'
    RENDER = True
elif MODE == 3:
    MODE = 'check'
    RENDER = True
else:
    print('invalid mode')
    sys.exit(0)


def train(env, args):
    print(f"Training")

    model = ppo.PPO(env, args.model_path, **args.hyperparameters)

    if args.mode != 'restart':
        # Tries to load in an existing actor/critic model to continue training on
        actor_model = os.path.join(
            args.model_path, 'ppo_actor.pth')  # actor state dict
        critic_model = os.path.join(
            args.model_path, 'ppo_critic.pth')  # critic state dict

        if actor_model != '' and critic_model != '':
            # if invalid path
            if not os.path.isfile(actor_model):
                print('acotor model path incorrect or not exists!')
                sys.exit(0)
            elif not os.path.isfile(critic_model):
                print('critic model path incorrect or not exists!')
                sys.exit(0)

            print(
                f"Loading in {os.path.basename(actor_model)} and {os.path.basename(critic_model)}...")
            model.actor.load_state_dict(torch.load(actor_model, map_location=torch.device(DEVICE)))
            model.critic.load_state_dict(torch.load(critic_model, map_location=torch.device(DEVICE)))
            print(f"Successfully loaded state dicts.")
        # Don't train from scratch if user accidentally forgets actor/critic model
        elif actor_model != '' or critic_model != '':
            print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
            sys.exit(0)
        else:
            print(f"Training from scratch.")

    model.learn(args.iteration)

    print('\n\nTraining done')


def test(env, args):
    """
            Tests the model.

            Parameters:
                    env - the environment to test the policy on
                    actor_model - the actor model to load in

            Return:
                    None
    """
    print(f"Testing")

    # If the actor model is not specified, then exit
    if args.model_path == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)
    else:
        actor_model = os.path.join(
            args.model_path, 'ppo_actor.pth')  # actor state dict
        if not os.path.isfile(actor_model):
            print('acotor model path incorrect or not exists!')
            sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = policy_network.ActorNetwork(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(
        actor_model, map_location=torch.device(DEVICE)))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy.eval_policy(
        policy=policy, env=env, render=args.hyperparameters['render'])

    print('\n\nTesting done')


def check(env, args):
    print("Checking")

    if args.model_path == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)
    else:
        action_folder = os.path.join(
            args.model_path, 'action')  # actor state dict
        if not os.path.exists(action_folder):
            print('action folder path incorrect or not exists!')
            sys.exit(0)

    for f in os.listdir(action_folder):
        print('-'*10, 'Rendering', f, '-'*10)
        with open(os.path.join(action_folder, f), 'rb') as f:
            env.reset()
            time.sleep(1)

            action_list = pickle.load(f)
            env.render()
            for action in action_list:
                env.step(action)
                env.render()

    print('\n\nChecking done')


def main(model_dir="/home/jerry/Projects/finger_skills/src/finger_skills/"):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('On GPU')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        print('On CPU')

    class Temp:
        hyperparameters = {
            'timesteps_per_batch': TIMESTEPS_PER_BATCH,
            'max_timesteps_per_episode': MAX_TIMESTEPS_PER_EPISODE,
            'gamma': 0.95,
            'n_updates_per_iteration': 20,
            'lr': 5e-3,
            'clip': 0.2,
            'render': RENDER,
            'render_every_i': 1,
            'save_freq': 5,
            'seed': 0,
        }

        mode = MODE  # train/restart/test
        iteration = ITERATION  # iteration in train iterate through one batch, iteration in test iterate through one game

        model_path = os.path.join(model_dir, 'model_info')

    args = Temp()

    # make environment and model
    env = env_finger.EnvFingers(render=args.hyperparameters['render'], dt=DT)

    if args.mode == 'check':
        check(env, args)
    elif args.mode == 'test':
        test(env, args)
    else:
        train(env, args)

    env.close()


if __name__ == '__main__':
    main()
