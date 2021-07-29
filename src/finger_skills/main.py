'''
This file use to train/test the model using PPO
it only works for environment with continuous observation and action space
'''

import sys
import os

import gym
import torch

import ppo
import policy_network
import eval_policy


def train(env, args):
    print(f"Training")

    model = ppo.PPO(env, **args.hyperparameters)

    if args.mode != 'restart':
        # Tries to load in an existing actor/critic model to continue training on

        if args.actor_model != '' and args.critic_model != '':
            # if invalid path
            if not os.path.isfile(args.actor_model):
                print('acotor model path incorrect or not exists!')
                sys.exit(0)
            elif not os.path.isfile(args.actor_model):
                print('critic model path incorrect or not exists!')
                sys.exit(0)

            print(
                f"Loading in {os.path.basename(args.actor_model)} and {os.path.basename(args.critic_model)}...")
            model.actor.load_state_dict(torch.load(args.actor_model))
            model.critic.load_state_dict(torch.load(args.critic_model))
            print(f"Successfully loaded.")
        # Don't train from scratch if user accidentally forgets actor/critic model
        elif args.actor_model != '' or args.critic_model != '':
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
    if args.actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)
    elif not os.path.isfile(args.actor_model):
        print('acotor model path incorrect or not exists!')
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = policy_network.Network(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(args.actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy.eval_policy(
        policy=policy, env=env, episode=args.iteration, render=args.hyperparameters['render'])

    print('\n\nTesting done')


def main(args):
    # make environment and model
    env = gym.make('Pendulum-v0')

    if args.mode != 'test':
        train(env, args)
    else:
        test(env, args)

    env.close()


if __name__ == '__main__':
    class Temp:
        hyperparameters = {
            'timesteps_per_batch': 2048,
            'max_timesteps_per_episode': 1600,
            'gamma': 0.99,
            'n_updates_per_iteration': 5,
            'lr': 3e-4,
            'clip': 0.2,
            'render': False,
            'render_every_i': 10,
            'save_freq': 10,
            'seed': 0,
        }

        mode = 'test'  # train/restart/test
        iteration = 10  # iteration in train iterate through one batch, iteration in test iterate through one game

        actor_model = '/home/jerry/Projects/finger_skills/src/finger_skills/model_state_dict/ppo_actor.pth'
        critic_model = '/home/jerry/Projects/finger_skills/src/finger_skills/model_state_dict/ppo_critic.pth'

    args = Temp()
    main(args)
