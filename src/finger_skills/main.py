'''
This file use to train/test the model using PPO
it only works for environment with continuous observation and action space
'''

import sys
import os

import gym
import torch

import ppo

# TODO: hyperparameters/test

def train(env, args):
    
    model = ppo.PPO(env, 0.005, 0.95, batch_size=4800, update_per_i=5,
                max_step_per_episode=1000, clip=0.2, show_every=20, seed=0, render=True)

    if args.mode == 'restart':
        # Tries to load in an existing actor/critic model to continue training on
        actor_model = '/home/jerry/Projects/finger_skills/src/finger_skills/model_state_dict/TODO'
        critic_model = '/home/jerry/Projects/finger_skills/src/finger_skills/model_state_dict/TODO'

        if actor_model != '' and critic_model != '':
            # if invalid path
            if not os.path.isfile(actor_model):
                print('acotor model path incorrect or not exists!')
                sys.exit(0)
            elif not os.path.isfile(actor_model):
                print('critic model path incorrect or not exists!')
                sys.exit(0)

            print(f"Loading in {os.path.basename(actor_model)} and {os.path.basename(critic_model)}...")
            model.actor.load_state_dict(torch.load(actor_model))
            model.critic.load_state_dict(torch.load(critic_model))
            print(f"Successfully loaded.")
        # Don't train from scratch if user accidentally forgets actor/critic model
        elif actor_model != '' or critic_model != '':
            print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
            sys.exit(0)
        else:
            print(f"Training from scratch.")

    model.learn(5)

    print('training done')


def test(env, args):
    pass


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
        hyperparameters = {}
        mode = 'train'  # or 'test

    args = Temp()
    main(args)
