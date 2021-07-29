"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""

import numpy as np


def _log_summary(batch_lens, batch_rets, total_ep):
    """
            Print to stdout what we've logged so far in the most recent episode.

            Parameters:
                    None

            Return:
                    None
    """
    # calculate average
    batch_lens = np.mean(batch_lens)
    batch_rets = np.mean(batch_rets)

    # Round decimal places for more aesthetic logging messages
    batch_lens = str(round(batch_lens, 2))
    batch_rets = str(round(batch_rets, 2))

    # Print logging statements
    print(flush=True)
    print(
        f"-------------------- Total Episode #{total_ep} --------------------", flush=True)
    print(f"Avg Episodic Length: {batch_lens}", flush=True)
    print(f"Avg Episodic Return: {batch_rets}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy, env, episode, render):
    """
        Returns a generator to roll out each episode given a trained policy and
        environment to test on. 

        Parameters:
                policy - The trained policy to test
                env - The environment to evaluate the policy on
                render - Specifies whether to render or not

        Return:
                TODO
    """
    batch_lens = []
    batch_rews = []

    # Rollout
    for _ in range(episode):
        # print one * for each episode
        if _ % 50 == 0:
            # change line
            print()
        print('*', end='', flush=True)

        obs = env.reset()
        done = False

        # Logging data
        ep_len = 0            # episodic length
        ep_ret = 0            # episodic return

        while not done:
        # Track episodic length
            ep_len += 1

            # Render environment if specified, off by default
            if render:
                env.render()

            # Query deterministic action from policy and run it
            action = policy(obs).detach().numpy()
            obs, rew, done, _ = env.step(action)

            # Sum all episodic rewards as we go along
            ep_ret += rew

        # returns episodic length and return in this iteration
        batch_lens.append(ep_len)
        batch_rews.append(ep_ret)

    return batch_lens, batch_rews


def eval_policy(policy, env, episode=10, render=False):
    """
            The main function to evaluate our policy with. It will iterate a generator object
            "rollout", which will simulate each episode and return the most recent episode's
            length and return. We can then log it right after. And yes, eval_policy will run
            forever until you kill the process. 

            Parameters:
                    policy - The trained policy to test, basically another name for our actor model
                    env - The environment to test the policy on
                                        episode - Number of games to run
                    render - Whether we should render our episodes. False by default.

            Return:
                    None
    """
    # Rollout with the policy and environment, and log each episode's data
    batch_lens, batch_rews = rollout(policy, env, episode, render)
    _log_summary(batch_lens, batch_rews, episode)
