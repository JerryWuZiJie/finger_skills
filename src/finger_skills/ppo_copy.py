import sys

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Network, self).__init__()
        self.layer0 = nn.Linear(in_dim, 64)
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        a0 = F.relu(self.layer0(obs))
        a1 = F.relu(self.layer1(a0))
        out = self.layer2(a1)

        return out

        # TODO: return F.softmax(out)


class PPO:
    def __init__(self, env, lr, gamma, batch_size, update_per_i, max_step_per_episode, clip, show_every, seed=None, render=True):
        if isinstance(seed, int):
            torch.manual_seed(seed)
            np.random.seed(seed=seed)
        torch.autograd.set_detect_anomaly(True)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # STEP 1: initialize actor critic
        self.actor = Network(self.obs_dim, self.act_dim)
        self.critic = Network(self.obs_dim, 1)
        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # initialize hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_per_i = update_per_i
        self.max_step_per_episode = max_step_per_episode
        self.clip = clip
        self.show_every = show_every
        self.render = render

        # coveriance matrix for query action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_iter):
        '''
        train the model
        '''

        current_iter = 0

        # STEP 2: in the loop
        while current_iter < total_iter:
            # STEP 3: collect set of trajectories
            batch_rews, batch_obs, batch_acts = self.rollout(current_iter)

            current_iter += 1

            # STEP 4: compute reward to go
            batch_rtg = self.compute_rtg(batch_rews)

            # STEP 5: compute advantage estimation, A_k = Q - V
            V = self.critic(batch_obs).squeeze()
            A_k = batch_rtg - V.detach()  # TODO: detach
            # normalize A_k
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # get previous log_prob for actor
            log_prob_k = self.get_logprob(batch_obs, batch_acts).detach()

            for _ in range(self.update_per_i):
                # STEP 6: update actor
                log_prob_cur = self.get_logprob(batch_obs, batch_acts)
                ratio = torch.exp(log_prob_cur - log_prob_k)
                clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)
                actor_loss = (-torch.min(ratio*A_k, clip*A_k)).mean()  # TODO: why mean()?

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # TODO: STEP 7: update critic inside the loop?
                current_V = self.critic(batch_obs).squeeze()
                critic_loss = nn.MSELoss()(current_V, batch_rtg)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

        # save at the end of the training
        torch.save(self.actor.state_dict(), './ppo_actor.pth')
        torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self, current_iter):
        '''
        collect batch_size data
        '''
        current_steps = 0
        first_round = True
        batch_rews = []
        batch_obs = []
        batch_acts = []

        # iterate the batch
        while current_steps < self.batch_size:
            reward_list = []
            obs = self.env.reset()
            done = False

            # iterate for one run
            for _ in range(self.max_step_per_episode):
                if first_round and (current_iter%self.show_every == 0):
                    if self.render:
                        self.env.render()
                    else:
                        first_round = False
                        print('-'*10, 'render')

                current_steps += 1

                batch_obs.append(obs)

                action = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                batch_acts.append(action)
                reward_list.append(rew)

                if done:
                    first_round = False
                    break

            batch_rews.append(reward_list)

        # turn list into tensor and return
        batch_rews = torch.tensor(batch_rews, dtype=torch.float)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)

        return batch_rews, batch_obs, batch_acts

    def get_action(self, obs):
        '''
        get action from sampling
        '''
        mean = self.actor(obs)
        distribution = torch.distributions.MultivariateNormal(
            mean, self.cov_mat)
        action = distribution.sample()

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy()  # TODO: why detach and numpy?

    def get_logprob(self, obs, act):
        '''
        calculate the log_prob(s) of given action(s) and observation(s)
        '''
        mean = self.actor(obs)
        distribution = torch.distributions.MultivariateNormal(
            mean, self.cov_mat)
        log_prob = distribution.log_prob(act)

        return log_prob

    def compute_rtg(self, rews_list):
        rtg = []

        # computer in backward order
        for rews in reversed(rews_list):
            discounted_rew = 0
            for rew in reversed(rews):
                discounted_rew = rew + self.gamma*discounted_rew
                rtg.append(discounted_rew)

        rtg.reverse()

        return torch.tensor(rtg, dtype=torch.float)


def main():
    actor_model = ''
    critic_model = ''

    env = gym.make('Pendulum-v0')
    model = PPO(env, 0.005, 0.95, batch_size=4800, update_per_i=5,
                max_step_per_episode=1000, clip=0.2, show_every=20, seed=0, render=True)
                
    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(
            f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    # Don't train from scratch if user accidentally forgets actor/critic model
    elif actor_model != '' or critic_model != '':
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    model.learn(5)


if __name__ == '__main__':
    main()
