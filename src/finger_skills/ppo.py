import os

import torch
import torch.nn as nn
import numpy as np

import policy_network


save_path = os.path.join(
    "/home/jerry/Projects/finger_skills/src/finger_skills", 'model_state_dict')
if not os.path.exists(save_path):
    os.mkdir(save_path)


class PPO:
    def __init__(self, env, lr, gamma, batch_size, update_per_i, max_step_per_episode, clip, show_every, seed=None, render=True, **hyperparameters):
        # initialize hyperparameters
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # STEP 1: initialize actor critic
        self.actor = policy_network.Network(self.obs_dim, self.act_dim)
        self.critic = policy_network.Network(self.obs_dim, 1)
        # optimizer
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr)

        # coveriance matrix for query action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters

        Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                                        hyperparameters defined below with custom values.

        Return:
                None
        """
        # Initialize default values for hyperparameters ----

        # Algorithm hyperparameters --

        # Number of timesteps to run per batch
        self.timesteps_per_batch = 4800
        # Max number of timesteps per episode
        self.max_timesteps_per_episode = 1600
        # Number of times to update actor/critic per iteration
        self.n_updates_per_iteration = 5
        self.lr = 0.005                                 # Learning rate of actor optimizer
        # Discount factor to be applied when calculating Rewards-To-Go
        self.gamma = 0.95
        # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.clip = 0.2

        # Miscellaneous parameters --

        # If we should render during rollout
        self.render = True
        self.render_every_i = 10                        # Only render every n iterations
        # How often we save in number of iterations
        self.save_freq = 10
        # Sets the seed of our program, used for reproducibility of results
        self.seed = None

        # Change any default values to custom values for specified hyperparameters ----
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            np.random.seed(seed=self.seed)
            print(f"Successfully set seed to {self.seed}")

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

            for _ in range(self.n_updates_per_iteration):
                # STEP 6: update actor
                log_prob_cur = self.get_logprob(batch_obs, batch_acts)
                ratio = torch.exp(log_prob_cur - log_prob_k)
                clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)
                actor_loss = (-torch.min(ratio*A_k, clip*A_k)
                              ).mean()  # TODO: why mean()?

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # TODO: STEP 7: update critic inside the loop?
                current_V = self.critic(batch_obs).squeeze()
                critic_loss = nn.MSELoss()(current_V, batch_rtg)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            
            if current_iter % self.save_freq == 0:
                self.save_state_dict()

        # save at the end of the training
        self.save_state_dict()

    def save_state_dict(self):
        # save at the end of the training
        torch.save(self.actor.state_dict(),
                   os.path.join(save_path, 'ppo_actor.pth'))
        torch.save(self.critic.state_dict(),
                   os.path.join(save_path, 'ppo_critic.pth'))
        print('state dicts saved successfully')

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
        while current_steps < self.timesteps_per_batch:
            reward_list = []
            obs = self.env.reset()
            done = False

            # iterate for one run
            for _ in range(self.max_timesteps_per_episode):
                if first_round and (current_iter % self.render_every_i == 0):
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
