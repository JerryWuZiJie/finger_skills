import os
import time

import torch
import torch.nn as nn
import numpy as np

import policy_network


class PPO:
    def __init__(self, env, actor_save_path, critic_save_path, **hyperparameters):
        # initialize hyperparameters
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # STEP 1: initialize actor critic
        self.actor = policy_network.ActorNetwork(self.obs_dim, self.act_dim)
        self.critic = policy_network.CriticNetwork(self.obs_dim, 1)
        # optimizer
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr)

        # criterion for critic loss
        self.mse = nn.MSELoss()

        # save model path
        # create path
        i = actor_save_path.rfind('/')
        save_path = actor_save_path[:i]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # store path
        self.a_path = actor_save_path
        self.c_path = critic_save_path

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

        # log
        self.logger = {
            'delta_t': time.time(),
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],     # losses of critic network in current iteration
        }

        # STEP 2: in the loop
        while self.logger['i_so_far'] < total_iter:
            # STEP 3: collect set of trajectories

            batch_rtg, batch_obs, batch_acts = self.rollout()

            # stack the data and create dataloader to draw mini batch
            batch_data = torch.hstack([batch_rtg, batch_obs, batch_acts])
            generator = torch.Generator(
                device='cuda' if torch.cuda.is_available() else 'cpu')
            train_batch = torch.utils.data.DataLoader(
                batch_data, batch_size=100, shuffle=True, generator=generator)

            self.logger['i_so_far'] += 1

            # train for mini batch
            for mini_batch in train_batch:
                mbatch_rtg = mini_batch[:, :1]
                mbatch_obs = mini_batch[:, 1:1+self.obs_dim]
                mbatch_acts = mini_batch[:, -self.act_dim:]

                # STEP 5: compute advantage estimation, A_k = Q - V
                V = self.critic(mbatch_obs)
                A_k = mbatch_rtg - V.detach()
                # normalize A_k
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                # get previous log_prob for actor
                log_prob_k = self.get_logprob(mbatch_obs, mbatch_acts).detach()

                for _ in range(self.n_updates_per_iteration):
                    # STEP 6: update actor
                    log_prob_cur = self.get_logprob(mbatch_obs, mbatch_acts)
                    ratio = torch.exp(log_prob_cur - log_prob_k)
                    clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)
                    actor_loss = (-torch.min(ratio*A_k, clip*A_k)).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    # STEP 7: update critic
                    current_V = self.critic(mbatch_obs)
                    critic_loss = self.mse(current_V, mbatch_rtg)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

                    # Log losses
                    self.logger['actor_losses'].append(actor_loss.detach())
                    self.logger['critic_losses'].append(critic_loss.detach())

            # print summary
            self._log_summary()

            if self.logger['i_so_far'] % self.save_freq == 0:
                # save state dict
                self.save_state_dict()

        # save at the end of the training
        self.save_state_dict()

    def save_state_dict(self):
        # save at the end of the training
        torch.save(self.actor.state_dict(), self.a_path)
        torch.save(self.critic.state_dict(), self.c_path)
        print('state dicts saved successfully')

    def rollout(self):
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
                if first_round and (self.logger['i_so_far'] % self.render_every_i == 0):
                    if self.render:
                        self.env.render()

                current_steps += 1

                batch_obs.append(obs)

                action = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                batch_acts.append(action)
                reward_list.append(rew)

                if done:
                    # increase reward if solve the environment
                    for i in range(len(reward_list)):
                        reward_list[i] += 100
                    break

            # disable rendering for rest of the batch
            if first_round:
                first_round = False

            batch_rews.append(reward_list)
            self.logger['batch_lens'].append(len(reward_list))

        # turn list into tensor and return
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)

        # STEP 4: compute reward to go
        batch_rtg = self.compute_rtg(batch_rews)

        # store log info
        self.logger['batch_rews'] = batch_rews

        return batch_rtg, batch_obs, batch_acts

    def get_action(self, obs):
        '''
        get action from sampling
        '''
        mean = self.actor(obs)
        cov_mat = torch.diag(self.actor.cov)
        distribution = torch.distributions.MultivariateNormal(
            mean, cov_mat)
        action = distribution.sample()

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.cpu().detach().numpy()

    def get_logprob(self, obs, act):
        '''
            calculate the log_prob(s) of given action(s) and observation(s)
        '''
        mean = self.actor(obs)
        cov_mat = torch.diag(self.actor.cov)
        distribution = torch.distributions.MultivariateNormal(
            mean, cov_mat)
        log_prob = distribution.log_prob(act)

        return log_prob

    def compute_rtg(self, rews_list):
        rtg = []

        # computer in backward order
        for rews in reversed(rews_list):
            discounted_rew = 0
            for rew in reversed(rews):
                discounted_rew = rew + self.gamma*discounted_rew
                rtg.append([discounted_rew])

        rtg.reverse()

        return torch.tensor(rtg, dtype=torch.float)

    def _log_summary(self):
        """
                Print to stdout what we've logged so far in the most recent batch.

                Parameters:
                        None

                Return:
                        None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time()
        delta_t = (self.logger['delta_t'] - delta_t)

        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews)
                              for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().cpu().mean()
                                 for losses in self.logger['actor_losses']])
        avg_critic_losses = np.mean([losses.float().cpu().mean()
                                     for losses in self.logger['critic_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_losses = str(round(avg_critic_losses, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
