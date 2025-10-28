import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
import numpy as np


class MultiStepTDBuffer(ReplayBuffer):
    """
    Replay buffer with multi-step TD returns and HDRA.
    """

    def __init__(self, *args, n_steps=3, gamma=0.96, N=10, penalty_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma
        self.N = N
        self.penalty_scale = penalty_scale

    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)

        # HDRA: retroactively penalize recent transitions on crash
        if done and reward < 0:
            for j in range(1, min(self.N + 1, self.pos + 1)):
                idx = (self.pos - j) % self.buffer_size
                if self.full or idx < self.pos:
                    old = self.rewards[idx]
                    penalty = (self.N - j) / self.N * abs(reward) * self.penalty_scale
                    self.rewards[idx] = old - penalty

    def _compute_n_step_return(self, batch_indices):
        """
        Compute n-step returns for the given batch indices.
        Returns: n_step_rewards, n_step_next_obs, n_step_dones, actual_n
        """
        n_step_rewards = np.zeros(len(batch_indices), dtype=np.float32)
        n_step_next_obs = np.zeros((len(batch_indices), *self.obs_shape), dtype=np.float32)
        n_step_dones = np.zeros(len(batch_indices), dtype=np.float32)
        actual_n = np.zeros(len(batch_indices), dtype=np.int32)

        for i, idx in enumerate(batch_indices):
            cumulative_reward = 0.0
            discount = 1.0
            steps = 0

            for step in range(self.n_steps):
                current_idx = (idx + step) % self.buffer_size

                # Check if this transition exists and hasn't wrapped around
                if not self.full and current_idx >= self.pos:
                    break

                cumulative_reward += discount * self.rewards[current_idx, 0]
                discount *= self.gamma
                steps += 1

                # If terminal state, stop accumulating
                if self.dones[current_idx, 0]:
                    n_step_dones[i] = 1.0
                    break

            n_step_rewards[i] = cumulative_reward
            final_idx = (idx + steps) % self.buffer_size
            n_step_next_obs[i] = self.next_observations[final_idx]
            actual_n[i] = steps

        return n_step_rewards, n_step_next_obs, n_step_dones, actual_n

    def sample(self, batch_size: int, env=None):
        """
        Sample with multi-step TD computation.
        """
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size - self.n_steps,
                                            size=batch_size) + self.pos) % self.buffer_size
        else:
            # Only sample from valid indices that allow n-step lookahead
            upper = max(1, self.pos - self.n_steps)
            batch_inds = np.random.randint(0, upper, size=batch_size)

        # Get standard transitions
        data = self._get_samples(batch_inds, env=env)

        # Compute n-step returns
        n_step_rewards, n_step_next_obs, n_step_dones, actual_n = self._compute_n_step_return(batch_inds)

        # Create new ReplayBufferSamples with n-step values
        # We need to use _replace or create a new instance depending on SB3 version

        modified_data = ReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=self.to_torch(n_step_next_obs),
            dones=self.to_torch(n_step_dones.reshape(-1, 1)),
            rewards=self.to_torch(n_step_rewards.reshape(-1, 1)),
            n_steps=torch.from_numpy(actual_n).to(self.device)  # MODIFIED type_aliases.py from sb3 to add this field.
        )

        return modified_data


class MultiStepSAC(SAC):
    """
    SAC with multi-step TD learning.
    Requires using MultiStepTDBuffer as replay buffer.
    """

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Modified training to handle n-step returns properly.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Handle variable n-step discounts
            if hasattr(replay_data, 'n_steps'):
                # Compute gamma^n for each sample
                gamma_n = self.gamma ** replay_data.n_steps.float().unsqueeze(1)
            else:
                gamma_n = self.gamma

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

                # Modified TD target for n-step returns
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma_n * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # if gradient_step % self.target_update_interval == 0:
            #     self.polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))