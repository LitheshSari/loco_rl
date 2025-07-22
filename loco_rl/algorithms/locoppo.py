import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# this is different from conventional PPO storage
from loco_rl.modules.actor_critic_vae import ActorCriticVae
from loco_rl.modules.rnd import RandomNetworkDistillation
from loco_rl.storage import RolloutStorageVae as RolloutStorage


class LocoPPO:
    def __init__(
        self,
        policy: ActorCriticVae,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # VAE parameters
        kl_weight=1.0,
        vae_learning_rate=1e-3,
        num_adaptation_module_substeps=1,
    ):
        self.device = device

        # RND components
        if rnd_cfg is not None:
            # Extract learning rate and remove it from the original dict
            learning_rate = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.kl_weight = kl_weight
        self.vae_learning_rate = vae_learning_rate
        self.num_adaptation_module_substeps = num_adaptation_module_substeps

        # PPO components
        self.policy = policy
        self.policy.to(device)
        self.storage = None  # initialized later
        #! 这个用于 RL 优化
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.transition = RolloutStorage.Transition()

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        obs_history_shape = [self.policy.num_history, *actor_obs_shape]
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            actions_shape,
            self.device,
        )

    def test_mode(self):
        self.policy.eval()

    def train_mode(self):
        self.policy.train()

    def act(self, obs_history, privileged_obs):
        # Compute the actions and values
        obs = obs_history[:, -1, :]
        # TODO: 这里需要考虑 obs_history 的 shape, 并且这是一个硬编码，和 obs_term 顺序有关
        base_vel = privileged_obs[
            :, :3
        ]  # Assuming the first 3 dimensions are the base velocity
        self.transition.actions = self.policy.act(obs, obs_history).detach()
        self.transition.values = self.policy.evaluate(privileged_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.observation_histories = obs_history
        self.transition.privileged_observations = privileged_obs
        self.transition.base_vel = base_vel
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Prepare next observations: if done, use current obs; else, use next_obs from infos
        next_obs = infos["observations"]["next_obs"][:, -1, :].detach()
        self.transition.next_observations = torch.where(
            dones.unsqueeze(-1).bool(), self.transition.observations.clone(), next_obs
        )

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_entropy_loss = 0
        mean_surrogate_loss = 0
        mean_recons_loss = 0
        mean_vel_loss = 0
        mean_kld_loss = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            privileged_obs_batch,
            obs_history_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            base_vel_batch,
            dones_batch,
            next_obs_batch,
        ) in generator:
            #! Training  Student, encoder 是同时需要 RL 和 VAE
            self.policy.act(obs_batch, obs_history_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(privileged_obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Beta-VAE loss
            (
                code,
                code_vel,
                recon_est,
                mean_vel,
                logvar_vel,
                mean_latent,
                logvar_latent,
            ) = self.policy.cenet_forward(obs_history_batch)

            recons_loss = nn.MSELoss()(recon_est, next_obs_batch)
            vel_loss = nn.MSELoss()(code_vel, base_vel_batch)
            kld_loss = -0.5 * torch.sum(
                1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp()
            )
            mean_recons_loss += recons_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_kld_loss += kld_loss.item()

            autoenc_loss = recons_loss + vel_loss + self.kl_weight * kld_loss

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + autoenc_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_recons_loss /= num_updates * self.num_adaptation_module_substeps
        mean_vel_loss /= num_updates * self.num_adaptation_module_substeps
        mean_kld_loss /= num_updates * self.num_adaptation_module_substeps

        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy_loss,
            "recons_loss": mean_recons_loss,
            "vel_loss": mean_vel_loss,
            "kld_loss": mean_kld_loss,
        }
        return loss_dict
