from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from loco_rl.utils import resolve_nn_activation

class ActorCriticVae(nn.Module):
    is_recurrent = False
    def __init__(self, 
                 num_actor_obs, 
                 num_privileged_obs, 
                 num_actions, 
                 num_latent, 
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 decoder_hidden_dims = [512, 256, 128],
                 activation="elu", 
                 init_noise_std=1.0,
                 num_history = 5,
                 noise_std_type: str = "scalar",
                    **kwargs
                 ):
        super().__init__()

        self.num_history = num_history
        activation = resolve_nn_activation(activation)

        # Actor Shape : num_obs + context_latent_z + context_latent_linear_vel
        actor_input_dim = num_actor_obs + num_latent + 3 
        critic_input_dim = num_privileged_obs
        decoder_output_dim = num_actor_obs
        context_latent_dim = num_latent + 3  # latent_z + linear_vel

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_history*num_actor_obs,128),
            activation,
            nn.Linear(128,64),
            activation,
        )
        self.encode_mean_latent = nn.Linear(64, num_latent)
        self.encode_logvar_latent = nn.Linear(64, num_latent)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(context_latent_dim, decoder_hidden_dims[0]))
        decoder_layers.append(activation)
        for layer_index in range(len(decoder_hidden_dims)):
            if layer_index == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[layer_index], decoder_output_dim))
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[layer_index], decoder_hidden_dims[layer_index + 1]))
                decoder_layers.append(activation)
        self.decoder = nn.Sequential(*decoder_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Encoder MLP: {self.encoder}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def cenet_forward(self, obs_history):
        obs_history = obs_history.reshape(obs_history.shape[0], -1)  # (bz, T, obs_dim) -> (bz, T * obs_dim)
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)

        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_mean_vel(distribution)
        code_latent = self.reparameterize(mean_latent,logvar_latent)
        code_vel = self.reparameterize(mean_vel,logvar_vel)
        code = torch.cat((code_vel,code_latent),dim=-1)
        decode = self.decoder(code)
        return code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self.std)

    def act(self, observations, obs_history, **kwargs):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code,observations), dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = obs_history[:, -1, :]
        observations = torch.cat((code, observations), dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
