# ############################################################
# '''
# import hydra
# import datetime
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # From CPG - RL
# # r_dot_dot[i] = a * (a/4 * (mu[i] - r[i]) - r_dot[i])


# class WrigglyCpg():
#     a = 150                 # Convergence factor - can change accordingly
#     def __init__(self,
#                  r,         # current aplitude of the oscillator
#                  mu,        # intrinsic amplitude of the oscillator 
#                  freq,      # frequency; can alternate between freq and time period
#                  w,         # w[i][j] weights
#                  phi        # phi[i][j] phase biases
#                  ):
#         self.r = r
#         self.mu = mu
#         self.freq = freq
#         self.w = w
#         self.phi = phi
#         print("Initiating CPG oscillators")

#     def oscillate_specfreq():
#         pass
#         # sample random amplitudes, frequencies, phase biases, and weights in order to see what propels the snake forward the best


# # r[i] -               current amplitude of the oscillator
# # a > 0 -           convergence factor
# # mu[i] -           intrinsic amplitude
# ############################################################
# # theta_dot[i] = freq[i] + summation(j) [r[j] * w[i][j] * sin(theta[j] - theta[i] - phi[i][j])]
# # theta[i] -        phase of the oscillator
# # freq[i] -         frequency
# # w[i][j] -         weights
# # phi[i][j] -       phase biases
# ############################################################
# # theta_dot[i] = freq[i]
# # phi_dot[i] = psi[i]
# ############################################################
# # Ranges of parameters selected at 100
# # mu -              [1, 2]
# # freq -            [0, 4.5]
# # psi -             [-1.5, 1.5] represents phase bias changes
# # a =               150
# ############################################################
# # From Salamander/Lamprey
# # 
# '''

import sys
sys.path.append("/home/venky/proj1")


from wriggly.simulation.robot import wriggly_from_swimmer
import hydra
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class MyActor(nn.Module):
    def __init__(self, frequencies, amplitudes, phases, num_actuators):
        super().__init__()

        # Internal constants and initializations
        self.MEAN_POSITION = 1024  # Arbitrary, you can modify this as needed from Code1
        self.dt = 0.001
        self.PI = np.pi

        # Actuator parameters
        self.frequencies = nn.Parameter(frequencies) 
        self.amplitudes = nn.Parameter(amplitudes)
        self.phases = nn.Parameter(phases)
        self.num_actuators = num_actuators
        self.range = torch.tensor([np.pi/2, np.pi, np.pi/2, np.pi, np.pi/2])

        # Initialize current amplitudes and rates of change
        self.current_amplitudes = torch.zeros(self.num_actuators)
        self.current_amplitude_dot = torch.zeros(self.num_actuators)
        
        self.apply(utils.weight_init)

    def differential_equation(self, y, dydt):
        mu = 1024
        c = 150
        d2ydt2 = c * (c / 4 * (mu - y) - dydt)
        return d2ydt2

    def rk4_step(self, y, dydt):
        k1 = self.dt * self.differential_equation(y, dydt)
        k2 = self.dt * self.differential_equation(y + 0.5 * dydt * self.dt, dydt + 0.5 * k1)
        k3 = self.dt * self.differential_equation(y + 0.5 * dydt * self.dt, dydt + 0.5 * k2)
        k4 = self.dt * self.differential_equation(y + dydt * self.dt, dydt + k3)

        dydt_new = dydt + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y_new = y + dydt * self.dt
        return y_new, dydt_new

    def true_params(self):
        frequencies = F.softplus(self.frequencies)
        amplitudes = F.tanh(self.amplitudes)
        phases = F.softplus(self.phases)
        amplitudes *= self.range
        return frequencies, amplitudes, phases

    def forward(self, obs, t, std):
        b = t.shape[0]
        mu = torch.zeros(b, self.num_actuators, device=t.device)
        f, a, p = self.true_params()

        # Apply oscillation with RK4 integration method
        for i in range(mu.shape[-1]): 
            self.current_amplitudes[i], self.current_amplitude_dot[i] = self.rk4_step(
                self.current_amplitudes[i],
                self.current_amplitude_dot[i]
                )
            mu[:, i] = self.MEAN_POSITION + self.current_amplitudes[i] * torch.sin(2 * self.PI * f[i] * t + p[i])

        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

class MyCritic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        # self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + 1 + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + 1 + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, t, action):
        #h = self.trunk(obs)
        h_action = torch.cat([obs, t, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)


        return q1, q2


class MyDrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        # self.encoder = Encoder(obs_shape).to(device)
        self.actor = MyActor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        # remove encoder

        self.critic = MyCritic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = MyCritic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        # self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        #self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        #self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode, t):
        obs = torch.as_tensor(obs, device=self.device)
        #obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, t, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.57, 1.57) #was -1 to 1
        return action.cpu().numpy()[0]

    def update_critic(self, obs, t, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs,t , stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, t, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, t, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        #self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        #self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, t, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, t, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, t, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics