# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.append("/home/venky/proj1")


from wriggly_train.envs.wriggly.robot import wriggly_from_swimmer
import hydra
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wriggly_train.training.utils as utils


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
    def __init__(self, num_actuators, frequencies=None, amplitudes=None, phases=None, init_std=0.5):
        #repr_dim, action_shape, feature_dim, hidden_dim
        super().__init__()

        # self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())

        # self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(hidden_dim, hidden_dim),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(hidden_dim, action_shape[0]))
        
        # replace above nn with the parameters

        # self.frequencies = nn.Parameter(torch.rand(num_actuators)) # softplus/exp/
        # self.amplitudes = nn.Parameter(torch.rand(num_actuators))  # tanh activation
        # self.phases = nn.Parameter(torch.rand(num_actuators))
        if frequencies is None:
            self.frequencies = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
        else:
            self.frequencies = nn.Parameter(frequencies) # softplus/exp/
        if amplitudes is None:
            self.amplitudes = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
        else:
            self.amplitudes = nn.Parameter(amplitudes) # softplus/exp/
        if phases is None:
            self.phases = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
        else:
            self.phases = nn.Parameter(phases) # softplus/exp/

        self.std = nn.Parameter(torch.ones(num_actuators) * init_std)

        self.num_actuators = num_actuators
        self.apply(utils.weight_init)
        self.range = torch.tensor([np.pi/2, np.pi, np.pi/2, np.pi, np.pi/2])

    def underlying_params(self):
        return self.frequencies, self.amplitudes, self.phases

    def true_params(self):
        frequencies = F.softplus(self.frequencies)
        amplitudes = F.tanh(self.amplitudes)*self.range.to(self.amplitudes.device)
        phases = F.softplus(self.phases)
        return frequencies, amplitudes, phases

    def forward(self, obs, t, std):
        # h = self.trunk(obs)
    
        # mu = self.policy(h) #init as zeros
        # mu = torch.tanh(mu)
        # std = torch.ones_like(mu) * std

        # dist = utils.TruncatedNormal(mu, std)
        b = t.shape[0]
        mu = torch.zeros(b,self.num_actuators,device= t.device)
        f, a, p = self.true_params()
        # Apply oscillation
        for i in range(mu.shape[-1]):
            mu[:, i] += a[i] * torch.sin(2 * np.pi * f[i] * t + p[i])

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
        if len(t.shape) != len(obs.shape):
            t = t.unsqueeze(-1) 
        # print(obs.shape, t.shape, action.shape)
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
        # print(action_shape)
        self.actor = MyActor(action_shape[0]).to(device)
        # remove encoder
        repr_dim = 10
        self.critic = MyCritic(repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = MyCritic(repr_dim, action_shape,
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

    def act(self, obs, step, eval_mode):
        # print(obs)
        joints = np.concatenate([obs['joints'], obs['jointvel']])
        t = torch.as_tensor(np.array([obs['time']]),device=self.device)
        obs = torch.as_tensor(joints, device=self.device)
        #obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, t, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-2, 2) #was -1 to 1
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
        with torch.autograd.set_detect_anomaly(True):
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
        with torch.autograd.set_detect_anomaly(True):
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
        obs, t, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        # obs = self.aug(obs.float())
        # next_obs = self.aug(next_obs.float())
        # # encode
        # obs = self.encoder(obs)
        # with torch.no_grad():
        #     next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, t, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), t, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics