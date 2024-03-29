from typing import Union
from torch import Tensor
from torch.nn.modules.module import Module
from wriggly_train.envs.wriggly.robots import wriggly_from_swimmer
import hydra
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wriggly_train.training.utils as utils
from wriggly_train.policy.cpg_policy import CPGPolicy
from wriggly_train.policy.mlp_cpgs import MLPDelta
from wriggly_train.training.utils import dict_to_flat

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

class DrQv2Actor(nn.Module):
    def __init__(self, policy: CPGPolicy, init_std=0.5) -> None:
        super().__init__()
        print('actor init', policy)
        self.policy = policy
        self.init_std = init_std
    
    def forward(self, obs, t, std):
        # b = t.shape[0]
        # mu = torch.zeros(b,self.num_actuators,device= t.device)
        # f, a, p = self.true_params()
        # # Apply oscillation
        # for i in range(mu.shape[-1]):
        #     mu[:, i] += a[i] * torch.sin(2 * np.pi * f[i] * t + p[i])
        if len(obs.shape) == len(t.shape):
            new_obs = torch.cat([obs, t], dim=-1)
        elif len(obs.shape) == len(t.shape) - 1:
            new_obs = torch.cat([obs, t[0]])
        else:
            new_t = t.unsqueeze(1)
            print(obs.shape, new_t.shape)
            new_obs = torch.cat([obs, new_t], dim=1)
        mu = self.policy(new_obs)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist
    
    def __getattr__(self, name: str) -> Tensor | Module:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)
        # if name == 'policy':
        #     return self.policy
        # print('name', name)
        # print(self.policy)
        # import ipdb; ipdb.set_trace()

# class MyActor(nn.Module):
#     def __init__(self, num_actuators, frequencies=None, amplitudes=None, phases=None, init_std=0.5):
#         #repr_dim, action_shape, feature_dim, hidden_dim
#         super().__init__()

#         # self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
#         #                            nn.LayerNorm(feature_dim), nn.Tanh())

#         # self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
#         #                             nn.ReLU(inplace=True),
#         #                             nn.Linear(hidden_dim, hidden_dim),
#         #                             nn.ReLU(inplace=True),
#         #                             nn.Linear(hidden_dim, action_shape[0]))
        
#         # replace above nn with the parameters

#         # self.frequencies = nn.Parameter(torch.rand(num_actuators)) # softplus/exp/
#         # self.amplitudes = nn.Parameter(torch.rand(num_actuators))  # tanh activation
#         # self.phases = nn.Parameter(torch.rand(num_actuators))
#         if frequencies is None:
#             self.frequencies = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
#         else:
#             self.frequencies = nn.Parameter(frequencies) # softplus/exp/
#         if amplitudes is None:
#             self.amplitudes = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
#         else:
#             self.amplitudes = nn.Parameter(amplitudes) # softplus/exp/
#         if phases is None:
#             self.phases = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
#         else:
#             self.phases = nn.Parameter(phases) # softplus/exp/

#         self.std = nn.Parameter(torch.ones(num_actuators) * init_std)

#         self.num_actuators = num_actuators
#         self.apply(utils.weight_init)
#         self.range = torch.tensor([np.pi/2, np.pi, np.pi/2, np.pi, np.pi/2])

#     def underlying_params(self):
#         return self.frequencies, self.amplitudes, self.phases

#     def true_params(self):
#         frequencies = F.softplus(self.frequencies)
#         amplitudes = F.tanh(self.amplitudes)*self.range.to(self.amplitudes.device)
#         phases = F.softplus(self.phases)
#         return frequencies, amplitudes, phases
    
#     def get_true_params(self):
#         freq, amp, phase = self.true_params()
#         return {
#             'Frequency': freq.cpu().detach().numpy(),
#             'Amplitude': amp.cpu().detach().numpy(),
#             'Phase': phase.cpu().detach().numpy()
#         }
    
#     def print_true_params(self):
#         freq, amp, phase = self.true_params()
#         print("Frequency: ", freq)
#         print("Amplitude: ", amp)
#         print("Phase: ", phase)

#     def forward(self, obs, t, std):
#         b = t.shape[0]
#         mu = torch.zeros(b,self.num_actuators,device= t.device)
#         f, a, p = self.true_params()
#         # Apply oscillation
#         for i in range(mu.shape[-1]):
#             mu[:, i] += a[i] * torch.sin(2 * np.pi * f[i] * t + p[i])

#         std = torch.ones_like(mu) * std
#         dist = utils.TruncatedNormal(mu, std)
#         return dist

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

        # # # # transformed params from naive random search
        # frequency = torch.tensor([0.7287, 0.8134, 0.7327, 0.5979, 0.1452])
        # amplitude = torch.tensor([0.5174, 1.1551, 1.3948, 1.3612, 0.9059])
        # phase = torch.tensor([3.3565, 0.3632, 1.5797, 2.7197, 3.9647])

        # # phase = torch.tensor([ 4.0064, -3.1963,  2.3760,  1.4613,  3.3991])
        # # frequency = torch.tensor([-0.9120,  0.0916, -0.8205,  0.1050, -1.6120])
        # # amplitude = torch.tensor([0.9050, 0.8280, 2.0001, 1.3550, 0.5464])
        # self.actor = MyActor(action_shape[0],frequencies=frequency, amplitudes=amplitude, phases=phase).to(device)

        # for high vel
        # Frequency: tensor([0.3259, 0.3420, 0.0214, 0.4218, 0.3906]), Amplitude: tensor([1.4040, 0.7122, 1.1086, 1.1147, 0.9277]), Phase: tensor([1.2277, 2.6753, 5.6710, 3.5273, 5.7863])
        # frequency = torch.tensor([0.3259, 0.3420, 0.0214, 0.4218, 0.3906])
        # amplitude = torch.tensor([1.4040, 0.7122, 1.1086, 1.1147, 0.9277])
        # phase = torch.tensor([1.2277, 2.6753, 5.6710, 3.5273, 5.7863])

        # maxvel, base of 50
        # frequency = torch.tensor([ 0.0536, -1.5105, -0.0681, -0.3290, -0.3829])
        # amplitude = torch.tensor([1.0596, 1.6072, 1.2798, 0.3475, 0.5415])
        # phase = torch.tensor([1.0760, 1.2676, 0.9958, 3.2318, 2.1892])

        # '''1105: Frequency: tensor([0.7797, 0.8540, 0.7756, 0.8069, 0.7436]), Amplitude: tensor([0.4550, 1.7164, 1.5673, 2.5919, 0.7335]), Phase: tensor([0.1001, 2.1576, 3.0394, 1.2417, 3.0698])'''
        # frequency = torch.tensor([0.7797, 0.8540, 0.7756, 0.8069, 0.7436])
        # amplitude = torch.tensor([0.4550, 1.7164, 1.5673, 2.5919, 0.7335])
        # phase = torch.tensor([0.1001, 2.1576, 3.0394, 1.2417, 3.0698])

        '''
        Transformemd Phase:  tensor([1.8392, 5.8254, 4.2703, 6.1321, 2.1468])
        Transformemd Frequency:  tensor([ 0.4610, -0.6689,  0.3663,  0.1774, -0.1784])
        Transformemd Amplitude:  tensor([0.6211, 0.8757, 0.2556, 1.2622, 0.6168])  
        '''

        # frequency = torch.tensor([ 0.4610, -0.6689,  0.3663,  0.1774, -0.1784])
        # amplitude = torch.tensor([0.6211, 0.8757, 0.2556, 1.2622, 0.6168])
        # phase = torch.tensor([1.8392, 5.8254, 4.2703, 6.1321, 2.1468])


        frequency = torch.tensor([ 0.4610, -0.6689,  0.3663,  0.1774, -0.1784])
        amplitude = torch.tensor([0.6211, 3.14, 0.62, 3.14, 0.6168])
        phase = torch.tensor([1.8392, 5.8254, 4.2703, 6.1321, 2.1468])

        # self.actor = MyActor(action_shape[0]).to(device)
        # self.policy = CPGPolicy(action_shape[0],frequencies=frequency, amplitudes=amplitude, phases=phase).to(device)
        print('obs shape**********************************88')
        print(obs_shape)
        self.policy = MLPDelta(obs_shape, action_shape[0]).to(device)
        print('self.policy', self.policy)
        self.actor = DrQv2Actor(self.policy, init_std=0.5).to(device)

        self.policy.print_true_params()

        repr_dim = 10
        self.critic = MyCritic(repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = MyCritic(repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())


        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)



        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training

        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        # print(obs)
        # joints = np.concatenate([obs['joints'], obs['jointvel']])
        t_np = np.array([obs['time']]).copy()
        t = torch.as_tensor(t_np,device=self.device).float()
        del obs['time']
        o_np = dict_to_flat(obs)
        obs = torch.as_tensor(o_np, device=self.device).float()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, t, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1, 1)
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

        # optimize critic
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
        self.critic_opt.step()


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