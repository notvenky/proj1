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

class MyActor(nn.Module):
    def __init__(self, num_actuators):
        super().__init__()
        self.num_actuators = num_actuators

        self.mu = nn.Parameter(torch.randn(num_actuators))  # intrinsic amplitude
        self.a = nn.Parameter(torch.randn(num_actuators))  # convergence factor
        self.w = nn.Parameter(torch.randn(num_actuators, num_actuators))  # weight matrix
        self.phi = nn.Parameter(torch.randn(num_actuators, num_actuators))  # phase bias matrix

        self.frequencies = nn.Parameter(torch.randn(num_actuators))

    def dynamics(self, r, r_dot, theta):
        r_dot_dot = self.a * (self.a / 4 * (self.mu - r) - r_dot)

        theta_dot = self.frequencies.clone()
        for i in range(self.num_actuators):
            print(r.shape)
            print(theta.shape)
            print(self.w.shape)
            print(self.phi.shape)

            for j in range(self.num_actuators):
                theta_dot[i] += torch.squeeze(r[j]) * torch.squeeze(self.w[i, j]) * torch.squeeze(torch.sin(theta[j] - theta[i] - self.phi[i, j]))


        return r_dot_dot, theta_dot
    
    def rk4_step(self, r, r_dot, theta, dt):
        r_dot_dot1, theta_dot1 = self.dynamics(r, r_dot, theta)
        k1_r = r_dot
        k1_r_dot = r_dot_dot1
        k1_theta = theta_dot1

        r_dot_dot2, theta_dot2 = self.dynamics(r, r_dot, theta)
        k2_r = r_dot
        k2_r_dot = r_dot_dot2
        k2_theta = theta_dot2

        r_dot_dot3, theta_dot3 = self.dynamics(r, r_dot, theta)
        k3_r = r_dot
        k3_r_dot = r_dot_dot3
        k3_theta = theta_dot3

        r_dot_dot4, theta_dot4 = self.dynamics(r, r_dot, theta)
        k4_r = r_dot
        k4_r_dot = r_dot_dot4
        k4_theta = theta_dot4

        r = r + dt / 6 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        r_dot = r_dot + dt / 6 * (k1_r_dot + 2 * k2_r_dot + 2 * k3_r_dot + k4_r_dot)
        theta = theta + dt / 6 * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
        print(r.shape)
        print(theta.shape)
        print(self.w.shape)
        print(self.phi.shape)
        return r, r_dot, theta

    def forward(self, obs, t, dt):
        b = t.shape[0]

        r = torch.zeros(b, self.num_actuators, device=t.device)
        r_dot = torch.zeros_like(r)
        theta = torch.zeros_like(r)

        for _ in range(10):
            r, r_dot, theta = self.rk4_step(r, r_dot, theta, dt)

        # Use r and theta to generate the command
        mu = r
        f, a, p = self.true_params()
        for i in range(mu.shape[-1]):
            mu[:, i] += a[i] * torch.sin(theta[i])


class MyCritic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
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
        if len(t.shape) != len(obs.shape):
            t = t.unsqueeze(-1) 
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
        # self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        # self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        # print(obs)
        joints = np.concatenate([obs['joints'], obs['jointvel']])
        t = torch.as_tensor(np.array([obs['time']]),device=self.device)
        obs = torch.as_tensor(joints, device=self.device)
        # obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, t, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1, 1) #was -1 to 1
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
            # self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
        self.critic_opt.step()
        # self.encoder_opt.step()

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