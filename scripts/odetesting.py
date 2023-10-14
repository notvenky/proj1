from torchdiffeq import odeint
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

a = 150.0

def clone_t(a):
    return torch.cat(list((b.clone() for b in a)))

class TestModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.a = 150.0

    def forward(self, t, state):
        # TODO: add theta
        r, u, mu = state # [r, r_dot, mu, theta]
        dr = u
        du = self.a*(self.a / 4 * (mu - r) - u)
        # retirm dr, du (r_dot_dot), dmu, and dtheta
        return dr, du, torch.zeros_like(mu) 

    def simulate(self, dt, T, mu_func):
        t = torch.linspace(0, T, int(1/dt*T))
        state = (torch.tensor([1.0,]), torch.tensor([-0.2,]), mu_func(t[0]))
        all_states = [clone_t(state)]
        for cur_t in t:
            dt_ = torch.tensor([cur_t, cur_t + dt])
            # state[2] = mu_func(cur_t)
            state = (state[0], state[1], mu_func(cur_t))
            ret = odeint(self, state, dt_)
            state = ret[0][-1], ret[1][-1], ret[2][-1]
            # import ipdb; ipdb.set_trace()
            all_states.append(clone_t(state))

        return torch.stack(all_states), t


def mu_func(t):
    return torch.sin(t).unsqueeze(0)

model = TestModule()
ret, t = model.simulate(0.01, 10, mu_func)
rs = ret[:, 0]
plt.plot(t, rs[:-1])
plt.savefig("difeq.png")
# plt.show()
