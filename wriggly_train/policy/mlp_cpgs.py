from wriggly_train.policy.cpg_policy import CPGPolicy
import torch.nn as nn

class MLPDelta(CPGPolicy):
    def __init__(self, d_obs, num_actuators, frequencies=None, amplitudes=None, phases=None, init_std=0.5):
        super().__init__(num_actuators, frequencies, amplitudes, phases, init_std)
        self.offset_mlp = nn.Sequential(
            nn.Linear(d_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actuators),
        )

    def forward(self, obs):
        cpg_act = super().forward(obs)
        offset_act = self.offset_mlp(obs)
        return cpg_act + offset_act