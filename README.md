# proj1

- Robot Env Registration: wriggly_train.envs.wriggly.robot.wriggly_from_swimmer

- Mujoco Files: wriggly_train.envs.wriggly_mujoco

- DrQv2: wriggly_train.training (MUJOCO_GL=egl python train.py task=wriggly_move)

- Baselines - wriggly_train.training.baselines (ppo, ddpg, td3, sac)

- Evolutionary Strategies: 
- I'm working on integrating this with my envs, so far works with the standard suite
- - OpenAI-ES: scripts (./openai_es.sh) (train_openai_es.py)
- - TSA-ES: scripts (./tsaes) (train_tsaes.py). To run ARS within TSA-ES, lookahead and momentum both = 0.0
- - - Code in wriggly_train.tsaes and wriggly_train.tsaes_model

- Action Parametrizations: wriggly_train.training.{}
- - drqv2 and basic_sine
- - ijspeert (in progress)
- - kuramoto (in progress)
- - matsuoka_halfcenter (in progress)