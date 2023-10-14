from sb3_contrib import ARS
from wriggly_train.envs.wriggly.robots import wriggly_from_swimmer
from dm_control import viewer
import cv2
import imageio
import dmc2gymnasium
from datetime import datetime

import os
video_dir = 'videos/ars_videos'
os.makedirs(video_dir, exist_ok=True)

def grabFrame(env):
    rgbArr = env.physics.render(480, 600, camera_id=5)
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

# viewer.launch(env, policy=my_policy)
if __name__ == '__main__':
    model = '/home/venky/proj1/wriggly_train/training/baselines/logs/ars/2023-10-08/2023-10-08_15-58-04/best_model.zip'
    model = ARS.load(model)

    # env = wriggly_from_swimmer.move()
    # env = dmc2gym.make(domain_name='wriggly', task_name='move', episode_length=2000, camera_id=5, height=480, width=600)
    env = dmc2gymnasium.DMCGym('wriggly', 'approach_target', )
    obs, info = env.reset()

    frames = []

    for j in range(1):
        total_reward = 0
        obs, info = env.reset()
        done = False
        max_steps = 1000
        t = 0
        while not done and t < max_steps:
        # for i in range(2000):
            frame = env.render(height=480, width=600, camera_id=5)
            
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, trunc, info = env.step(action)
            print (t, reward, done)
            total_reward += reward
            frames.append(frame)
            t += 1
            
        print("Total Reward: ", total_reward)
    run_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    # import ipdb; ipdb.set_trace()
    substeps = env.unwrapped._env._n_sub_steps
    timestep = env.unwrapped._env.physics.timestep()
    fps = int(1 / (substeps * timestep))
    imageio.mimwrite(f'{video_dir}/{run_prefix}_wriggly_ars.mp4' , frames, fps=fps)
    env.close()