from stable_baselines3 import PPO
from wriggly_train.envs.wriggly.robot import wriggly_from_swimmer
from dm_control import viewer
import cv2
import imageio
import dmc2gym

model = '/home/venky/proj1/wriggly_train/training/baselines/logs/ppo/2023-09-17/2023-09-17_17-28-29/best_model.zip'
model = PPO.load(model)

# env = wriggly_from_swimmer.move()
# env = dmc2gym.make(domain_name='wriggly', task_name='move', episode_length=2000, camera_id=5, height=480, width=600)
env = dmc2gym.make(domain_name='wriggly', task_name='move_no_time', episode_length=2000, camera_id=5, height=480, width=600)
obs, info = env.reset()


def my_policy(obs, ):
  return model.predict(obs, deterministic=False)[0]

def grabFrame(env):
    # Get RGB rendering of env
    rgbArr = env.physics.render(480, 600, camera_id=0)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

# viewer.launch(env, policy=my_policy)

frames = []

for j in range(1):
    total_reward = 0
    obs, info = env.reset()
    for i in range(2000):
        frame = env.render()
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if i % 10 == 0:
            print(action)
        frames.append(frame)
    print("Total Reward: ", total_reward)
imageio.mimwrite('wriggly_ppo.mp4', frames, fps=60)
env.close()