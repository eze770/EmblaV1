import gymnasium as gym
import imageio
from gymnasium.wrappers import AddRenderObservation
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = AddRenderObservation(gym.make("Pusher-v5", render_mode="rgb_array", max_episode_steps=200), render_only=True)
env.reset()
#env.render()

#renderer = env.unwrapped.mujoco_renderer
#cam = renderer.viewer.cam
# cam.distance = 1.0
#print("Distance:", cam.distance)
#print("Azimuth:", cam.azimuth)
#print("Elevation:", cam.elevation)
#print("Lookat:", cam.lookat)

observationShape = env.observation_space.shape
actionSize = len(env.action_space.high.tolist())
print("actionSize: ", actionSize)
print(observationShape)
finalFilename = "test.mp4"
observations = np.empty((1000, *observationShape), dtype=np.float32)

idx = 0
terminated, truncated = False, False
while not terminated and not truncated:
    action = np.random.uniform(-1.5, 2.0, size=7)
    obs, reward, terminated, truncated, info = env.step(action)
    observations[idx] = obs
    qpos = env.unwrapped.data.qpos.copy()[:7]
    qvel = env.unwrapped.data.qvel.copy()
    idx = idx + 1

    timestep = env.unwrapped.model.opt.timestep
    frame_skip = env.unwrapped.frame_skip

    dt = timestep * frame_skip

print("qpos: ", qpos, "qvel: ", qvel, "dt: ", dt)
#observations = torch.as_tensor(observations[1], device=device).float()
#training_imges_snapshot = observations.view(-1, *observationShape)
#print(training_imges_snapshot)

#with imageio.get_writer(finalFilename, fps=30) as video:
 #   for ob in observations:
  #      video.append_data(ob)

env.close()
