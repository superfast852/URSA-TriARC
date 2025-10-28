# TODO: work on the simulation itself, make the car break more, make a better pure pursuit or more realistic params.
# TODO: the memory leaks??? Check video on c_env_no_vid (yes, ironic) to see if it matches.
# TODO: c_env_no_vid is much nicer on a lot of stuff, borrow from there. That being said, videos don't work on wandb.
# TODO: THOUGH they don't leak so that's a good path

from demo import *
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import os
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import glob
from stable_baselines3.common.buffers import ReplayBuffer


class HDRABuffer(ReplayBuffer):
    def __init__(self, *args, N=10, penalty_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        self.penalty_scale = penalty_scale

    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
        # if a crash occurred, retroactively penalize
        if done and reward < 0:
            for j in range(1, self.N + 1):
                idx = (self.pos - j) % self.buffer_size
                old = self.rewards[idx]
                self.rewards[idx] = old - (self.N - j) / self.N * abs(reward) * self.penalty_scale


class DrivingEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 60
    }

    def __init__(self, map_path: str | tuple | list = "miami_optimized.png", lidar_res=20, max_speed=7,
                 max_steering_angle=np.deg2rad(50), sectors=150, lad=30, render_mode="rgb_array",
                 realtime=False, norm_vel_input=True):

        self.lidar_res = lidar_res
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.nvi = norm_vel_input
        self.multi = False
        if isinstance(map_path, list) or isinstance(map_path, tuple):
            self.sims = [Sim(m, race_neg=True,
                             max_steering_angle=max_steering_angle,
                             lidar_res=lidar_res, max_speed=max_speed,
                             realtime=realtime) for m in map_path]
            self.lines = [discretize_line(s.full_centerline, sectors) for s in self.sims]
            self.sim = self.sims[0]
            self.multi = True
            self.line = self.lines[0]
        else:
            self.sim = Sim(map_path, race_neg=True,
                           max_steering_angle=max_steering_angle,
                           lidar_res=lidar_res, max_speed=max_speed,
                           realtime=realtime)
            self.line = discretize_line(self.sim.full_centerline, sectors)

        self.runs = 0
        self.hit_lms = 0
        self.lad = lad

        self.pp = PurePursuit(self.line, lad)
        self.prev_endpoint = self.pp.curr_idx
        self.norm_angle = lambda x: np.arctan2(np.sin(x), np.cos(x))
        self.render_mode = render_mode

        # Define observation space as lidar distances and current linear/angular speed
        if norm_vel_input:
            obs_low = np.array([0.0] * lidar_res + [-1.0, -1.0])
            obs_high = np.array([1.0] * (lidar_res + 2))
        else:
            # todo: we could try randomizing speed and msa to generalize better
            obs_low = np.array([0.0] * lidar_res + [-self.max_speed, -np.pi])
            obs_high = np.array([1.0] * lidar_res + [self.max_speed, np.pi])
        self.observation_space = spaces.Box(low=obs_low.astype(np.float32), high=obs_high.astype(np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.reset()

    def random_location(self, set=True):
        point_idx = np.random.randint(0, len(self.line))
        px, py = self.line[point_idx] * self.sim.mapm / self.sim.mapres
        theta = np.arctan2(*(self.line[(point_idx + 1) % len(self.line)]
                             - self.line[point_idx])[::-1])
        if set:
            self.sim.car.x = px
            self.sim.car.y = py
            self.sim.car.theta = theta
        return px, py, theta

    def reset(self, seed=None, options=None):
        if self.multi:
            i = np.random.randint(0, len(self.sims))
            self.sim = self.sims[i]
            self.line = self.lines[i]
        self.random_location()
        self.pp = PurePursuit(self.line, self.lad)
        self.prev_endpoint = self.pp.find_waypoint(self.sim.get_pose())
        self.sim.crashed = False
        self.sim.car.velocity = 0.0
        self.sim.car.tr = 0.0
        self.sim.car.omega = 0.0
        self.sim.car.prev_omega = 0.0
        self.sim.car.prev_steering_angle = 0.0

        self.hit_lms = 0

        if self.sim.real_time:
            self.sim.stamp = time()
        return self._get_obs(), {}

    def step(self, action):
        v, w = action * [0.2*self.max_speed, 0.2617994]  # add 20% of max speed, and 15 deg steering
        # self.pp.find_waypoint(self.sim.get_pose())
        ppv, pps = self.pp(self.sim.get_pose())
        self.sim.update(v+ppv, w+pps)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self.sim.crashed or self.pp.laps > 5
        # if done:
        #     self.reset()
        return obs, reward, done, False, {}

    def _get_obs(self):
        pose, (_, polars), _ = self.sim.get_state()
        dists = np.array([p[0] for p in polars], dtype=np.float32) / self.sim.lidar.max_steps
        car_info = np.array([self.sim.car.velocity, self.norm_angle(self.sim.car.theta)], dtype=np.float32)
        if self.nvi:
            car_info /= [self.max_speed, np.pi]
        return np.concatenate([dists, car_info])

    def _compute_reward_old(self):
        target = self.line[self.pp.curr_idx]
        pose = self.sim.get_pose()
        dist = np.linalg.norm(target - np.array(pose[:2]))
        reward = self.sim.car.velocity * 0.1 - dist * 0.15  # small penalty for distance, motivate speed
        # TODO: encourage lap time instead of raw speed.
        if self.prev_endpoint != self.pp.curr_idx:
            reward += 5.0
            self.hit_lms += 1
            self.prev_endpoint = self.pp.curr_idx
        return reward

    def _compute_reward(self):
        """
        Reward based on progress along the track centerline.
        Encourages forward motion and penalizes deviation and crashes.
        """
        pose = self.sim.get_pose()
        target = self.line[self.pp.curr_idx]
        next_idx = (self.pp.curr_idx + 1) % len(self.line)
        next_point = self.line[next_idx]

        # --- 1. Progress along centerline ---
        track_dir = next_point - target
        if np.isclose(np.linalg.norm(track_dir), 0.0):
            next_idx = (self.pp.curr_idx + 2) % len(self.line)
            next_point = self.line[next_idx]
            track_dir = next_point - target

        car_vec = np.array(pose[:2]) - target
        progress = np.dot(car_vec, track_dir) / np.linalg.norm(track_dir)
        reward = progress * 0.5  # proportional reward for forward progress

        # --- 2. Penalize lateral deviation ---
        lateral_error = np.abs(np.cross(track_dir, car_vec) / (np.linalg.norm(track_dir))+1e-6)
        reward -= 0.2 * lateral_error

        # --- 3. Speed shaping (encourage smooth but fast control) ---
        # reward += 0.05 * v
        # reward -= 0.02 * (steer ** 2)  # penalize high steering angles

        # --- 4. Terminal penalty ---
        if self.sim.crashed:
            reward -= 10.0

        return float(reward)

    def render(self):
        img = self.sim.map.animate(pose=self.sim.get_pose(), show=False)
        for i, pt in enumerate(self.line):
            cv2.circle(img, tuple(pt.astype(int)), 2, (0, 0, 255) if i == self.pp.curr_idx else (255, 0, 0), -1)
        img = draw_controls_overlay(img, self.sim.car.velocity, self.sim.car.theta, self.max_speed,
                                    self.max_steering_angle)
        # if self.render_mode == "human":
        #     cv2.imshow("Driving", img)
        #     cv2.waitKey(1)
        return img


class WandbVideoLogger(BaseCallback):
    def __init__(self, video_dir, log_freq=5000, name="trackblitz", max_videos=3, verbose=0):
        super().__init__(verbose)
        self.video_dir = video_dir
        self.log_freq = log_freq
        self.name = name
        self.max_videos = max_videos

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            print("logging video...")
            videos = sorted(glob.glob(os.path.join(self.video_dir, "*.mp4")), key=os.path.getmtime)
            for i, video_path in enumerate(videos[-self.max_videos:]):  # log last few
                wandb.log({f"{self.name}_video": wandb.Video(video_path, fps=30, format="mp4")})
        try:
            wandb.log({"landmark_count": env.env.envs[0].env.hit_lms})
            wandb.log({"reward": self.locals["rewards"][0]})
            lt = env.env.envs[0].env.pp.laptime
            if lt > 5:
              wandb.log({"laptime": lt})
        except:
            pass
        return True

# ---- Configuration ----
epochs = 100_000
buf_size = 1000
log_freq = 2500
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": epochs,
    "env_name": "TrackBlitz"
}

run = wandb.init(
    project="gg_experiments",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

callbacks = CallbackList([
    WandbCallback(
        model_save_path=f"runs/{run.id}/models",
        verbose=1
    ),
    WandbVideoLogger(
        video_dir=f"runs/{run.id}/videos",
        log_freq=log_freq,
        name="car_racing"
    )
])
def make_env():
    env = DrivingEnv(map_path=("./tracks/miami_optimized.png",
                               "./tracks/racetrack.png",
                               "./tracks/zandvoort.png"))
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
env = VecVideoRecorder(
    env,
    f"runs/{run.id}/videos",
    record_video_trigger=lambda x: x % log_freq == 0,
    video_length=1000,
)

# ---- Model ----
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=buf_size,
    tensorboard_log=f"runs/{run.id}/tb",
    replay_buffer_class=HDRABuffer,
    replay_buffer_kwargs={"N": 10, "penalty_scale": 1.0},
)

model.learn(total_timesteps=epochs,
            callback=callbacks,
            log_interval=log_freq,
            progress_bar=True
            )
model.save(f"runs/{run.id}/models/sac_car_racing_end")
run.finish()
episodes = 10
for ep in range(episodes):
    obs, info = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        img = env.render()

        if terminated or truncated:
            obs, info = env.reset()
            break