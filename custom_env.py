# TODO: the memory leaks??? Check video on c_env_no_vid (yes, ironic) to see if it matches.
# TODO: c_env_no_vid is much nicer on a lot of stuff, borrow from there. That being said, videos don't work on wandb.
# TODO: THOUGH they don't leak so that's a good path

#1--- TODO: We reworked the simulation, so now we need to port the changes over.

from demo import *
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.buffers import ReplayBuffer
import cv2, time, wandb
from os import makedirs
# from threading import Thread  # AGAIN. THIS DOES NOT WORK.

VIDEO_FOLDER = "./videos/" + time.strftime("%Y|%m|%d - %H:%M:%S") + "/"
DT = 1/60
makedirs(VIDEO_FOLDER, exist_ok=True)

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

    def __init__(self, map_path: str | tuple | list = "miami_optimized.png", render_mode="rgb_array",
                 lidar_res=20, sectors=150, lad=30,
                 max_speed=10, wheelbase=1.0, pp_max_speed=9.0,
                 realtime=False, norm_vel_input=True, dt=1/60):

        self.lidar_res = lidar_res
        self.max_speed = max_speed
        self.pp_max_speed = pp_max_speed
        self.lad = lad
        self.max_steering_angle = np.deg2rad(45)  # this is STANDARD from here on out.
        self.nvi = norm_vel_input
        self.runs = 0
        self.hit_lms = 0
        self.dt = dt

        self.multi = False
        if isinstance(map_path, list) or isinstance(map_path, tuple):
            self.sims = [Sim(m, race_neg=True,
                             lidar_res=lidar_res, max_speed=max_speed,
                             realtime=realtime, wheelbase=wheelbase, dt=dt) for m in map_path]
            self.lines = [discretize_line(s.full_centerline, sectors) for s in self.sims]
            self.sim = self.sims[0]
            self.multi = True
            self.line = self.lines[0]
        else:
            self.sim = Sim(map_path, race_neg=True,
                           lidar_res=lidar_res, max_speed=max_speed,
                           realtime=realtime, wheelbase=wheelbase, dt=dt)
            self.line = discretize_line(self.sim.full_centerline, sectors)

        self.pp = PurePursuit(self.line, lad, max_speed=pp_max_speed)
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
            self.sim.car.pose = np.array([px, py, theta])
        return px, py, theta

    def reset(self, seed=None, options=None):
        if self.multi:
            i = np.random.randint(0, len(self.sims))
            self.sim = self.sims[i]
            self.line = self.lines[i]
        self.random_location()
        self.pp = PurePursuit(self.line, self.lad, self.pp_max_speed)
        self.prev_endpoint = self.pp.find_waypoint(self.sim.get_pose())

        self.sim.crashed = False
        self.sim.car.speed = 0.0
        self.sim.car.steering = 0.0
        self.sim.v = np.array([0., 0., 0.])
        self.hit_lms = 0

        if self.sim.real_time:
            self.sim.stamp = time()
        return self._get_obs(), {}

    def step(self, action):
        # TODO: if we're changing the car's output, won't pure pursuit fight the residuals?
        # Because if we're steering away from the pp yaw, pp will want to correct, right?

        v, w = action * [0.2*self.max_speed, 0.2617994]  # add 20% of max speed, and 15 deg steering
        ppv, pps = self.pp(self.sim.get_pose())
        self.sim.update(v+ppv, w+pps)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self.sim.crashed or self.pp.laps > 5
        # if done:
        #     self.reset()
        return obs, reward, done, False, {}

    def _get_obs(self):
        pose, polars = self.sim.get_state(minimal=True)

        # TODO: the lidar class in sim is shit. rework it. we've done better than this. ~5 months ago.
        dists = np.array([p[0] for p in polars], dtype=np.float32) / self.sim.lidar.max_steps
        car_info = np.array([self.sim.car.speed, self.sim.car.steering], dtype=np.float32)
        if self.nvi:
            car_info /= [self.max_speed, self.max_steering_angle]
        return np.concatenate([dists, car_info])

    def _compute_reward_old(self):
        target = self.line[self.pp.curr_idx]
        pose = self.sim.get_pose()
        dist = np.linalg.norm(target - np.array(pose[:2]))
        reward = self.sim.car.speed * 0.1 - dist * 0.15  # small penalty for distance, motivate speed
        # TODO: encourage lap time instead of raw speed.
        if self.prev_endpoint != self.pp.curr_idx:
            reward += 5.0
            self.hit_lms += 1
            self.prev_endpoint = self.pp.curr_idx
        return reward

    def _compute_reward(self):
        pose = self.sim.get_pose()

        # Track progress more accurately
        if self.prev_endpoint != self.pp.curr_idx:
            reward = 10.0  # Larger reward for waypoint hits
            self.hit_lms += 1
            self.prev_endpoint = self.pp.curr_idx
        else:
            reward = 0.0

        # Speed incentive (scaled by normalized speed)
        speed_reward = (self.sim.car.speed / self.max_speed) * 0.5
        reward += speed_reward

        # Penalize lateral deviation
        target = self.line[self.pp.curr_idx]
        next_idx = (self.pp.curr_idx + 1) % len(self.line)
        next_point = self.line[next_idx]
        track_dir = next_point - target

        if not np.isclose(np.linalg.norm(track_dir), 0.0):
            car_vec = np.array(pose[:2]) - target
            lateral_error = np.abs(np.cross(track_dir, car_vec) / (np.linalg.norm(track_dir) + 1e-6))
            reward -= 0.5 * lateral_error  # Stronger lateral penalty

        # Penalize high steering angles (encourage smooth driving)
        steering_penalty = 0.1 * (self.sim.car.steering / self.max_steering_angle) ** 2
        reward -= steering_penalty

        # Terminal crash penalty
        if self.sim.crashed:
            reward -= 50.0  # Much larger penalty

        return float(reward)

    def render(self):
        img = self.sim.map.animate(pose=self.sim.get_pose(), show=False)
        for i, pt in enumerate(self.line):
            cv2.circle(img, tuple(pt.astype(int)), 2, (0, 0, 255) if i == self.pp.curr_idx else (255, 0, 0), -1)
        img = draw_controls_overlay(img, self.sim.car.speed, self.sim.car.steering, self.max_speed,
                                    self.max_steering_angle)
        return img


class WandBVideoLogger(BaseCallback):
    def __init__(self, video_env, log_freq=5000, name="trackblitz", max_videos=3, verbose=0):
        super().__init__(verbose)
        self.video_env = video_env
        self.log_freq = log_freq
        self.video_log_freq = log_freq*10
        self.name = name
        self.max_videos = max_videos
        self.video_count = 0
        self.video_name = f"rollout_{self.num_timesteps}.mp4"
        self.video_index = 1

    @property
    def video_path(self):
        return VIDEO_FOLDER + self.video_name

    def _log(self, log_video=True) -> None:
        try:
            print("Logging to wandb...")
            wandb.log({"landmark_count": env.envs[0].env.hit_lms})
            wandb.log({"reward": self.locals["rewards"][0]})
            lt = env.envs[0].env.pp.laptime
            if lt > 5:
                wandb.log({"laptime": lt})
            if log_video:
                print(f"[VideoLogger] Recording rollout #{self.video_index} at {self.num_timesteps} steps...")
                self._record()
                wandb.log({
                    f"{self.name}/video": wandb.Video(self.video_path, format="mp4")
                })
        except Exception as e:
            print("Failed to log:", e)

    def _record(self):
        obs, _ = self.video_env.reset()
        init_size = self.video_env.render().shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = cv2.VideoWriter(self.video_path, fourcc, int(1/DT), (init_size[1], init_size[0]))
        for n in range(10000):  # short rollout
            action, _ = self.model.predict(np.array(obs), deterministic=True)
            obs, _, done, *_ = self.video_env.step(action)
            frame = self.video_env.render()  # unwrap frame
            if frame.shape != init_size:
                continue
            w.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if done:
                print(n + 1)  # , "crashed" if self.video_env.sim.crashed else "")
                break
        w.release()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            self.video_name = f"rollout_{self.video_index:03d}.mp4"
            self.video_index += 1
            self._log(self.num_timesteps % self.video_log_freq == 0)

        return True


if __name__ == '__main__':
    # ---- Configuration ----
    epochs = 1_000_000
    buf_size = 10_000_000
    log_freq = 10_000
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": epochs,
        "env_name": "TrackBlitz"
    }
    
    # ---- Sim Parameters ----
    wheelbase = 1.25
    lookahead_distance = 25
    max_speed = 10
    pp_max_speed = max_speed*0.9
    run = wandb.init(
        project="gg_experiments",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    vid_env = DrivingEnv("./tracks/miami_optimized.png",
                         wheelbase=wheelbase, max_speed=max_speed,
                         lad=lookahead_distance, pp_max_speed=pp_max_speed,
                         dt=DT)
    
    callbacks = CallbackList([
        WandbCallback(
            model_save_path=f"runs/{run.id}/models",
            verbose=1
        ),
        WandBVideoLogger(
            video_env=vid_env,
            log_freq=log_freq,
            name="car_racing"
        )
    ])
    def make_env():
        env = DrivingEnv(map_path=("./tracks/miami_optimized.png",
                                   "./tracks/racetrack.png",
                                   "./tracks/zandvoort.png"),
                         wheelbase=wheelbase, max_speed=max_speed,
                         lad=lookahead_distance, pp_max_speed=pp_max_speed,
                         dt=DT)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # ---- Model ----
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=buf_size,
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
    env = DrivingEnv("./tracks/miami_optimized.png",
                         wheelbase=wheelbase, max_speed=max_speed,
                         lad=lookahead_distance, pp_max_speed=pp_max_speed,
                         dt=DT)
    for ep in range(episodes):
        obs, info = env.reset()
        for i in range(1000):
            action, states = model.predict(obs, deterministic=True)
            obs, reward, terminated, *_ = env.step(action)
            img = env.render()
    
            if terminated:
                obs, info = env.reset()
                break