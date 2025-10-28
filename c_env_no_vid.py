from demo import *
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

class DrivingEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 30
    }

    def __init__(self, map_path="miami_optimized.png", lidar_res=20, max_speed=10,
                 max_steering_angle=np.deg2rad(50), sectors=150, lad=30, render_mode="rgb_array",
                 realtime=False, norm_vel_input=False):
        
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
            obs_low = np.array([0.0]*lidar_res + [-1.0, -1.0])
            obs_high = np.array([1.0]*(lidar_res+2))
        else:
            # todo: we could try randomizing speed and msa to generalize better
            obs_low = np.array([0.0]*lidar_res + [-self.max_speed, -np.pi])
            obs_high = np.array([1.0]*lidar_res + [self.max_speed, np.pi])
        self.observation_space = spaces.Box(low=obs_low.astype(np.float32), high=obs_high.astype(np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.reset()


    def random_location(self, set=True):
        point_idx = np.random.randint(0, len(self.line))
        px, py = self.line[point_idx]*self.sim.mapm/self.sim.mapres
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
        v, w = action*[self.max_speed, self.max_steering_angle]
        self.pp.find_waypoint(self.sim.get_pose())

        self.sim.update(v, w)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self.sim.crashed or self.pp.laps > 3
        # if done:
        #     self.reset()
        return obs, reward-10.0*self.sim.crashed, done, False, {}

    def _get_obs(self):
        pose, (_, polars), _ = self.sim.get_state()
        dists = np.array([p[0] for p in polars], dtype=np.float32)/self.sim.lidar.max_steps
        car_info = np.array([self.sim.car.velocity, self.norm_angle(self.sim.car.theta)], dtype=np.float32)
        if self.nvi:
            car_info /= [self.max_speed, np.pi]
        return np.concatenate([dists, car_info])

    def _compute_reward(self):
        target = self.line[self.pp.curr_idx]
        pose = self.sim.get_pose()
        dist = np.linalg.norm(target - np.array(pose[:2]))
        reward = self.sim.car.velocity * 0.05 -dist * 0.1  # small penalty for distance, motivate speed
        # TODO: encourage lap time instead of raw speed.
        if self.prev_endpoint != self.pp.curr_idx:
            reward += 2.0
            self.hit_lms += 1
            self.prev_endpoint = self.pp.curr_idx
        return reward

    def render(self):
        img = self.sim.map.animate(pose=self.sim.get_pose(), show=False)
        for i, pt in enumerate(self.line):
            cv2.circle(img, tuple(pt.astype(int)), 2, (0, 0, 255) if i == self.pp.curr_idx else (255, 0, 0), -1)
        img = draw_controls_overlay(img, self.sim.car.velocity, self.sim.car.theta, self.max_speed, self.max_steering_angle)
        if self.render_mode == "human":
            cv2.imshow("Driving", img)
            cv2.waitKey(1)
        return img


class WandBVideoLogger(BaseCallback):
    def __init__(self, video_env, log_freq=5000, name="trackblitz", max_videos=3, verbose=0):
        super().__init__(verbose)
        self.video_env = video_env
        self.log_freq = log_freq
        self.name = name
        self.max_videos = max_videos
        self.video_count = 0
        self.video_path = f"videos/rollout_{self.num_timesteps}.mp4"
        self.lc = 1

    def _log(self) -> None:
        try:
            print("Logging to wandb...")
            wandb.log({f"reward0": self.locals["rewards"][0]})
            obs, _ = self.video_env.reset()
            init_size = self.video_env.render().shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w = cv2.VideoWriter(self.video_path, fourcc, 60.0, (init_size[1], init_size[0]))
            for n in range(10000):  # short rollout
                action, _ = self.model.predict(np.array(obs), deterministic=True)
                obs, _, done, *_ = self.video_env.step(action)
                frame = self.video_env.render()  # unwrap frame
                if frame.shape != init_size:
                    continue
                w.write(frame)
                if done:
                    print(n+1)  # , "crashed" if self.video_env.sim.crashed else "")
                    break
            w.release()
            wandb.log({
                f"{self.name}/video": wandb.Video(self.video_path, format="mp4")
            })
        except Exception as e:
            print("Failed to log:", e)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.lc * self.log_freq:
            self.video_path = f"videos/rollout_{self.lc * self.log_freq}.mp4"
            self.lc += 1
            self._log()
        return True


# ---- Configuration ----
epochs = 10_000_000
buf_size = 1_000_000
log_freq = 100
n_envs = 28



if __name__ == "__main__":
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

    def make_env(track=None, realtime=True):
        def f():
            env = DrivingEnv(["miami_optimized.png", "racetrack.png", "zandvoort.png"] if track is None else track, 
                            realtime=realtime, lidar_res=30,
                            norm_vel_input=True)
            env = Monitor(env)
            return env
        return f

    vid_env = DrivingEnv(["miami_optimized.png", "racetrack.png", "zandvoort.png"], realtime=False, lidar_res=30, norm_vel_input=True)

    callbacks = CallbackList([
        WandbCallback(
            model_save_path=f"runs/{run.id}/models",
            verbose=1
        ),
        EvalCallback(
            eval_env=DummyVecEnv([make_env("miami_optimized.png", realtime=False)]),
            best_model_save_path=f"./logs/{run.id}/best_model",
            log_path=f"./logs/{run.id}/results",
            eval_freq=log_freq,
            n_eval_episodes=1,
            deterministic=True,
            render=False,
        ),
        WandBVideoLogger(
            video_env=vid_env,
            log_freq=log_freq,
            name="car_racing"
        )
    ])
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    learn = True
    # ---- Model ----
    if learn:
        if os.path.exists("latest_finished_run.zip"):
            print("Loading preexisting model... ", end="")
            model = SAC.load(
                "latest_finished_run",
                env=env,
                verbose=2,
                buffer_size=buf_size,
                tensorboard_log=f"runs/{run.id}/tb",
                batch_size=512,
            )
            if os.path.exists("latest_finished_run_replay_buffer.pkl"):
                print("... and replay buffer.")
                model.load_replay_buffer("latest_finished_run_replay_buffer.pkl")
        else:
            model = SAC(
                "MlpPolicy",
                env,
                verbose=2,
                buffer_size=buf_size,
                tensorboard_log=f"runs/{run.id}/tb",
                batch_size=512,
            )
        
        model.learn(total_timesteps=epochs,
                    callback=callbacks,
                    log_interval=10000,
                    progress_bar=True
                    )
        model.save("latest_finished_run")
        model.save(f"runs/{run.id}/models/sac_car_racing_end")
        model.save_replay_buffer(f"runs/{run.id}/models/sac_car_racing_end_replay_buffer")
        model.save_replay_buffer(f"latest_finished_run_replay_buffer.pkl")

    def test_env():
        env = DrivingEnv(["miami_optimized.png", "racetrack.png", "zandvoort.png"], realtime=False, lidar_res=30, norm_vel_input=True)
        env = Monitor(env)
        return env
    env = DummyVecEnv([make_env()])
    model = SAC.load("latest_finished_run", env=env)
    run.finish()


    episodes = 10
    for ep in range(episodes):
        obs = env.reset()
        out = cv2.VideoWriter(f"test_ep-{ep}.mp4", fourcc, 60.0, env.render().shape[:2][::-1])
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            env_out = env.step(action)
            obs, reward, terminated, *_= env_out
            img = env.render()
            out.write(img)
            if terminated and i >= 500:
                obs = env.reset()
                break
        out.release()
