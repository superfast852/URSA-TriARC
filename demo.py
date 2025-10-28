import numpy as np
from NavStack import Map, RRT
import cv2
import math
from time import sleep, time, strftime
from numba import njit, typed
from TrackParsing import parse_trackimg, discretize_line

"""
STABLE CONFIGURATIONS:
max_speed=10, lad=25, L=1.25, steering_bias=1.15, pp_max=0.9  # For training, set steering_bias=1.15
lap_time = 16.6
"""

pmap = map
inbounds = lambda x, y, m: 0 <= x < m.shape[1] and 0 <= y < m.shape[0]


def curvature(p_prev, p, p_next, axis=0):
    a = np.linalg.norm(p - p_prev, axis=axis)
    b = np.linalg.norm(p_next - p, axis=axis)
    c = np.linalg.norm(p_next - p_prev, axis=axis)
    s = (a + b + c) / 2
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 1e-10))
    return 4 * area / (a * b * c)

def segment_curvature(line):
    return curvature(np.roll(line, -1, axis=0), line, np.roll(line, 1, axis=0), axis=1)

def speed_limit(mu, curv):
    return np.sqrt(np.maximum(mu * 9.81 / (np.abs(curv) + 1e-10), 0))


class AckermannSteering:
    """
    Simulates an Ackermann steering robot in real-life conditions.
    """
    def __init__(self, wheelbase=2, max_steering_angle=np.deg2rad(45),
                 acceleration=4, max_speed=10, braking=4, dt=1/60, fric=1.55):
        """
        Initializes the simulation class.

        Args:
            wheelbase (float): Distance between the front and rear axles (meters).
            max_steering_angle (float): Maximum allowed steering angle for the servo (radians).
            dt (float, optional): Simulation time step (seconds). Defaults to 1/60.
        """
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
        self.steer_rate = 2*np.pi
        self.dt = dt
        self.a = acceleration
        self.ms = max_speed
        self.braking_force = braking
        self.mu = fric

        # Robot state (x, y, heading)
        self.pose = np.array([0.0, 0.0, 0.0])
        self.speed = 0.0
        self.steering = 0.0
        self.v = np.array([0.0, 0.0, 0.0])

        # constants
        self.Cd = 0.35  # drag coefficient
        self.A = 0.05  # frontal area (m^2)
        self.rho = 1.225  # air density
        self.max_g = 0.0

    def calculate_acceleration(self, throttle):
        # Normal load (simplified)
        Fz = 2 * 9.81

        # Tire-road friction limit (longitudinal)
        Fmax = self.mu * Fz * 2.0  # both axles

        # Compute traction/braking forces
        if throttle >= 0:
            F_drive = throttle * self.a * 4.0
            F_brake = 0.0
        else:
            F_drive = 0.0
            F_brake = -throttle * self.braking_force * 4.0

        # Aerodynamic drag
        F_drag = 0.5 * self.rho * self.Cd * self.A * self.speed ** 2

        # Net force (limited by friction circle)
        Fx = np.clip(F_drive - F_brake - F_drag, -Fmax, Fmax)

        # Acceleration
        return Fx / 4.0

    def simple_accel(self, throttle, dt):
        # --- Apply longitudinal friction
        sign = self._sign(throttle)
        if abs(throttle) < abs(self.speed):
            if abs(self.speed) < 0.05:
                self.speed = 0.0
            else:
                self.speed -= self.braking_force * dt * sign # Avoid overshooting past 0
        elif abs(throttle) > 0.01:
            self.speed += (self.a * dt * sign)
            # * (2.0 if self._sign(target_speed) != self._sign(self.speed) else 1.0))

    @staticmethod
    def _sign(a):
        return -1.0 if a < 0.0 else 1.0

    def update(self, speed, steering_angle, dt=0.0):
        """
        Updates the robot's position and heading based on speed and steering angle.

        Args:
            speed (float): Linear speed of the robot (meters/second).
            steering_angle (float): Desired steering angle for the servo (radians).
                Clamped to the maximum allowed steering angle.
        """
        if dt == 0.0:
            dt = self.dt

        target_speed = np.clip(speed, -self.ms, self.ms)
        self.simple_accel(target_speed, dt)

        # Clamp inputs within limits
        self.speed = np.clip(self.speed, -self.ms, self.ms)
        self.steering = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        # Update robot's heading (considering previous steering for smoother transitions)
        self.v = np.array([np.cos(self.pose[2]),
                           np.sin(self.pose[2]),
                           np.tan(self.steering) / self.wheelbase]) * self.speed
        # Lateral acceleration
        # sideslip = np.arctan(np.tan(self.steering) / 2.0)
        # lateral_velocity = self.speed*np.sin(sideslip)

        self.pose += self.v*dt
        self.pose[2] %= math.pi * 2
        # print(self.pose, np.tan(self.steering), self.dt)

    def get_pose(self, px=1, m=1):
        """
        Returns the current robot pose (x, y, heading) in meters and radians.
        """
        p = px/m
        pose = self.pose.copy()
        pose[:2] *= p
        return pose.tolist()

    def get_state(self):
        return [self.pose, self.v, self.steering]


class LIDAR:
    def __init__(self, map, res=180):
        self.res = res
        self.map = map
        self.convert = lambda x: (x[2], x[3])
        self.max_steps = int(map.map.shape[0]*(2**(1/2)))   # expected to be a 2d plane.
        self.cast_ray(map.getValidPoint(), 0)  # to compile the function

    @staticmethod
    @njit
    def _cast_ray(map: np.array, start_pos: typed.List, angle: float, max_distance: int) -> tuple:
        count = 0
        bases = (math.cos(angle), math.sin(angle))
        while True:
            x, y = start_pos[0] + (count * bases[0]), start_pos[1] + (count * bases[1])
            if x < 0 or y < 0 or x > map.shape[1] or y > map.shape[0]:
                return None  # Goes into the void.
            if map[int(y), int(x)]:
                return x, y, count, angle  # Found an obstacle. return coordinates and distance
            count += 1
            if count > max_distance:
                return None  # Exceeded max distance

    def cast_ray(self, start_pos, angle):
        return self._cast_ray(self.map.map, typed.List(start_pos), angle, self.max_steps)

    def get_scan(self, position):
        l = np.linspace(0 + position[2], 2 * math.pi + position[2], self.res)
        xy = position[:2]

        if not inbounds(*xy, self.map.map) or not self.map.isValidPoint(xy):
            if len(l) > self.res:
                l = l[:self.res]
            return [xy], np.vstack(([0]*self.res, l)).T

        rays = [self.cast_ray(xy, i) for i in l]
        # These are the xy points. I think we need to convert them to polar coordinates.
        rays = list(filter(lambda x: x is not None, rays))
        polars = list(pmap(self.convert, rays))
        points = list(pmap(lambda x: x[:2], rays))
        return points, polars


class Sim:
    def __init__(self, map="racetrack.png", race_neg=False, map_m=35,
                    lidar_res=20, realtime=True, dt=1/60, full_stop=False,
                 **kwargs):
        # TODO: add resizing of map images :p
        # future me here. what??
        self.full_centerline = np.empty((0, 2))
        if isinstance(map, str) and "." in map and map.split(".")[-1] in ["jpeg", "jpg", "png"]:
            map, self.full_centerline = parse_trackimg(map, race_neg)
            map = map.astype(np.float32)
            map /= 255.0
            map = 1-map
        self.map = Map(map, map_m)
        self.real_time=realtime
        self.mapres = self.map.map.shape[0]
        self.mapm = self.map.map_meters
        self.fs = full_stop

        self.car = AckermannSteering(dt = None if realtime else dt, **kwargs)
        self.lidar = LIDAR(self.map, lidar_res)

        self.dt = 0 if self.real_time else dt
        self.stamp = time()
        self.scan = None
        pt = self.map.getValidPoint()
        self.car.pose[0] = pt[0] * self.mapm / self.mapres
        self.car.pose[1] = pt[1] * self.mapm / self.mapres
        self.prev_pose = self.car.pose[:2].copy()
        self.crashed = False


    def get_state(self, minimal=False):
        pose = self.get_pose()

        pose[0] = int(pose[0])
        pose[1] = int(pose[1])
        self.scan = self.lidar.get_scan(pose)
        if minimal:
            return pose, self.scan[1]
        frame, polars = self.scan
        frame = np.clip(np.array(frame, dtype=int) - pose[:2] + self.map.map_center, 0, self.mapres-1)[:, ::-1].T.tolist()
        lidar_view = np.zeros((self.mapres, self.mapres, 3), np.uint8)
        lidar_view[frame[0], frame[1]] = (255, 255, 255)
        cv2.arrowedLine(lidar_view, *self.map._posearrowext([*self.map.map_center, pose[2]], 10), (0, 255, 0), 2)
        return pose, self.scan, lidar_view

    def update(self, v, w):
        self.dt = time() - self.stamp if self.real_time else self.dt
        self.stamp = time()
        self.car.update(v, w, self.dt)
        self.crashed = self.hasCrashed()
        if self.crashed:  # crashed
            self.car.pose[:2] = self.prev_pose
            if self.fs:
                self.car.v = np.zeros(3)
        else:
            self.prev_pose = self.car.pose[:2].copy()

    def hasCrashed(self):
        return not self.map.isValidPoint(np.array(self.get_pose()[:2], dtype=int))
    
    def get_pose(self):
        return self.car.get_pose(self.mapres, self.mapm)


class PurePursuit:
    def __init__(self, path, lad, max_speed=10.0, steering_bias=1.0, print_laptime=False):
        self.p = path
        self.lad = lad
        self.ms = max_speed
        self.max_turn = np.deg2rad(45)
        self.segcount = len(self.p)
        self.steering_bias = steering_bias  # 1.15 is a good steering bias

        # Waypoint management
        self.target = self.p[0]
        self.curr_idx = -1
        self.initial_idx = 0

        # Laps/timekeeping
        self.laps = 0
        self.start_time = time()
        self.laptime = 0.0
        self.print_laptime = print_laptime


    def __call__(self, pose, curv_segment_length=10, fric=1.20):
        self.curr_idx = int(self.find_waypoint(pose))
        tg = self.p[self.curr_idx]
        steer = self.getSteer(pose, tg)

        v = self.getSpeed(curv_segment_length, fric)
        if v < 2.5:
            v = 2.5
        return v, steer

    def getSteer(self, pose, tg):
        att = np.arctan2(tg[1] - pose[1], tg[0] - pose[0])
        return np.arctan2(np.sin(att - pose[2]), np.cos(att - pose[2]))*self.steering_bias

    def getSpeed(self, curv_segment_length, fric):
        # To account for looping. yes, a lot. Maybe implement circular indexing overloading?
        if curv_segment_length < 3:
            # get instantaneous curvature
            p1, p2, p3 = self.curr_idx - 1, self.curr_idx, self.curr_idx + 1
            p1 %= self.segcount
            p3 %= self.segcount
            curv = curvature(self.p[p1], self.p[p2], self.p[p3])
        else:
            segment_indices = list(range(self.curr_idx, self.curr_idx + curv_segment_length))
            segment_indices = [i % self.segcount for i in segment_indices]
            points = self.p[segment_indices]
            curv = np.sum(np.abs(segment_curvature(points)))
        max_speed = speed_limit(fric, curv)
        return np.clip(max_speed, -self.ms, self.ms)

    def find_waypoint(self, pose, update=True, reset=False):
        if self.curr_idx == -1 or reset:
            curr_idx = np.argmin(np.linalg.norm(self.p - pose[:2], axis=1))
            if update:
                self.initial_idx = curr_idx
                self.start_time = time()
        elif np.linalg.norm(self.p[self.curr_idx] - pose[:2]) < self.lad:
            curr_idx = (self.curr_idx + 1) % len(self.p)
            if curr_idx == self.initial_idx and update:
                self.laps += 1
                self.laptime = time() - self.start_time
                if self.print_laptime:
                    print(f"Laps: {self.laps} | time: {self.laptime:.2f}s")
                self.start_time = time()
        else:
            curr_idx = self.curr_idx
        if update:
            self.curr_idx = curr_idx
        return curr_idx


def draw_controls_overlay(img, velocity, steering_angle, max_speed, max_steering_angle=np.deg2rad(45)):
    """
    Draws velocity bar and steering dial on the image.
    """
    overlay = img.copy()
    h, w = img.shape[:2]

    # --- Velocity Bar (left side)
    bar_height = int(h * 0.4)
    bar_width = 20
    top = int(h * 0.3)
    left = 10
    value = int(bar_height * velocity / max_speed)

    cv2.rectangle(overlay, (left, top), (left + bar_width, top + bar_height), (50, 50, 50), -1)
    cv2.rectangle(overlay, (left, top + bar_height - value), (left + bar_width, top + bar_height), (0, 255, 0), -1)
    cv2.putText(overlay, f"{velocity:.1f} m/s", (left + 30, top + bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Steering Dial (bottom right)
    radius = 40
    cx = 60
    cy = 60
    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), 2)

    angle = np.clip(steering_angle / max_steering_angle, -1, 1)
    heading = np.pi / 2 - angle * (np.pi / 2)
    end_x = int(cx + radius * np.cos(heading))
    end_y = int(cy - radius * np.sin(heading))
    cv2.line(overlay, (cx, cy), (end_x, end_y), (0, 0, 255), 3)
    cv2.putText(overlay, f"{np.rad2deg(steering_angle):.1f} deg", (cx - 50, cy + radius + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay


if __name__ == '__main__':
    from ctrl import XboxController
    will_record = False
    max_speed = 10
    lad = 25
    steer_speed = 1.2

    sim = Sim(map="./tracks/zandvoort.png", race_neg=True, lidar_res=30, max_speed=max_speed, realtime=not will_record)
    sim.car.wheelbase = steer_speed  # smaller wheelbase → faster turning
    c = XboxController(0)
    pp = PurePursuit(discretize_line(sim.full_centerline, 350), lad, max_speed*0.9)
    sim.car.pose[:2] = pp.p[0] * sim.mapm / sim.mapres
    sim.car.pose[2] = np.arctan2(*(pp.p[1] - pp.p[0])[::-1])

    running = True
    control = True

    # --- Racing line visualization ---
    view_racing_line = False
    recording = False
    trajectory = []

    def stop():
        global running, pp
        pp = PurePursuit(discretize_line(sim.full_centerline), lad)
        running = False

    def swap_control():
        global control
        control = not control

    def random_location():
        point_idx = np.random.randint(0, len(pp.p))
        px, py = pp.p[point_idx] * sim.mapm / sim.mapres
        theta = np.arctan2(*(pp.p[(point_idx + 1) % len(pp.p)]
                             - pp.p[point_idx])[::-1])
        sim.car.pose = np.array((px, py, theta))

    def toggle_raceline():
        global view_racing_line, recording, trajectory, pp
        view_racing_line = not view_racing_line
        recording = False
        trajectory = []
        pp.print_laptime = view_racing_line
        print(f"Racing line viewer: {'ON' if view_racing_line else 'OFF'}")

    # Register controller triggers
    c.setTrigger("Start", stop)
    c.setTrigger("A", swap_control)
    c.setTrigger("B", random_location)
    c.setTrigger("Y", toggle_raceline)

    while running:
        pose, (frame, polars), lidar_view = sim.get_state()
        if control:
            turn = c.RJoyX / 10
            sim.car.pose[2] += turn if abs(turn) > 0.03 else 0
            v = (c.RT - c.LT) * max_speed
            steer = sim.car.max_steering_angle * c.LJoyX
        else:
            v, steer = pp(pose)

        sim.update(v, steer)
        img = sim.map.animate(pose=sim.get_pose(), show=False)
        img = draw_controls_overlay(img, sim.car.speed, steer, max_speed)

        # Draw centerline + pursuit point
        for i, pt in enumerate(pp.p):
            color = (0, 0, 255) if i == pp.curr_idx else (255, 0, 0)
            cv2.circle(img, tuple(pt.astype(int)), 2, color, -1)

        # --- Racing line viewer logic ---
        if view_racing_line:
            # Detect crossing start/finish line (around index 0)
            if pp.curr_idx == 0 and not recording:
                if will_record:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    w = cv2.VideoWriter('./videos/demo_runs/lap_'+strftime('%H:%M:%S')+'.avi', fourcc, 1/sim.dt, img.shape[::-1][1:])
                recording = True
                trajectory = []
                print("Recording racing line...")

            # Store path as we move
            if recording:
                trajectory.append(sim.get_pose()[:2])

                # Draw in green
                for i in range(1, len(trajectory)):
                    p1 = tuple(np.int32(trajectory[i - 1]))
                    p2 = tuple(np.int32(trajectory[i]))
                    cv2.line(img, p1, p2, (0, 255, 0), 2)
                if will_record:
                    w.write(img)

        # img = np.hstack([lidar_view, img])
        # img = draw_controls_overlay(img, sim.car.speed, steer, max_speed)
        cv2.imshow("Sim car", img)
        cv2.waitKey(int(100/6) if will_record else 1)
    w.release()