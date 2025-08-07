import pygame
import numpy as np
from pygame.locals import *
from scipy.optimize import least_squares
import time
import random
from collections import deque

# Constants
FPS = 60
WORLD_SIZE = 15  # 15m x 15m
PIXELS_PER_METER = 40
SCREEN_SIZE = WORLD_SIZE * PIXELS_PER_METER
ROVER_RADIUS = 0.2  # meters
ROVER_WHEELBASE = 0.4  # meters
MAX_SPEED = 1.5  # m/s
MAX_ANGULAR_SPEED = 2.0  # rad/s
COLLISION_PUSH = 0.1  # Increased push-back distance

class LandmarkDetector:
    def __init__(self, max_range=5.0, fov=np.pi/1.5, noise_std=0.05):
        self.max_range = max_range
        self.fov = fov
        self.noise_std = noise_std
    
    def detect_landmarks(self, rover_pose, landmarks, obstacles):
        x, y, theta = rover_pose
        detected = []
        for lm_id, (lx, ly) in enumerate(landmarks):
            dx, dy = lx - x, ly - y
            distance = np.sqrt(dx**2 + dy**2)
            if distance > self.max_range:
                continue
            angle_to_landmark = np.arctan2(dy, dx)
            angle_diff = np.abs(self.normalize_angle(angle_to_landmark - theta))
            if angle_diff > self.fov / 2:
                continue
            if self.is_occluded(x, y, lx, ly, obstacles):
                continue
            noise_x = np.random.normal(0, self.noise_std * distance)
            noise_y = np.random.normal(0, self.noise_std * distance)
            detected.append({
                'id': lm_id,
                'position': (lx + noise_x, ly + noise_y),
                'relative_pos': (dx + noise_x, dy + noise_y),
                'distance': distance,
                'bearing': angle_to_landmark
            })
        return detected
    
    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def is_occluded(self, x1, y1, x2, y2, obstacles):
        for ox, oy, w, h in obstacles:
            if self.line_intersects_rect(x1, y1, x2, y2, ox, oy, w, h):
                return True
        return False
    
    def line_intersects_rect(self, x1, y1, x2, y2, rx, ry, rw, rh):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        corners = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
        for i in range(4):
            if intersect((x1, y1), (x2, y2), corners[i], corners[(i + 1) % 4]):
                return True
        return False

class PoseGraphNode:
    def __init__(self, pose, timestamp, robot_id, node_id):
        self.pose = np.array(pose, dtype=float)
        self.timestamp = timestamp
        self.robot_id = robot_id
        self.node_id = node_id
        self.landmarks = []

class PoseGraphEdge:
    def __init__(self, from_node, to_node, constraint, information_matrix, edge_type='odometry'):
        self.from_node = from_node
        self.to_node = to_node
        self.constraint = np.array(constraint, dtype=float)
        self.information_matrix = np.array(information_matrix, dtype=float)
        self.edge_type = edge_type

class ExplorationPlanner:
    def __init__(self):
        self.grid_size = 0.5
        self.grid_width = int(WORLD_SIZE / self.grid_size)
        self.grid_height = int(WORLD_SIZE / self.grid_size)
        self.explored_grid = np.zeros((self.grid_width, self.grid_height))
        self.obstacle_grid = np.zeros((self.grid_width, self.grid_height))
        self.exploration_radius = 3.0
    
    def update_exploration_map(self, pose, obstacles):
        x, y, _ = pose
        x = np.clip(x, 0, WORLD_SIZE - 0.1)
        y = np.clip(y, 0, WORLD_SIZE - 0.1)
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            self.explored_grid[gx, gy] = 1
            radius = int(self.exploration_radius / self.grid_size)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        dx*dx + dy*dy <= radius*radius):
                        self.explored_grid[nx, ny] = 1
        for ox, oy, w, h in obstacles:
            for px in np.arange(ox, ox + w, self.grid_size):
                for py in np.arange(oy, oy + h, self.grid_size):
                    gx, gy = self.world_to_grid(px, py)
                    if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                        self.obstacle_grid[gx, gy] = 1
    
    def world_to_grid(self, x, y):
        return int(np.clip(x / self.grid_size, 0, self.grid_width - 1)), int(np.clip(y / self.grid_size, 0, self.grid_height - 1))
    
    def grid_to_world(self, gx, gy):
        return (gx + 0.5) * self.grid_size, (gy + 0.5) * self.grid_size
    
    def find_frontiers(self):
        frontiers = []
        for gx in range(1, self.grid_width - 1):
            for gy in range(1, self.grid_height - 1):
                if self.explored_grid[gx, gy] == 0 and self.obstacle_grid[gx, gy] == 0:
                    if any(self.explored_grid[gx + dx, gy + dy] == 1
                           for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)):
                        wx, wy = self.grid_to_world(gx, gy)
                        frontiers.append((wx, wy))
        return frontiers
    
    def get_exploration_target(self, robot_pose, other_robot_poses):
        frontiers = self.find_frontiers()
        if not frontiers:
            return np.random.uniform(2, WORLD_SIZE-2), np.random.uniform(2, WORLD_SIZE-2)
        scores = []
        for fx, fy in frontiers:
            dist = np.sqrt((fx - robot_pose[0])**2 + (fy - robot_pose[1])**2)
            if dist < 0.5:
                score = -float('inf')
            else:
                score = -dist
                for ox, oy, _ in other_robot_poses:
                    other_dist = np.sqrt((fx - ox)**2 + (fy - oy)**2)
                    if other_dist < 2.0:
                        score -= 10.0
            scores.append(score)
        best_idx = np.argmax(scores)
        return frontiers[best_idx]

class KimeraMultiSLAM:
    def __init__(self, num_robots=2):
        self.num_robots = num_robots
        self.pose_graph_nodes = {}
        self.pose_graph_edges = []
        self.landmark_map = {}
        self.robot_trajectories = {i: [] for i in range(num_robots)}
        self.last_optimization_time = time.time()
        self.optimization_interval = 5.0
        self.loop_closure_threshold = 1.5
        self.inter_robot_threshold = 2.0
        self.node_counter = 0
        self.landmark_detector = LandmarkDetector()
        self.robot_local_maps = {i: {} for i in range(num_robots)}
        self.shared_landmarks = set()
        self.exploration_planners = {i: ExplorationPlanner() for i in range(num_robots)}
    
    def add_pose_node(self, robot_id, pose, timestamp, landmarks_detected):
        node_id = f"r{robot_id}_n{self.node_counter}"
        self.node_counter += 1
        node = PoseGraphNode(pose, timestamp, robot_id, node_id)
        node.landmarks = landmarks_detected
        self.pose_graph_nodes[node_id] = node
        self.robot_trajectories[robot_id].append(node_id)
        for landmark in landmarks_detected:
            lm_id = landmark['id']
            if lm_id not in self.landmark_map:
                self.landmark_map[lm_id] = {'position': landmark['position'], 'observations': []}
            self.landmark_map[lm_id]['observations'].append({
                'robot_id': robot_id,
                'node_id': node_id,
                'relative_pos': landmark['relative_pos']
            })
            self.robot_local_maps[robot_id][lm_id] = landmark['position']
            if len({obs['robot_id'] for obs in self.landmark_map[lm_id]['observations']}) > 1:
                self.shared_landmarks.add(lm_id)
        return node_id
    
    def add_odometry_edge(self, from_node_id, to_node_id, odometry_constraint):
        if from_node_id in self.pose_graph_nodes and to_node_id in self.pose_graph_nodes:
            information_matrix = np.diag([10.0, 10.0, 5.0])
            edge = PoseGraphEdge(from_node_id, to_node_id, odometry_constraint, information_matrix)
            self.pose_graph_edges.append(edge)
    
    def detect_loop_closures(self):
        loop_closures = []
        for robot_id in range(self.num_robots):
            trajectory = self.robot_trajectories[robot_id]
            if len(trajectory) < 10:
                continue
            current_node = self.pose_graph_nodes[trajectory[-1]]
            for old_node_id in trajectory[:-10]:
                old_node = self.pose_graph_nodes[old_node_id]
                dist = np.linalg.norm(current_node.pose[:2] - old_node.pose[:2])
                if dist < self.loop_closure_threshold:
                    common_landmarks = {lm['id'] for lm in current_node.landmarks} & {lm['id'] for lm in old_node.landmarks}
                    if len(common_landmarks) >= 2:
                        constraint = self.compute_relative_pose(old_node.pose, current_node.pose)
                        edge = PoseGraphEdge(old_node_id, trajectory[-1], constraint, np.diag([5.0, 5.0, 2.0]), 'loop_closure')
                        loop_closures.append(edge)
                        break
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if self.robot_trajectories[i] and self.robot_trajectories[j]:
                    node_i = self.pose_graph_nodes[self.robot_trajectories[i][-1]]
                    node_j = self.pose_graph_nodes[self.robot_trajectories[j][-1]]
                    dist = np.linalg.norm(node_i.pose[:2] - node_j.pose[:2])
                    if dist < self.inter_robot_threshold:
                        shared = set(self.robot_local_maps[i]) & set(self.robot_local_maps[j])
                        if len(shared) >= 2:
                            constraint = self.compute_relative_pose(node_i.pose, node_j.pose)
                            edge = PoseGraphEdge(self.robot_trajectories[i][-1], self.robot_trajectories[j][-1], 
                                               constraint, np.diag([3.0, 3.0, 1.0]), 'inter_robot')
                            loop_closures.append(edge)
                            break
        return loop_closures
    
    def compute_relative_pose(self, pose1, pose2):
        dx = pose2[0] - pose1[0]
        dy = pose2[1] - pose1[1]
        dtheta = np.arctan2(np.sin(pose2[2] - pose1[2]), np.cos(pose2[2] - pose1[2]))
        return np.array([dx, dy, dtheta])
    
    def optimize_pose_graph(self):
        if len(self.pose_graph_nodes) < 5 or len(self.pose_graph_edges) < 3:
            return
        node_ids = list(self.pose_graph_nodes.keys())
        x0 = np.array([self.pose_graph_nodes[nid].pose for nid in node_ids[1:]]).flatten()
        try:
            result = least_squares(
                self.pose_graph_residuals, x0, args=(node_ids,),
                method='trf', ftol=1e-6, xtol=1e-6, max_nfev=20
            )
            if result.success:
                optimized_poses = result.x.reshape(-1, 3)
                for i, node_id in enumerate(node_ids[1:]):
                    self.pose_graph_nodes[node_id].pose = optimized_poses[i]
        except:
            pass
    
    def pose_graph_residuals(self, x, node_ids):
        poses = [self.pose_graph_nodes[node_ids[0]].pose] + x.reshape(-1, 3).tolist()
        pose_dict = {nid: p for nid, p in zip(node_ids, poses)}
        residuals = []
        for edge in self.pose_graph_edges:
            if edge.from_node not in pose_dict or edge.to_node not in pose_dict:
                continue
            pose_from = pose_dict[edge.from_node]
            pose_to = pose_dict[edge.to_node]
            cos_theta = np.cos(pose_from[2])
            sin_theta = np.sin(pose_from[2])
            dx = pose_to[0] - pose_from[0]
            dy = pose_to[1] - pose_from[1]
            predicted = np.array([
                cos_theta * dx + sin_theta * dy,
                -sin_theta * dx + cos_theta * dy,
                np.arctan2(np.sin(pose_to[2] - pose_from[2]), np.cos(pose_to[2] - pose_from[2]))
            ])
            error = predicted - edge.constraint
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
            weights = np.sqrt(np.diag(edge.information_matrix))
            residuals.extend(weights * error)
        return np.array(residuals)
    
    def update_landmark_map(self):
        for lm_id, lm_data in self.landmark_map.items():
            if len(lm_data['observations']) < 2:
                continue
            positions = []
            for obs in lm_data['observations']:
                node = self.pose_graph_nodes.get(obs['node_id'])
                if node:
                    rx, ry, rt = node.pose
                    rel_x, rel_y = obs['relative_pos']
                    global_x = rx + rel_x * np.cos(rt) - rel_y * np.sin(rt)
                    global_y = ry + rel_x * np.sin(rt) + rel_y * np.cos(rt)
                    positions.append([global_x, global_y])
            if positions:
                lm_data['position'] = tuple(np.mean(positions, axis=0))

class Rover:
    def __init__(self, x, y, theta, robot_id):
        self.x = x
        self.y = y
        self.theta = theta
        self.robot_id = robot_id
        self.v = 0.0
        self.omega = 0.0
        self.trajectory = deque(maxlen=1000)
        self.last_pose = (x, y, theta)
        self.target = None
        self.radius = ROVER_RADIUS
        self.wheelbase = ROVER_WHEELBASE
        self.odom_noise = [0.01, 0.01, 0.02]
        self.landmark_detector = LandmarkDetector()
        self.stuck_counter = 0
        self.max_stuck = 5  # Reduced to react faster
        self.escape_rotation = 0.0  # Track rotation during escape
    
    def update(self, wl, wr, dt, obstacles, rovers):
        v = (wl + wr) / 2
        omega = (wr - wl) / self.wheelbase
        v = np.clip(v, -MAX_SPEED, MAX_SPEED)
        omega = np.clip(omega, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        
        if self.stuck_counter > self.max_stuck:
            # Escape maneuver: rotate 180 degrees, then move forward
            if abs(self.escape_rotation) < np.pi:
                omega = MAX_ANGULAR_SPEED * np.sign(random.choice([-1, 1]))
                v = 0
                self.escape_rotation += omega * dt
            else:
                v = MAX_SPEED * 0.5
                omega = 0
            self.stuck_counter = min(self.stuck_counter, self.max_stuck + 5)  # Allow some movement
        else:
            self.escape_rotation = 0.0
        
        new_x = self.x + v * np.cos(self.theta) * dt
        new_y = self.y + v * np.sin(self.theta) * dt
        new_theta = self.theta + omega * dt
        
        collided = self.check_collision(new_x, new_y, obstacles, rovers)
        if collided:
            dx, dy = 0, 0
            if len(collided) == 2:  # Rover collision
                ox, oy = collided
                dist = np.sqrt((self.x - ox)**2 + (self.y - oy)**2)
                if dist > 0:
                    dx = (self.x - ox) * COLLISION_PUSH / dist
                    dy = (self.y - oy) * COLLISION_PUSH / dist
            else:  # Obstacle collision
                ox, oy, w, h = collided
                cx, cy = ox + w/2, oy + h/2
                dist = np.sqrt((self.x - cx)**2 + (self.y - cy)**2)
                if dist > 0:
                    dx = (self.x - cx) * COLLISION_PUSH / dist
                    dy = (self.y - cy) * COLLISION_PUSH / dist
            new_x += dx
            new_y += dy
            
            if self.check_collision(new_x, new_y, obstacles, rovers):
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        if not self.check_collision(new_x, new_y, obstacles, rovers):
            self.x, self.y, self.theta = new_x, new_y, np.arctan2(np.sin(new_theta), np.cos(new_theta))
            self.v, self.omega = v, omega
            self.trajectory.append((self.x, self.y))
        
        self.x = np.clip(self.x, self.radius, WORLD_SIZE - self.radius)
        self.y = np.clip(self.y, self.radius, WORLD_SIZE - self.radius)
    
    def check_collision(self, x, y, obstacles, rovers):
        for ox, oy, w, h in obstacles:
            if (ox - self.radius <= x <= ox + w + self.radius and 
                oy - self.radius <= y <= oy + h + self.radius):
                return (ox, oy, w, h)
        for other in rovers:
            if other.robot_id != self.robot_id:
                dist = np.sqrt((x - other.x)**2 + (y - other.y)**2)
                if dist < 2 * self.radius:
                    return (other.x, other.y)
        return False
    
    def get_odometry_constraint(self):
        dx = self.x - self.last_pose[0] + np.random.normal(0, self.odom_noise[0])
        dy = self.y - self.last_pose[1] + np.random.normal(0, self.odom_noise[1])
        dtheta = np.arctan2(np.sin(self.theta - self.last_pose[2]), np.cos(self.theta - self.last_pose[2]))
        dtheta += np.random.normal(0, self.odom_noise[2])
        return np.array([dx, dy, dtheta])
    
    def update_last_pose(self):
        self.last_pose = (self.x, self.y, self.theta)
    
    def get_pose(self):
        return self.x, self.y, self.theta
    
    def get_swarm_commands(self, target, obstacles, rovers):
        if not target:
            return 0, 0
        tx, ty = target
        dx = tx - self.x
        dy = ty - self.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.3:
            return 0, 0
        
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(angle_to_target - self.theta), np.cos(angle_to_target - self.theta))
        base_v = min(MAX_SPEED, dist * 2.0)
        base_omega = angle_diff * 3.0
        
        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        neighbors = 0
        colliding = self.stuck_counter > 0
        
        for other in rovers:
            if other.robot_id == self.robot_id:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 2.0 and dist > 0:
                separation -= np.array([dx, dy]) / (dist**2 + 0.1)
                alignment += np.array([np.cos(other.theta), np.sin(other.theta)])
                cohesion += np.array([dx, dy])
                neighbors += 1
        
        if neighbors > 0:
            alignment = alignment / neighbors
            cohesion = cohesion / neighbors
            separation = separation / neighbors
            swarm_v = (separation * (3.0 if colliding else 1.0) + alignment * 0.5 + cohesion * 0.1) * 0.5
            base_v += np.linalg.norm(swarm_v) * MAX_SPEED
            base_omega += np.arctan2(swarm_v[1], swarm_v[0]) * 0.5
        
        avoidance = np.zeros(2)
        for ox, oy, w, h in obstacles:
            cx, cy = ox + w/2, oy + h/2
            dx = self.x - cx
            dy = self.y - cy
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1.5 and dist > 0:
                avoidance += np.array([dx, dy]) / (dist**2 + 0.1)
        if np.linalg.norm(avoidance) > 0:
            base_omega += np.arctan2(avoidance[1], avoidance[0]) * 2.0
        
        wl = base_v - base_omega * self.wheelbase / 2
        wr = base_v + base_omega * self.wheelbase / 2
        return np.clip(wl, -MAX_SPEED, MAX_SPEED), np.clip(wr, -MAX_SPEED, MAX_SPEED)

class Environment:
    def __init__(self):
        self.obstacles = [
            (3, 3, 2, 1), (8, 8, 1, 2), (12, 5, 1, 1),
            (5, 11, 2, 1), (1, 7, 1, 1)
        ]
        self.landmarks = [
            (2, 2), (4, 1), (7, 2), (10, 3), (13, 4),
            (1, 5), (6, 6), (9, 7), (12, 8), (14, 9),
            (3, 10), (7, 11), (11, 12), (14, 13), (2, 14)
        ]

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Kimera-Multi SLAM Swarm Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    env = Environment()
    slam = KimeraMultiSLAM(num_robots=2)
    rovers = [
        Rover(2, 2, 0, 0),
        Rover(13, 13, np.pi, 1)
    ]
    colors = [(255, 0, 0), (0, 255, 0)]
    
    step = 0
    last_slam_update = time.time()
    running = True
    
    while running and step < 2000:
        dt = clock.tick(FPS) / 1000.0
        if dt > 0.1:
            dt = 0.1
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        
        for rover in rovers:
            pose = rover.get_pose()
            slam.exploration_planners[rover.robot_id].update_exploration_map(pose, env.obstacles)
            other_poses = [r.get_pose() for r in rovers if r.robot_id != rover.robot_id]
            rover.target = slam.exploration_planners[rover.robot_id].get_exploration_target(pose, other_poses)
            wl, wr = rover.get_swarm_commands(rover.target, env.obstacles, rovers)
            rover.update(wl, wr, dt, env.obstacles, rovers)
            
            if step % 5 == 0:
                detected = rover.landmark_detector.detect_landmarks(pose, env.landmarks, env.obstacles)
                node_id = slam.add_pose_node(rover.robot_id, pose, time.time(), detected)
                if node_id and len(slam.robot_trajectories[rover.robot_id]) > 1:
                    slam.add_odometry_edge(slam.robot_trajectories[rover.robot_id][-2], node_id, rover.get_odometry_constraint())
                rover.update_last_pose()
        
        if time.time() - last_slam_update > slam.optimization_interval:
            slam.pose_graph_edges.extend(slam.detect_loop_closures())
            slam.optimize_pose_graph()
            slam.update_landmark_map()
            last_slam_update = time.time()
        
        screen.fill((240, 240, 240))
        
        planner = slam.exploration_planners[0]
        for i in range(planner.grid_width):
            for j in range(planner.grid_height):
                if planner.obstacle_grid[i, j]:
                    color = (80, 80, 80)
                elif planner.explored_grid[i, j]:
                    color = (200, 200, 200)
                else:
                    continue
                wx, wy = planner.grid_to_world(i, j)
                pygame.draw.rect(screen, color, 
                    (wx * PIXELS_PER_METER - 0.2 * PIXELS_PER_METER,
                     (WORLD_SIZE - wy) * PIXELS_PER_METER - 0.2 * PIXELS_PER_METER,
                     0.4 * PIXELS_PER_METER, 0.4 * PIXELS_PER_METER))
        
        for ox, oy, w, h in env.obstacles:
            pygame.draw.rect(screen, (80, 80, 80),
                (ox * PIXELS_PER_METER, (WORLD_SIZE - oy - h) * PIXELS_PER_METER,
                 w * PIXELS_PER_METER, h * PIXELS_PER_METER))
        
        for i, (lx, ly) in enumerate(env.landmarks):
            pygame.draw.circle(screen, (100, 100, 100),
                              (int(lx * PIXELS_PER_METER), int((WORLD_SIZE - ly) * PIXELS_PER_METER)), 6)
            text = font.render(str(i), True, (0, 0, 0))
            screen.blit(text, (lx * PIXELS_PER_METER + 8, (WORLD_SIZE - ly) * PIXELS_PER_METER - 8))
        
        for lm_id, lm_data in slam.landmark_map.items():
            lx, ly = lm_data['position']
            color = (255, 255, 0) if lm_id not in slam.shared_landmarks else (255, 165, 0)
            pygame.draw.circle(screen, color,
                              (int(lx * PIXELS_PER_METER), int((WORLD_SIZE - ly) * PIXELS_PER_METER)), 8)
            text = font.render(str(lm_id), True, (0, 0, 0))
            screen.blit(text, (lx * PIXELS_PER_METER + 8, (WORLD_SIZE - ly) * PIXELS_PER_METER - 8))
        
        for i, rover in enumerate(rovers):
            points = [(x * PIXELS_PER_METER, (WORLD_SIZE - y) * PIXELS_PER_METER) for x, y in rover.trajectory]
            if len(points) > 1:
                pygame.draw.lines(screen, colors[i], False, points, 2)
            
            slam_points = [(n.pose[0] * PIXELS_PER_METER, (WORLD_SIZE - n.pose[1]) * PIXELS_PER_METER)
                          for nid in slam.robot_trajectories[rover.robot_id]
                          for n in [slam.pose_graph_nodes[nid]]]
            if len(slam_points) > 1:
                pygame.draw.lines(screen, (colors[i][0]//2, colors[i][1]//2, colors[i][2]//2), False, slam_points, 3)
            
            cx, cy = rover.x * PIXELS_PER_METER, (WORLD_SIZE - rover.y) * PIXELS_PER_METER
            pygame.draw.circle(screen, colors[i],
                              (int(cx), int(cy)), int(ROVER_RADIUS * PIXELS_PER_METER))
            pygame.draw.circle(screen, (0, 0, 0),
                              (int(cx), int(cy)), int(ROVER_RADIUS * PIXELS_PER_METER), 1)
            dx = ROVER_RADIUS * PIXELS_PER_METER * np.cos(rover.theta)
            dy = -ROVER_RADIUS * PIXELS_PER_METER * np.sin(rover.theta)
            pygame.draw.line(screen, (0, 0, 0), (cx, cy), (cx + dx, cy + dy), 3)
            
            fov_angle = np.pi / 1.5
            for offset in [-fov_angle/2, fov_angle/2]:
                fov_x = cx + 2 * PIXELS_PER_METER * np.cos(rover.theta + offset)
                fov_y = cy - 2 * PIXELS_PER_METER * np.sin(rover.theta + offset)
                pygame.draw.line(screen, colors[i], (cx, cy), (fov_x, fov_y), 1)
            
            if rover.target:
                tx, ty = rover.target
                pygame.draw.circle(screen, (255, 255, 255),
                                  (int(tx * PIXELS_PER_METER), int((WORLD_SIZE - ty) * PIXELS_PER_METER)), 5)
                pygame.draw.circle(screen, (0, 0, 0),
                                  (int(tx * PIXELS_PER_METER), int((WORLD_SIZE - ty) * PIXELS_PER_METER)), 5, 1)
        
        for edge in slam.pose_graph_edges:
            if edge.edge_type in ['loop_closure', 'inter_robot']:
                p1 = slam.pose_graph_nodes[edge.from_node].pose
                p2 = slam.pose_graph_nodes[edge.to_node].pose
                color = (255, 0, 255) if edge.edge_type == 'loop_closure' else (0, 255, 255)
                pygame.draw.line(screen, color,
                    (p1[0] * PIXELS_PER_METER, (WORLD_SIZE - p1[1]) * PIXELS_PER_METER),
                    (p2[0] * PIXELS_PER_METER, (WORLD_SIZE - p2[1]) * PIXELS_PER_METER), 2)
        
        info = [
            f"Nodes: {len(slam.pose_graph_nodes)}",
            f"Edges: {len(slam.pose_graph_edges)}",
            f"Landmarks: {len(slam.landmark_map)}",
            f"Shared: {len(slam.shared_landmarks)}",
            f"Step: {step}"
        ]
        for i, text in enumerate(info):
            surface = font.render(text, True, (0, 0, 0))
            screen.blit(surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        step += 1
    
    pygame.quit()

if __name__ == "__main__":
    main()