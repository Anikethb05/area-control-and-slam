# ==============================================================
#  EKF-SLAM + Camera + Occupancy Grid - Complete Implementation
#  Following "Probabilistic Robotics" by Thrun, Burgard, Fox
#  Fixed: Proper collision, correct EKF equations, EXACT ellipse logic
# ==============================================================

import math
import numpy as np
import pygame
import pygame.gfxdraw

# ---------------------  Environment Helper ---------------------
class Environment:
    def __init__(self):
        self.scale = 40          # 1 m → 40 px (reduced from 50)
        self.offset = (30, 30)   # reduced from (50, 50)

    def position2pixel(self, pos):
        x, y = pos
        return (int(self.offset[0] + x * self.scale),
                int(self.offset[1] + y * self.scale))

    def dist2pixellen(self, d):
        if np.isnan(d):
            return 0
        return int(d * self.scale)

    def get_surface(self):
        return pygame.display.get_surface()

    def show_map(self):
        surf = self.get_surface()
        surf.fill((255, 255, 255))                     # WHITE BACKGROUND
        arena_w, arena_h = 20 * self.scale, 20 * self.scale
        pygame.draw.rect(surf, (0, 0, 0),
                         (*self.offset, arena_w, arena_h), 2)


# ---------------------  Global Parameters ---------------------
WIDTH, HEIGHT = 1200, 800
FPS = 60
GRID_SIZE = 20
MAZE_W, MAZE_H = 20, 20
ROBOT_FOV = 3.0
CAM_FOV_DEG = 60
CAM_MAX_RANGE = 200

# ---------------------  EKF SLAM (Thrun's Book) ------------------
n_state = 3
landmarks = [(4,4),(4,8),(8,8),(12,8),(16,8),(16,4),(12,4),(8,4)]
n_landmarks = len(landmarks)

# ---- Noise (tuned down) ----
R = np.diag([0.0005, 0.0005, 0.0001])   # motion noise
Q = np.diag([0.001, 0.002])            # measurement noise

mu = np.zeros((n_state + 2*n_landmarks, 1))
sigma = np.zeros((n_state + 2*n_landmarks, n_state + 2*n_landmarks))
mu[:] = np.nan
np.fill_diagonal(sigma, 10)           # tighter initial uncertainty

Fx = np.block([[np.eye(3), np.zeros((3, 2*n_landmarks))]])

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))

def sim_measurement(x, landmarks):
    rx, ry, theta = x[0], x[1], x[2]
    zs = []
    for lidx, (lx, ly) in enumerate(landmarks):
        dist = np.hypot(lx-rx, ly-ry)
        if dist > ROBOT_FOV: continue
        phi = np.arctan2(ly-ry, lx-rx) - theta
        phi = normalize_angle(phi)
        zs.append((dist, phi, lidx))
    return zs

def prediction_update(mu, sigma, u, dt):
    """EKF Prediction Step"""
    rx, ry, theta = mu[0,0], mu[1,0], mu[2,0]
    v, w = u[0], u[1]

    if abs(w) > 0.01:
        dx = -(v/w)*np.sin(theta) + (v/w)*np.sin(theta + w*dt)
        dy =  (v/w)*np.cos(theta) - (v/w)*np.cos(theta + w*dt)
    else:
        dx = v*np.cos(theta)*dt
        dy = v*np.sin(theta)*dt
    dtheta = w*dt

    mu[0] += dx
    mu[1] += dy
    mu[2] += dtheta
    mu[2] = normalize_angle(mu[2])

    G = np.eye(sigma.shape[0])
    if abs(w) > 0.01:
        G[0,2] = (v/w)*(np.cos(theta) - np.cos(theta + w*dt))
        G[1,2] = (v/w)*(np.sin(theta) - np.sin(theta + w*dt))
    else:
        G[0,2] = -v*np.sin(theta)*dt
        G[1,2] =  v*np.cos(theta)*dt

    sigma = G @ sigma @ G.T + np.transpose(Fx) @ R @ Fx
    return mu, sigma

def measurement_update(mu, sigma, zs):
    """EKF Measurement Update Step"""
    rx, ry, theta = mu[0,0], mu[1,0], mu[2,0]
    delta_zs = [np.zeros((2,1)) for _ in range(n_landmarks)]
    Ks = [np.zeros((mu.shape[0],2)) for _ in range(n_landmarks)]
    Hs = [np.zeros((2,mu.shape[0])) for _ in range(n_landmarks)]

    for dist, phi, lidx in zs:
        lm = mu[n_state + lidx*2 : n_state + lidx*2+2]
        if np.isnan(lm[0]):
            lm[0] = rx + dist*np.cos(phi+theta)
            lm[1] = ry + dist*np.sin(phi+theta)
            mu[n_state + lidx*2 : n_state + lidx*2+2] = lm

        delta = lm - np.array([[rx],[ry]])
        q = np.linalg.norm(delta)**2

        z_est = np.array([[np.sqrt(q)],
                          [np.arctan2(delta[1,0],delta[0,0])-theta]])
        z_est[1] = normalize_angle(z_est[1])
        z_act = np.array([[dist],[phi]])
        delta_zs[lidx] = z_act - z_est
        delta_zs[lidx][1] = normalize_angle(delta_zs[lidx][1])

        Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
        Fxj[n_state:n_state+2, n_state+2*lidx:n_state+2*lidx+2] = np.eye(2)

        H = np.array([[-delta[0,0]/np.sqrt(q), -delta[1,0]/np.sqrt(q), 0,
                       delta[0,0]/np.sqrt(q), delta[1,0]/np.sqrt(q)],
                      [ delta[1,0]/q, -delta[0,0]/q, -1,
                       -delta[1,0]/q,  delta[0,0]/q ]])
        H = H @ Fxj
        Hs[lidx] = H
        S = H @ sigma @ H.T + Q
        Ks[lidx] = sigma @ H.T @ np.linalg.inv(S)

    mu_offset = np.zeros_like(mu)
    sigma_factor = np.eye(sigma.shape[0])
    for lidx in range(n_landmarks):
        mu_offset += Ks[lidx] @ delta_zs[lidx]
        sigma_factor -= Ks[lidx] @ Hs[lidx]
    mu = mu + mu_offset
    mu[2] = normalize_angle(mu[2])
    sigma = sigma_factor @ sigma
    return mu, sigma


# --------------------- Plot -----------------
def sigma2transform(sig):
    eigvals, eigvecs = np.linalg.eig(sig)
    angle = 180.0 * np.arctan2(eigvecs[1,0], eigvecs[0,0]) / np.pi
    return eigvals, angle

def show_uncertainty_ellipse(env, centre, eigen_px, angle):
    """Draw a 3-sigma ellipse"""
    w, h = eigen_px
    # 3-sigma scaling
    w = max(w * 6, 5)          # minimum 5 px
    h = max(h * 6, 5)
    # guarantee at least 2×2 surface
    size = (int(max(w, 2)), int(max(h, 2)))
    surf = pygame.Surface(size, pygame.SRCALPHA)
    rect = surf.get_rect()
    pygame.draw.ellipse(surf, (255,0,0), rect, 2)
    rot = pygame.transform.rotate(surf, angle)
    dest = rot.get_rect(center=centre)
    env.get_surface().blit(rot, dest)

def show_robot_estimate(mu, sigma, env):
    p = env.position2pixel((mu[0,0], mu[1,0]))
    evals, ang = sigma2transform(sigma[:2,:2])
    w = (env.dist2pixellen(np.sqrt(evals[0])), env.dist2pixellen(np.sqrt(evals[1])))
    show_uncertainty_ellipse(env, p, w, ang)

def show_landmark_estimate(mu, sigma, env):
    for l in range(n_landmarks):
        idx = n_state + l*2
        if np.isnan(mu[idx,0]): continue
        p = env.position2pixel((mu[idx,0], mu[idx+1,0]))
        evals, ang = sigma2transform(sigma[idx:idx+2, idx:idx+2])
        if max(evals) > 15: continue
        w = (max(env.dist2pixellen(np.sqrt(evals[0])),5),
             max(env.dist2pixellen(np.sqrt(evals[1])),5))
        show_uncertainty_ellipse(env, p, w, ang)

def show_landmark_location(landmarks, env):
    for lm in landmarks:
        p = env.position2pixel(lm)
        r = env.dist2pixellen(0.2)
        pygame.gfxdraw.filled_circle(env.get_surface(),
                                     p[0], p[1], r, (0,255,255))

def show_measurements(x, zs, env):
    rx, ry = env.position2pixel((x[0], x[1]))
    for dist, phi, _ in zs:
        lx = x[0] + dist*np.cos(phi + x[2])
        ly = x[1] + dist*np.sin(phi + x[2])
        lp = env.position2pixel((lx, ly))
        pygame.gfxdraw.line(env.get_surface(),
                            rx, ry, lp[0], lp[1], (155,155,155))


# ---------------------  Camera & Occupancy -------------------
class Camera:
    def __init__(self, fov=CAM_FOV_DEG, max_range=CAM_MAX_RANGE):
        self.fov = math.radians(fov)
        self.max_range = max_range
        self.rays = 15

    def get_visible_landmarks(self, robot_pos, robot_angle, landmarks, walls):
        visible = []
        rx, ry = robot_pos
        for lm_id, (lx, ly) in landmarks.items():
            dx, dy = lx-rx, ly-ry
            dist = math.hypot(dx, dy)
            if dist > self.max_range: continue
            occluded = any(w.clipline((rx,ry),(lx,ly)) for w in walls)
            if occluded: continue
            bearing = math.atan2(dy, dx)
            angle_diff = (bearing - robot_angle)
            while angle_diff > math.pi: angle_diff -= 2*math.pi
            while angle_diff < -math.pi: angle_diff += 2*math.pi
            if abs(angle_diff) > self.fov/2: continue
            dist += np.random.normal(0, 2)
            bearing += np.random.normal(0, 0.02)
            visible.append({'id':lm_id, 'distance':dist,
                            'bearing':angle_diff, 'position':(lx,ly)})
        return visible

class OccupancyGrid:
    def __init__(self, width, height, cell_size, offset_x, offset_y):
        self.w = width // cell_size
        self.h = height // cell_size
        self.cell = cell_size
        self.ox, self.oy = offset_x, offset_y
        self.grid = np.full((self.h, self.w), 0.5)
        self.hits = []
        self.grid_hits = set()

    def update(self, robot_pos, robot_angle, cam, walls):
        self.hits.clear(); self.grid_hits.clear()
        rx, ry = robot_pos
        for i in range(cam.rays):
            off = (i/(cam.rays-1)-0.5)*cam.fov
            a = robot_angle + off
            for d in range(0, int(cam.max_range), 5):
                x = rx + d*math.cos(a)
                y = ry + d*math.sin(a)
                gx = int((x-self.ox)/self.cell)
                gy = int((y-self.oy)/self.cell)
                if not (0<=gx<self.w and 0<=gy<self.h): break
                hit = any(w.collidepoint(x,y) for w in walls)
                if hit:
                    self.grid[gy,gx] = min(1.0, self.grid[gy,gx]+0.1)
                    self.hits.append((x,y))
                    self.grid_hits.add((gx,gy))
                    break
                else:
                    self.grid[gy,gx] = max(0.0, self.grid[gy,gx]-0.05)

    def draw(self, surf):
        for y in range(self.h):
            for x in range(self.w):
                v = int(self.grid[y,x]*255)
                r = pygame.Rect(self.ox + x*self.cell,
                                self.oy + y*self.cell,
                                self.cell, self.cell)
                pygame.draw.rect(surf, (v,v,v), r)

class MiniMap:
    def __init__(self, x, y, w, h, maze_w, maze_h, cell):
        self.rect = pygame.Rect(x, y, w, h)
        self.scale = min(w/maze_w, h/maze_h)
        self.cell_draw = max(1, int(cell*self.scale))
        self.discovered = set()

    def update(self, hits):
        self.discovered.update(hits)

    def draw(self, surf):
        pygame.draw.rect(surf, (40,40,40), self.rect)
        pygame.draw.rect(surf, (255,255,255), self.rect, 2)
        for gx, gy in self.discovered:
            mx = int(gx * GRID_SIZE * self.scale)
            my = int(gy * GRID_SIZE * self.scale)
            r = pygame.Rect(self.rect.x + mx, self.rect.y + my,
                            self.cell_draw, self.cell_draw)
            clipped = r.clip(self.rect)
            if clipped.width and clipped.height:
                pygame.draw.rect(surf, (255,255,255), clipped)


# ---------------------  Robot -------------------------------
class Robot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.angle = math.pi/2
        self.v, self.w = 0.0, 0.0
        self.size = 0.3
        self.trail = []
        self.max_trail = 200
        self.v_noise = 0.08
        self.w_noise = 0.03

    def update(self, dt, walls):
        v_noisy = self.v + np.random.normal(0, self.v_noise)
        w_noisy = self.w + np.random.normal(0, self.w_noise)

        if abs(w_noisy) < 0.001:
            nx = self.x + v_noisy*math.cos(self.angle)*dt
            ny = self.y + v_noisy*math.sin(self.angle)*dt
            na = self.angle
        else:
            nx = self.x + (v_noisy/w_noisy)*(math.sin(self.angle + w_noisy*dt) -
                                            math.sin(self.angle))
            ny = self.y + (v_noisy/w_noisy)*(-math.cos(self.angle + w_noisy*dt) +
                                            math.cos(self.angle))
            na = self.angle + w_noisy*dt

        self.angle = na % (2*math.pi)

        # Enhanced collision detection - check multiple points around robot
        collision = False
        check_points = [
            (nx, ny),  # Center
            (nx + self.size * math.cos(na), ny + self.size * math.sin(na)),  # Front
            (nx - self.size * math.cos(na), ny - self.size * math.sin(na)),  # Back
            (nx + self.size * math.cos(na + math.pi/2), ny + self.size * math.sin(na + math.pi/2)),  # Left
            (nx + self.size * math.cos(na - math.pi/2), ny + self.size * math.sin(na - math.pi/2)),  # Right
        ]
        
        for px, py in check_points:
            # Convert to pixel coordinates for collision check
            px_pix = px * 40 + 30
            py_pix = py * 40 + 30
            for wall in walls:
                if wall.collidepoint(px_pix, py_pix):
                    collision = True
                    break
            if collision:
                break

        if not collision:
            self.x, self.y = nx, ny
        else:
            self.v = 0

        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail: self.trail.pop(0)

    def draw(self, env):
        p = env.position2pixel((self.x, self.y))
        pygame.draw.circle(env.get_surface(),
                           (255,0,0), p, env.dist2pixellen(self.size))
        ex = self.x + self.size*1.5*math.cos(self.angle)
        ey = self.y + self.size*1.5*math.sin(self.angle)
        ep = env.position2pixel((ex, ey))
        pygame.draw.line(env.get_surface(),
                         (255,255,0), p, ep, 3)


# ---------------------  Main Simulation --------------------
class SLAMSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("EKF-SLAM + Camera + Occupancy Grid")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

        self.env = Environment()
        self.walls = self._build_walls()
        self.robot = Robot(1.0, 1.0)
        self.camera = Camera()
        self.occ = OccupancyGrid(MAZE_W*40, MAZE_H*40,
                                 GRID_SIZE, 30, 30)
        self.minimap = MiniMap(800, 220, 350, 300,
                               MAZE_W*40, MAZE_H*40, GRID_SIZE)

        self.show_fov = True
        self.show_grid = True
        self.show_features = True

    def _build_walls(self):
        s = self.env.scale
        return [
            pygame.Rect(30, 30, MAZE_W*s, 10),
            pygame.Rect(30, 30, 10, MAZE_H*s),
            pygame.Rect(30, 30+MAZE_H*s, MAZE_W*s, 10),
            pygame.Rect(30+MAZE_W*s, 30, 10, MAZE_H*s),
            pygame.Rect(190, 30, 10, 200),
            pygame.Rect(190, 230, 250, 10),
            pygame.Rect(310, 320, 10, 200),
            pygame.Rect(30, 320, 180, 10),
            pygame.Rect(430, 120, 10, 250),
            pygame.Rect(430, 120, 200, 10),
            pygame.Rect(550, 440, 10, 120),
            pygame.Rect(550, 440, 130, 10),
        ]

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_UP:    self.robot.v = 1.5
                elif e.key == pygame.K_DOWN:  self.robot.v = -1.5
                elif e.key == pygame.K_LEFT:  self.robot.w = 1.2
                elif e.key == pygame.K_RIGHT: self.robot.w = -1.2
            elif e.type == pygame.KEYUP:
                if e.key in (pygame.K_UP, pygame.K_DOWN): self.robot.v = 0
                if e.key in (pygame.K_LEFT, pygame.K_RIGHT): self.robot.w = 0

    def update(self):
        if self.paused: return
        dt = self.clock.get_time() / 1000.0 or 1/FPS

        self.robot.update(dt, self.walls)

        global mu, sigma
        mu, sigma = prediction_update(mu, sigma,
                                     [self.robot.v, self.robot.w], dt)

        zs = sim_measurement([self.robot.x, self.robot.y, self.robot.angle],
                             landmarks)
        mu, sigma = measurement_update(mu, sigma, zs)

        if self.show_grid or self.show_fov:
            self.occ.update((self.robot.x*40+30, self.robot.y*40+30),
                            self.robot.angle, self.camera, self.walls)
            self.minimap.update(self.occ.grid_hits)

    def draw(self):
        self.env.show_map()

        if self.show_grid:
            self.occ.draw(self.screen)

        for w in self.walls:
            pygame.draw.rect(self.screen, (0,0,0), w)

        if self.show_fov:
            self._draw_fov_cone()

        if self.show_features:
            show_landmark_location(landmarks, self.env)

        zs = sim_measurement([self.robot.x, self.robot.y, self.robot.angle],
                             landmarks)
        show_measurements([self.robot.x, self.robot.y, self.robot.angle],
                          zs, self.env)

        show_robot_estimate(mu, sigma, self.env)
        show_landmark_estimate(mu, sigma, self.env)

        self.robot.draw(self.env)

        if self.show_fov:
            for x,y in self.occ.hits:
                pygame.draw.circle(self.screen, (255,0,0), (int(x),int(y)), 3)

        self.minimap.draw(self.screen)
        self._draw_info()
        pygame.display.flip()

    def _draw_fov_cone(self):
        rx = self.robot.x*40 + 30
        ry = self.robot.y*40 + 30
        a = self.robot.angle
        fov = self.camera.fov
        rng = self.camera.max_range
        pts = [(rx, ry)]
        for i in range(21):
            aa = a - fov/2 + (i/20)*fov
            pts.append((rx + rng*math.cos(aa), ry + rng*math.sin(aa)))
        pts.append((rx, ry))
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(s, (255,255,0,40), pts)
        pygame.draw.lines(s, (255,255,0), False, pts, 2)
        self.screen.blit(s, (0,0))

    def _draw_info(self):
        font = pygame.font.Font(None, 24)
        lines = [
            f"Pos: ({self.robot.x:.2f}, {self.robot.y:.2f})",
            f"Angle: {math.degrees(self.robot.angle):.1f} degrees",
            f"EKF: ({mu[0,0]:.2f}, {mu[1,0]:.2f})",
            f"Landmarks: {sum(~np.isnan(mu[n_state::2,0]))}",
            f"{'PAUSED' if self.paused else 'RUNNING'}",
        ]
        for i, txt in enumerate(lines):
            surf = font.render(txt, True, (255, 0, 0))  # ← RED COLOR
            self.screen.blit(surf, (810, 50 + i*28))

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


# ============================ Main ==============================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  EKF-SLAM + Camera + Occupancy Grid")
    print("  Controls: Arrow Keys | SPACE to pause")
    print("="*60 + "\n")

    # Initialise robot pose in EKF
    mu[0:3] = np.array([[1.0],[1.0],[math.pi/2]])
    sigma[0:3,0:3] = 0.02 * np.eye(3)     # very tight start

    sim = SLAMSimulation()
    sim.run()