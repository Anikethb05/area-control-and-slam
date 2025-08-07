import pygame
import numpy as np
import random
import math
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
MAP_WIDTH = 800
MAP_HEIGHT = 600
GRID_SIZE = 10
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

class Environment:
    def __init__(self):
        self.obstacles = []
        self.generate_obstacles()
    
    def generate_obstacles(self):
        """Generate random obstacles in the environment"""
        num_obstacles = 12
        for _ in range(num_obstacles):
            x = random.randint(80, MAP_WIDTH - 80)
            y = random.randint(80, MAP_HEIGHT - 80)
            width = random.randint(25, 60)
            height = random.randint(25, 60)
            
            # Make sure obstacles don't spawn too close to robot start position
            if math.sqrt((x - 100)**2 + (y - 100)**2) > 60:
                self.obstacles.append(pygame.Rect(x, y, width, height))
    
    def is_collision(self, x, y, radius=5):
        """Check if position collides with obstacles"""
        robot_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        for obstacle in self.obstacles:
            if robot_rect.colliderect(obstacle):
                return True
        return False
    
    def draw(self, screen):
        """Draw the environment"""
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, GRAY, obstacle)

class LidarSensor:
    def __init__(self, max_range=120, num_beams=36):
        self.max_range = max_range
        self.num_beams = num_beams
        self.angle_increment = 2 * math.pi / num_beams
    
    def scan(self, robot_x, robot_y, robot_angle, environment):
        """Perform lidar scan and return distances"""
        readings = []
        for i in range(self.num_beams):
            beam_angle = robot_angle + i * self.angle_increment
            distance = self.cast_ray(robot_x, robot_y, beam_angle, environment)
            readings.append(distance)
        return readings
    
    def cast_ray(self, start_x, start_y, angle, environment):
        """Cast a ray and return distance to nearest obstacle"""
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        for distance in range(1, int(self.max_range)):
            x = start_x + dx * distance
            y = start_y + dy * distance
            
            # Check bounds
            if x < 0 or x >= MAP_WIDTH or y < 0 or y >= MAP_HEIGHT:
                return distance
            
            # Check obstacles
            if environment.is_collision(x, y):
                return distance
        
        return self.max_range

class OccupancyGrid:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.grid = np.full((self.grid_height, self.grid_width), 0.5)  # Unknown = 0.5
        self.log_odds = np.zeros((self.grid_height, self.grid_width))
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        return gx, gy
    
    def grid_to_world(self, gx, gy):
        """Convert grid coordinates to world coordinates"""
        x = gx * self.resolution + self.resolution / 2
        y = gy * self.resolution + self.resolution / 2
        return x, y
    
    def update_cell(self, gx, gy, occupied):
        """Update a single cell with log-odds"""
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            if occupied:
                self.log_odds[gy, gx] += 0.4  # Occupied
            else:
                self.log_odds[gy, gx] -= 0.2  # Free
            
            # Convert log-odds to probability
            self.grid[gy, gx] = 1 / (1 + np.exp(-self.log_odds[gy, gx]))
    
    def update_scan(self, robot_x, robot_y, robot_angle, scan_data, max_range):
        """Update grid with lidar scan"""
        for i, distance in enumerate(scan_data):
            beam_angle = robot_angle + i * (2 * math.pi / len(scan_data))
            
            # Mark free cells along the beam
            for d in range(0, int(distance), 3):
                x = robot_x + math.cos(beam_angle) * d
                y = robot_y + math.sin(beam_angle) * d
                gx, gy = self.world_to_grid(x, y)
                self.update_cell(gx, gy, False)
            
            # Mark occupied cell at the end
            if distance < max_range:
                end_x = robot_x + math.cos(beam_angle) * distance
                end_y = robot_y + math.sin(beam_angle) * distance
                gx, gy = self.world_to_grid(end_x, end_y)
                self.update_cell(gx, gy, True)
    
    def draw(self, screen, offset_x=0, offset_y=0):
        """Draw the occupancy grid"""
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                prob = self.grid[gy, gx]
                if prob == 0.5:  # Unknown
                    continue
                
                # Convert probability to color
                color_val = int(255 * (1 - prob))
                color = (color_val, color_val, color_val)
                
                x = offset_x + gx * self.resolution
                y = offset_y + gy * self.resolution
                pygame.draw.rect(screen, color, 
                               (x, y, self.resolution, self.resolution))

class Robot:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 3.0
        self.path = deque(maxlen=300)
        self.stuck_counter = 0
        self.last_positions = deque(maxlen=10)
        self.exploration_timer = 0
        
    def random_walk(self, environment):
        """Perform random walk motion"""
        # Store current position
        self.last_positions.append((self.x, self.y))
        self.exploration_timer += 1
        
        # Check if stuck (not moving much)
        if len(self.last_positions) >= 5:
            recent_positions = list(self.last_positions)[-5:]
            distances = []
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                distances.append(math.sqrt(dx*dx + dy*dy))
            
            if sum(distances) < 8:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        # Random motion strategy
        if self.stuck_counter > 3:
            # Turn significantly if stuck
            self.angle += random.uniform(math.pi/2, 3*math.pi/2)
            self.stuck_counter = 0
        elif self.exploration_timer % 60 == 0:  # Change direction every 2 seconds
            self.angle += random.uniform(-math.pi/3, math.pi/3)
        elif random.random() < 0.08:  # 8% chance for random turn
            self.angle += random.uniform(-math.pi/4, math.pi/4)
        
        # Try to move forward
        new_x = self.x + math.cos(self.angle) * self.speed
        new_y = self.y + math.sin(self.angle) * self.speed
        
        # Check boundaries and collisions
        collision = False
        if (new_x < 20 or new_x > MAP_WIDTH - 20 or 
            new_y < 20 or new_y > MAP_HEIGHT - 20):
            collision = True
        elif environment.is_collision(new_x, new_y, 12):
            collision = True
        
        if collision:
            # Try different angles to find a clear path
            angles_to_try = [
                self.angle + math.pi/3,
                self.angle - math.pi/3,
                self.angle + math.pi/2,
                self.angle - math.pi/2,
                self.angle + 2*math.pi/3,
                self.angle - 2*math.pi/3,
                self.angle + math.pi
            ]
            
            moved = False
            for try_angle in angles_to_try:
                test_x = self.x + math.cos(try_angle) * self.speed
                test_y = self.y + math.sin(try_angle) * self.speed
                
                if (test_x >= 20 and test_x <= MAP_WIDTH - 20 and 
                    test_y >= 20 and test_y <= MAP_HEIGHT - 20 and
                    not environment.is_collision(test_x, test_y, 12)):
                    self.angle = try_angle
                    self.x = test_x
                    self.y = test_y
                    moved = True
                    break
            
            if not moved:
                # If completely stuck, turn around
                self.angle += math.pi + random.uniform(-0.3, 0.3)
        else:
            # Move normally
            self.x = new_x
            self.y = new_y
        
        self.angle = self.angle % (2 * math.pi)
        self.path.append((self.x, self.y))
    
    def draw(self, screen):
        """Draw the robot"""
        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(screen, BLUE, False, list(self.path), 2)
        
        # Draw robot body
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), 10)
        
        # Draw direction indicator
        end_x = self.x + math.cos(self.angle) * 18
        end_y = self.y + math.sin(self.angle) * 18
        pygame.draw.line(screen, RED, (int(self.x), int(self.y)), 
                        (int(end_x), int(end_y)), 3)

class RandomWalkSLAM:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Random Walk SLAM - Single Rover")
        self.clock = pygame.time.Clock()
        
        # Initialize components
        self.environment = Environment()
        self.robot = Robot(100, 100, random.uniform(0, 2*math.pi))
        self.lidar = LidarSensor(max_range=120, num_beams=36)
        self.occupancy_grid = OccupancyGrid(MAP_WIDTH, MAP_HEIGHT, GRID_SIZE)
        
        # UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # Statistics
        self.total_distance = 0
        self.last_position = (self.robot.x, self.robot.y)
        self.scan_count = 0
        
    def run(self):
        """Main SLAM loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset
                        self.reset_simulation()
                    elif event.key == pygame.K_SPACE:
                        # Pause/unpause
                        self.wait_for_space()
            
            # Robot motion
            self.robot.random_walk(self.environment)
            
            # Update statistics
            self.update_statistics()
            
            # SLAM mapping
            self.update_map()
            
            # Rendering
            self.draw()
            
            self.clock.tick(FPS)
        
        pygame.quit()
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.robot = Robot(100, 100, random.uniform(0, 2*math.pi))
        self.occupancy_grid = OccupancyGrid(MAP_WIDTH, MAP_HEIGHT, GRID_SIZE)
        self.total_distance = 0
        self.scan_count = 0
        self.last_position = (self.robot.x, self.robot.y)
    
    def wait_for_space(self):
        """Pause until space is pressed again"""
        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = False
    
    def update_statistics(self):
        """Update movement statistics"""
        dx = self.robot.x - self.last_position[0]
        dy = self.robot.y - self.last_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        self.total_distance += distance
        self.last_position = (self.robot.x, self.robot.y)
    
    def update_map(self):
        """Update the occupancy grid with lidar scan"""
        # Get lidar scan
        scan_data = self.lidar.scan(self.robot.x, self.robot.y, 
                                   self.robot.angle, self.environment)
        
        # Update occupancy grid - pass max_range from lidar
        self.occupancy_grid.update_scan(self.robot.x, self.robot.y,
                                       self.robot.angle, scan_data, 
                                       self.lidar.max_range)
        
        self.scan_count += 1
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(WHITE)
        
        # Draw occupancy grid first (background)
        self.occupancy_grid.draw(self.screen)
        
        # Draw environment obstacles
        self.environment.draw(self.screen)
        
        # Draw robot
        self.robot.draw(self.screen)
        
        # Draw current lidar scan
        self.draw_lidar_scan()
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_lidar_scan(self):
        """Draw the current lidar scan"""
        scan_data = self.lidar.scan(self.robot.x, self.robot.y, 
                                   self.robot.angle, self.environment)
        
        for i, distance in enumerate(scan_data):
            if distance < self.lidar.max_range:
                beam_angle = self.robot.angle + i * (2 * math.pi / len(scan_data))
                end_x = self.robot.x + math.cos(beam_angle) * distance
                end_y = self.robot.y + math.sin(beam_angle) * distance
                
                # Draw laser beam (thin line)
                pygame.draw.line(self.screen, ORANGE, 
                               (int(self.robot.x), int(self.robot.y)),
                               (int(end_x), int(end_y)), 1)
                
                # Draw hit point
                pygame.draw.circle(self.screen, YELLOW, 
                                 (int(end_x), int(end_y)), 2)
    
    def draw_ui(self):
        """Draw user interface"""
        # Background for UI
        ui_rect = pygame.Rect(MAP_WIDTH + 10, 0, SCREEN_WIDTH - MAP_WIDTH - 10, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, ui_rect)
        
        # Title
        legend_x = MAP_WIDTH + 20
        legend_y = 20
        
        title = self.font.render("Random Walk SLAM", True, WHITE)
        self.screen.blit(title, (legend_x, legend_y))
        
        # Legend
        legend_y += 40
        legend_items = [
            ("Red Circle: Robot", RED),
            ("Blue Line: Robot Path", BLUE),
            ("Orange Lines: Lidar Beams", ORANGE),
            ("Yellow Dots: Laser Hits", YELLOW),
            ("Gray Blocks: Obstacles", GRAY),
            ("Black/White: Occupancy Map", WHITE),
            ("", WHITE),
            ("Controls:", WHITE),
            ("R - Reset Simulation", WHITE),
            ("SPACE - Pause/Resume", WHITE)
        ]
        
        for i, (text, color) in enumerate(legend_items):
            if text:
                surface = self.small_font.render(text, True, color)
                self.screen.blit(surface, (legend_x, legend_y + i * 22))
        
        # Statistics
        stats_y = legend_y + len(legend_items) * 22 + 30
        
        # Calculate coverage
        total_cells = self.occupancy_grid.grid_width * self.occupancy_grid.grid_height
        mapped_cells = np.sum(self.occupancy_grid.grid != 0.5)
        coverage_percent = (mapped_cells / total_cells) * 100
        
        stats = [
            "Statistics:",
            f"Position: ({self.robot.x:.0f}, {self.robot.y:.0f})",
            f"Angle: {math.degrees(self.robot.angle):.0f}°",
            f"Distance Traveled: {self.total_distance:.0f}px",
            f"Scans Taken: {self.scan_count}",
            f"Map Coverage: {coverage_percent:.1f}%",
            f"Path Length: {len(self.robot.path)} points"
        ]
        
        for i, stat in enumerate(stats):
            color = YELLOW if i == 0 else WHITE
            surface = self.small_font.render(stat, True, color)
            self.screen.blit(surface, (legend_x, stats_y + i * 22))

if __name__ == "__main__":
    slam = RandomWalkSLAM()
    slam.run()