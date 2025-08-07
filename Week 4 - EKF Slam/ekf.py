'''
EKF SLAM Logic:

mu: state estimate vector
sigma: state uncertainity/covariance matrix

Two steps:
    - Prediction Update
        -From the control inputs u and some model, how does our state estimate evolve?
        -Moving only affects the state estimate of the robot
        -Model affects uncertainity of the system
        -Model noise also affects uncertainity of the system
    - Measurement Update
        -From what the robot observes, how do we update our state estimate?
        -We reconcile current uncertainty with the uncertainity of measurements
'''


from python_ugv_sim.utils import environment, vehicles
import numpy as np
import pygame


# Robot parameters
n_states = 3 # x, y, theta
n_landmarks = 1 # Number of landmarks, will be updated as we observe them

# State vector: [x, y, theta, landmark_1_x, landmark_1_y, ...]
mu=np.zeros((n_states + 2 * n_landmarks, 1))
sigma=np.zeros((n_states + 2*n_landmarks, n_states + 2*n_landmarks))

# Helpful matrices
Fx=np.block([[np.eye(n_states), np.zeros((n_states, 2*n_landmarks))]])

# <------------ EKF SLAM -------------->
def prediction_update(mu,sigma,u,dt):
    rx,ry,theta=mu[0],mu[1],mu[2] # Robot state
    v,w=u[0],u[1]
    # Update stsate
    state_model_mat=np.zeros((n_states,1))
    state_model_mat[0]=-(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w*dt) if np.abs(w)>1e-6 else v*dt*np.cos(theta)
    state_model_mat[1]=(v/w)*np.cos(theta)-(v.w)*np.cos(theta+w*dt) if np.abs(w)>1e-6 else v*dt*np.sin(theta)
    state_model_mat[2]=w*dt
    mu+=np.transpose(Fx).dot(state_model_mat) # Update state estimate
    return

def measurement_update(mu,sigma):
    return
                       
# <------------ EKF SLAM -------------->

# <------------ PLOT ------------------>
def show_uncertainity_ellipse(env,center,width,angle):
    target_rect = pygame.Rect(center[0]-int(width[0]/2),center[1]-int(width[1]/2),width[0],width[1])
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, env.red, (0, 0, *target_rect.size), 2)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    env.get_pygame_surface().blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))
# <------------ PLOT ------------------>


if __name__=='__main__':

    # Initialize pygame
    pygame.init()

    # Initialize robot and time step
    x_init = np.array([1,1,np.pi/2])  # Initial state: [x, y, theta]
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01

    # Initialize and display environment
    env = environment.Environment(map_image_path="./python_ugv_sim/maps/map_blank.png")

    running = True
    u = np.array([0.,0.]) # Controls, linear velocity and angular velocity
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            u = robot.update_u(u,event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u # Update controls based on key states
        robot.move_step(u,dt) # Integrate EOMs forward, i.e., move robot
        env.show_map() # Re-blit map
        env.show_robot(robot) # Re-blit robot
        pygame.display.update() # Update display