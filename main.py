import pygame
import random
import math
import numpy as np
import time
import sys

# Initialize Pygame
pygame.init()

# Set up the window
WIDTH, HEIGHT = 600, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Dot Simulation with Energy Tracking")
DEFAULT_BASE_MASS = 1

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define simulation properties
BASE_DOT_RADIUS = 3
DOT_COUNT = 200
MIN_DISTANCE = 90
MERGE_DISTANCE = 1
G = 15
TRAIL_LENGTH = 50
TRAIL_FADE_RATE = 0.5
MOMENTUM_LOSS_FACTOR = 0.5
EPSILON = 0.001
SPACE_SIZE = 180
FOV = 360
INITIAL_VELOCITY_RANGE = 0  # New constant for initial velocity range

# Camera parameters
camera_radius = 150
camera_theta = 0
camera_phi = math.pi / 4
camera_rotation_speed = 0.001

# Precompute constants
MIN_DISTANCE_SQ = MIN_DISTANCE ** 2
MERGE_DISTANCE_SQ = MERGE_DISTANCE ** 2

# Initialize font for on-screen text
font = pygame.font.SysFont(None, 24)

# Initialize clock for frame rate control
clock = pygame.time.Clock()

def generate_new_dot(mass=DEFAULT_BASE_MASS):
    """Generates a new dot with random position, specified mass, random velocity, and random color."""
    position = np.random.uniform(-SPACE_SIZE, SPACE_SIZE, 3)
    velocity = np.random.uniform(-INITIAL_VELOCITY_RANGE, INITIAL_VELOCITY_RANGE, 3)
    color = np.random.randint(100, 256, 3)
    return {
        "position": position,
        "velocity": velocity,
        "mass": mass,
        "speed": np.linalg.norm(velocity),
        "color": color
    }

def project(point):
    """Projects 3D coordinates into 2D using perspective projection."""
    x, y, z = point
    factor = FOV / (FOV + z)
    x = x * factor + WIDTH / 2
    y = -y * factor + HEIGHT / 2
    return x, y

# Define cube vertices and edges for the simulation space
cube_vertices = np.array([
    [-SPACE_SIZE, -SPACE_SIZE, -SPACE_SIZE],
    [-SPACE_SIZE, -SPACE_SIZE, +SPACE_SIZE],
    [-SPACE_SIZE, +SPACE_SIZE, -SPACE_SIZE],
    [-SPACE_SIZE, +SPACE_SIZE, +SPACE_SIZE],
    [+SPACE_SIZE, -SPACE_SIZE, -SPACE_SIZE],
    [+SPACE_SIZE, -SPACE_SIZE, +SPACE_SIZE],
    [+SPACE_SIZE, +SPACE_SIZE, -SPACE_SIZE],
    [+SPACE_SIZE, +SPACE_SIZE, +SPACE_SIZE],
])

cube_edges = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
]

# Initialize dots
dots = []
trails = []

for _ in range(DOT_COUNT):
    dot = generate_new_dot(mass=random.uniform(1, 5))
    dots.append(dot)
    trails.append([])

# Variable to store initial total energy
initial_total_energy = None

# Game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update camera angles for rotation
    camera_theta += camera_rotation_speed

    # Compute camera position in Cartesian coordinates
    sin_phi = math.sin(camera_phi)
    cos_phi = math.cos(camera_phi)
    sin_theta = math.sin(camera_theta)
    cos_theta = math.cos(camera_theta)

    camera_position = np.array([
        camera_radius * sin_phi * cos_theta,
        camera_radius * cos_phi,
        camera_radius * sin_phi * sin_theta
    ])

    # Compute camera orientation vectors
    forward = -camera_position / np.linalg.norm(camera_position)
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    # Build the view (camera) matrix
    rotation_matrix = np.array([right, up, forward])
    translation = -rotation_matrix @ camera_position
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = rotation_matrix
    view_matrix[:3, 3] = translation

    merge_actions = []
    num_dots = len(dots)

    total_potential_energy = 0.0  # Initialize total potential energy

    # Update positions and velocities
    for i in range(num_dots):
        dot_i = dots[i]
        pos_i = dot_i["position"]
        mass_i = dot_i["mass"]
        vel_i = dot_i["velocity"]
        total_force = np.zeros(3, dtype=float)

        for j in range(i + 1, num_dots):
            dot_j = dots[j]
            pos_j = dot_j["position"]
            mass_j = dot_j["mass"]

            displacement = pos_j - pos_i
            distance_sq = np.dot(displacement, displacement) + EPSILON
            distance = math.sqrt(distance_sq)

            # Calculate potential energy contribution (positive value)
            potential_energy = G * mass_i * mass_j / distance  # Positive value
            total_potential_energy += potential_energy

            if distance_sq < MERGE_DISTANCE_SQ:
                merge_actions.append((i, j))
                continue

            if distance_sq < MIN_DISTANCE_SQ:
                force_magnitude = G * mass_i * mass_j / MIN_DISTANCE_SQ
            else:
                force_magnitude = G * mass_i * mass_j / distance_sq

            force = (displacement / distance) * force_magnitude
            total_force += force
            # Newton's third law
            dots[j]["velocity"] -= force / mass_j

        # Update velocity and position
        vel_i += total_force / mass_i
        pos_i += vel_i
        dot_i["speed"] = np.linalg.norm(vel_i)

        # Keep dots within the 3D space boundaries
        for axis in range(3):
            if pos_i[axis] < -SPACE_SIZE:
                pos_i[axis] = -SPACE_SIZE
                vel_i[axis] = -vel_i[axis] * MOMENTUM_LOSS_FACTOR
            elif pos_i[axis] > SPACE_SIZE:
                pos_i[axis] = SPACE_SIZE
                vel_i[axis] = -vel_i[axis] * MOMENTUM_LOSS_FACTOR

        # Update trail
        trails[i].append(pos_i.copy())
        if len(trails[i]) > TRAIL_LENGTH:
            trails[i].pop(0)

    # Execute merges
    processed_indices = set()
    for i, j in merge_actions:
        if i in processed_indices or j in processed_indices:
            continue
        base_dot = dots[i]
        remove_dot = dots[j]

        total_mass = base_dot["mass"] + remove_dot["mass"]
        base_dot["velocity"] = (base_dot["velocity"] * base_dot["mass"] + remove_dot["velocity"] * remove_dot["mass"]) / total_mass
        base_dot["position"] = (base_dot["position"] * base_dot["mass"] + remove_dot["position"] * remove_dot["mass"]) / total_mass
        base_dot["mass"] = total_mass
        base_dot["color"] = (base_dot["color"] * base_dot["mass"] + remove_dot["color"] * remove_dot["mass"]) / total_mass
        base_dot["color"] = np.clip(base_dot["color"], 0, 255).astype(int)

        dots.pop(j)
        trails.pop(j)
        num_dots -= 1
        processed_indices.update({i, j})

    # Calculate the total kinetic energy
    total_kinetic_energy = sum(dot['mass'] * dot['speed'] ** 2 / 2 for dot in dots)

    # Calculate total energy
    total_energy = total_kinetic_energy + total_potential_energy

    # Initialize initial_total_energy after first calculation
    if initial_total_energy is None:
        initial_total_energy = total_energy

    # Calculate energy change since start
    energy_change = total_energy - initial_total_energy

    # Clear the window
    window.fill(BLACK)

    # Precompute transformation for all dots
    positions_world = np.array([dot["position"] for dot in dots])
    ones = np.ones((positions_world.shape[0], 1))
    positions_homogeneous = np.hstack((positions_world, ones))
    positions_camera = (view_matrix @ positions_homogeneous.T).T[:, :3]

    # Project positions
    factors = FOV / (FOV + positions_camera[:, 2] + EPSILON)
    positions_2d = positions_camera[:, :2] * factors[:, np.newaxis]
    positions_2d[:, 0] += WIDTH / 2
    positions_2d[:, 1] = -positions_2d[:, 1] + HEIGHT / 2

    # Transform cube vertices to camera coordinates
    cube_positions_world = cube_vertices
    cube_ones = np.ones((cube_positions_world.shape[0], 1))
    cube_positions_homogeneous = np.hstack((cube_positions_world, cube_ones))
    cube_positions_camera = (view_matrix @ cube_positions_homogeneous.T).T[:, :3]

    # Project cube vertices
    cube_factors = FOV / (FOV + cube_positions_camera[:, 2] + EPSILON)
    cube_positions_2d = cube_positions_camera[:, :2] * cube_factors[:, np.newaxis]
    cube_positions_2d[:, 0] += WIDTH / 2
    cube_positions_2d[:, 1] = -cube_positions_2d[:, 1] + HEIGHT / 2

    # Draw cube edges
    for edge in cube_edges:
        start, end = edge
        x1, y1 = cube_positions_2d[start]
        x2, y2 = cube_positions_2d[end]
        pygame.draw.line(window, WHITE, (int(x1), int(y1)), (int(x2), int(y2)))

    # Draw trails
    for i, trail in enumerate(trails):
        if not trail:
            continue
        dot_color = dots[i]["color"]
        # Transform trail points
        trail_positions = np.array(trail)
        ones = np.ones((trail_positions.shape[0], 1))
        trail_homogeneous = np.hstack((trail_positions, ones))
        trail_camera = (view_matrix @ trail_homogeneous.T).T[:, :3]
        factors = FOV / (FOV + trail_camera[:, 2] + EPSILON)
        trail_2d = trail_camera[:, :2] * factors[:, np.newaxis]
        trail_2d[:, 0] += WIDTH / 2
        trail_2d[:, 1] = -trail_2d[:, 1] + HEIGHT / 2

        # Draw trail segments
        points = trail_2d.astype(int)
        if len(points) > 1:
            pygame.draw.aalines(window, dot_color, False, points.tolist())

    # Draw dots
    for i, dot in enumerate(dots):
        x, y = positions_2d[i]
        distance = np.linalg.norm(dot["position"] - camera_position)
        radius = max(1, int(BASE_DOT_RADIUS * (FOV / (FOV + distance))))
        pygame.draw.circle(window, dot["color"], (int(x), int(y)), radius)

    # Render the energy texts
    kinetic_text = font.render(f"Kinetic Energy: {total_kinetic_energy:.2f}", True, WHITE)
    potential_text = font.render(f"Potential Energy: {total_potential_energy:.2f}", True, WHITE)
    total_energy_text = font.render(f"Total Energy: {total_energy:.2f}", True, WHITE)
    energy_change_text = font.render(f"Energy Change: {energy_change:.2f}", True, WHITE)
    window.blit(kinetic_text, (10, 10))
    window.blit(potential_text, (10, 30))
    window.blit(total_energy_text, (10, 50))
    window.blit(energy_change_text, (10, 70))

    # Update the display
    pygame.display.update()

    # Control the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
