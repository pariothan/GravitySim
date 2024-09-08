import pygame
import random
import math
import time

# Initialize Pygame
pygame.init()

# Set up the window
WIDTH, HEIGHT = 1000, 1000
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Dot Simulation")
DEFAULT_BASE_MASS = 1

def generate_new_dot(mass=DEFAULT_BASE_MASS):
    """Generates a new dot with random position, specified mass, and random color."""
    x = random.randint(BASE_DOT_RADIUS, WIDTH - BASE_DOT_RADIUS)
    y = random.randint(BASE_DOT_RADIUS, HEIGHT - BASE_DOT_RADIUS)
    z = random.randint(BASE_DOT_RADIUS, HEIGHT - BASE_DOT_RADIUS)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return {"x": x, "y": y, "z": z, "vx": 0, "vy": 0, "vz": 0, "mass": mass, "speed": 0, "color": color}

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define dot properties
BASE_DOT_RADIUS = 1  # Base radius for dot
DOT_COUNT = 100
MIN_DISTANCE = 90  # Minimum distance for force calculation
MERGE_DISTANCE = 10  # Distance within which dots merge
G = 1  # starting Gravitational constant
TRAIL_LENGTH = 150  # Length of the trail
TRAIL_FADE_RATE = 0.5  # Rate at which the trail fades
MOMENTUM_LOSS_FACTOR = 0.8  # Factor for momentum loss on edge collision
EPSILON = 0.001  # Small constant to prevent division by zero

dots = []
trails = []  # List to store trails for all dots

# Initialize dots
for i in range(DOT_COUNT):
    x = random.randint(BASE_DOT_RADIUS, WIDTH - BASE_DOT_RADIUS)
    y = random.randint(BASE_DOT_RADIUS, HEIGHT - BASE_DOT_RADIUS)
    z = random.randint(BASE_DOT_RADIUS, HEIGHT - BASE_DOT_RADIUS)
    vx = 0  # Initial velocity set to 0
    vy = 0  # Initial velocity set to 0
    vz = 0  # Initial velocity set to 0
    mass = random.randint(1, 50)  # Random mass for each dot
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color for each dot
    dots.append({"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz, "mass": mass, "speed": 0, "color": color})
    trails.append([])  # Initialize an empty trail for each dot
iters = 0
# Game loop
running = True
while running:
    start_time = time.time()
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    merge_actions = []

    # Move dots
    for i, dot in enumerate(dots[:]):
        total_fx = 0
        total_fy = 0
        total_fz = 0

        for j, other_dot in enumerate(dots[:]):
            if i != j:
                dx = other_dot["x"] - dot["x"]
                dy = other_dot["y"] - dot["y"]
                dz = other_dot["z"] - dot["z"]
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + EPSILON  # Add a small constant to prevent division by zero

                if 0 < distance < MERGE_DISTANCE:
                    # Mark dots for merging
                    merge_actions.append((i, j))
                    continue

                if distance < MIN_DISTANCE:
                    force = G * dot["mass"] * other_dot["mass"] / MIN_DISTANCE ** 2
                else:
                    force = G * dot["mass"] * other_dot["mass"] / distance ** 2

                # Calculate force components
                fx = force * dx / distance
                fy = force * dy / distance
                fz = force * dz / distance

                total_fx += fx
                total_fy += fy
                total_fz += fz

        # Update velocity and position
        dot["vx"] += total_fx / dot["mass"]
        dot["vy"] += total_fy / dot["mass"]
        dot["vz"] += total_fz / dot["mass"]
        dot["x"] += dot["vx"]
        dot["y"] += dot["vy"]
        dot["z"] += dot["vz"]
        dot["speed"] = math.sqrt(dot["vx"] ** 2 + dot["vy"] ** 2 + dot["vz"] ** 2)  # Calculate speed

        # Check for collisions with borders
        if dot["x"] <= BASE_DOT_RADIUS or dot["x"] >= WIDTH - BASE_DOT_RADIUS:
            dot["vx"] = -dot["vx"] * MOMENTUM_LOSS_FACTOR  # Lose half the momentum
        if dot["y"] <= BASE_DOT_RADIUS or dot["y"] >= HEIGHT - BASE_DOT_RADIUS:
            dot["vy"] = -dot["vy"] * MOMENTUM_LOSS_FACTOR  # Lose half the momentum
        if dot["z"] <= BASE_DOT_RADIUS or dot["z"] >= HEIGHT - BASE_DOT_RADIUS:
            dot["vz"] = -dot["vz"] * MOMENTUM_LOSS_FACTOR  # Lose half the momentum

        # Update trail
        trails[i].append((int(dot["x"]), int(dot["y"]), int(dot["z"])))
        trails[i] = trails[i][-TRAIL_LENGTH:]  # Keep only the last TRAIL_LENGTH points

    # Execute merges
    processed_indices = set()
    merge_actions.sort(key=lambda pair: max(pair[0], pair[1]), reverse=True)

    for i, j in merge_actions:
        if i in processed_indices or j in processed_indices:
            continue
        if i < j:
            to_remove = j
            base = i
        else:
            to_remove = i
            base = j
        # Merge dots
        base_dot = dots[base]
        to_remove_dot = dots[to_remove]
        total_mass = base_dot["mass"] + to_remove_dot["mass"]
        base_dot["vx"] = (base_dot["vx"] * base_dot["mass"] + to_remove_dot["vx"] * to_remove_dot["mass"]) / total_mass
        base_dot["vy"] = (base_dot["vy"] * base_dot["mass"] + to_remove_dot["vy"] * to_remove_dot["mass"]) / total_mass
        base_dot["vz"] = (base_dot["vz"] * base_dot["mass"] + to_remove_dot["vz"] * to_remove_dot["mass"]) / total_mass
        base_dot["x"] = (base_dot["x"] * base_dot["mass"] + to_remove_dot["x"] * to_remove_dot["mass"]) / total_mass
        base_dot["y"] = (base_dot["y"] * base_dot["mass"] + to_remove_dot["y"] * to_remove_dot["mass"]) / total_mass
        base_dot["z"] = (base_dot["z"] * base_dot["mass"] + to_remove_dot["z"] * to_remove_dot["mass"]) / total_mass
        base_dot["mass"] = total_mass
        # Calculate new color proportional to the respective masses
        base_dot["color"] = (
            max(0, min(255, int((base_dot["color"][0] * base_dot["mass"] + to_remove_dot["color"][0] * to_remove_dot["mass"]) / total_mass))),
            max(0, min(255, int((base_dot["color"][1] * base_dot["mass"] + to_remove_dot["color"][1] * to_remove_dot["mass"]) / total_mass))),
            max(0, min(255, int((base_dot["color"][2] * base_dot["mass"] + to_remove_dot["color"][2] * to_remove_dot["mass"]) / total_mass)))
        )
        dots.pop(to_remove)
        trails.pop(to_remove)
        processed_indices.add(i)
        processed_indices.add(j)

    # Clear the window
    window.fill(BLACK)

    # Draw trails for all dots
    for i, trail in enumerate(trails):
        dot_color = dots[i]["color"]  # Get the color of the corresponding dot
        for j in range(len(trail)):
            alpha = int(255 * (TRAIL_FADE_RATE ** j))  # Calculate alpha value based on fade rate
            trail_color = (dot_color[0], dot_color[1], dot_color[2], alpha)  # Use the dot color for the trail
            x, y = trail[j][0], trail[j][1]  # Project 3D coordinates to 2D
            pygame.draw.circle(window, trail_color, (x, y), BASE_DOT_RADIUS)
        if len(dots) < 2:
            # Calculate the total mass of all existing dots
            total_mass = sum(dot['mass'] for dot in dots)
            # Calculate the mass for each of the 100 new dots
            new_dot_mass = total_mass / 1500
            # Spawn 100 new dots with the calculated mass
            for _ in range(100):
                new_dot = generate_new_dot(mass=new_dot_mass)
                dots.append(new_dot)
                trails.append([])  # Add an empty trail for the new dot

    # Draw dots
    for dot in dots:
        # Scale radius according to the mass of the dot
        radius = BASE_DOT_RADIUS + int(math.log(dot["mass"], 10))
        x, y = int(dot["x"]), int(dot["y"])  # Project 3D coordinates to 2D
        pygame.draw.circle(window, dot["color"], (x, y), radius)

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
