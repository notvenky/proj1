import numpy as np
import matplotlib.pyplot as plt
import imageio

# Parameters
num_frames = 60  # Number of frames in the GIF
link_length = 50  # Length of each link

# Function to calculate snake link positions
def calculate_snake_positions(num_links, frame_num):
    angles = np.sin(np.linspace(0, 2 * np.pi, num_links) + np.pi * frame_num / num_frames)
    x_positions = np.cumsum(np.cos(angles) * link_length)
    y_positions = np.cumsum(np.sin(angles) * link_length)
    return x_positions, y_positions

# Function for salamander-like movement
def calculate_salamander_positions(num_links, frame_num):
    # Salamander body will have a waving motion, limbs will be static
    angles = np.sin(np.linspace(0, 2 * np.pi, num_links) + np.pi * frame_num / num_frames) / 3
    x_positions = np.cumsum(np.cos(angles) * link_length)
    y_positions = np.cumsum(np.sin(angles) * link_length)
    return x_positions, y_positions

# Function for quadruped movement
def calculate_quadruped_positions(num_links, frame_num):
    # Quadruped body will be more rigid, with slight up and down movement
    angles = np.sin(np.linspace(0, 2 * np.pi, num_links) + np.pi * frame_num / num_frames) / 5
    x_positions = np.cumsum(np.cos(angles) * link_length)
    y_positions = np.cumsum(np.sin(angles) * link_length)
    return x_positions, y_positions

# Function for spider movement
def calculate_spider_positions(num_links, frame_num):
    # Spider body will be static, legs will have a waving motion
    angles = np.sin(np.linspace(0, 2 * np.pi, num_links) + np.pi * frame_num / num_frames) / 2
    x_positions = np.cumsum(np.cos(angles) * link_length)
    y_positions = np.cumsum(np.sin(angles) * link_length)
    return x_positions, y_positions

# Create frames
frames = []
for frame_num in range(num_frames):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns grid
    axs = axs.flatten()

    # First row: snakes
    for i in range(3):
        num_links = (i + 1) * 4 + 2  # 6, 8, 12 links
        x_positions, y_positions = calculate_snake_positions(num_links, frame_num)
        axs[i].plot(x_positions, y_positions, marker='o')
        axs[i].set_xlim(-300, 300)
        axs[i].set_ylim(-300, 300)
        axs[i].axis('off')

    # Second row: salamander, quadruped, spider
    morphologies = [calculate_salamander_positions, calculate_quadruped_positions, calculate_spider_positions]
    for i, morphology in enumerate(morphologies):
        x_positions, y_positions = morphology(8, frame_num)
        axs[i + 3].plot(x_positions, y_positions, marker='o')
        axs[i + 3].set_xlim(-300, 300)
        axs[i + 3].set_ylim(-300, 300)
        axs[i + 3].axis('off')

    plt.tight_layout()

    # Save frame
    filename = f'frame_{frame_num}.png'
    plt.savefig(filename)
    plt.close()
    frames.append(imageio.imread(filename))

# Create GIF
imageio.mimsave('robot_morphologies_grid.gif', frames, fps=10)

# Clean up (optional)
import os
for filename in frames:
    os.remove(filename)