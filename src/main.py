import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from collections import defaultdict

GRID_SIZE = 6

# Define the network architecture (must match training)
class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 6, 256),  # 256 units
            nn.ReLU(),
            nn.Linear(256, 256),        # 256 units
            nn.ReLU()
        )
        self.actor = nn.Linear(256, 4)  # 256 units
        self.critic = nn.Linear(256, 1) # 256 units

    def forward(self, x):
        x = self.shared(x)
        return nn.functional.softmax(self.actor(x), dim=-1), self.critic(x)


def load_model(path):
    # Initialize the model
    model = PPONetwork()

    # Load the saved checkpoint
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)

    # Load only the network weights
    model.load_state_dict(checkpoint['network'])

    # Set the model to evaluation mode
    model.eval()

    return model


def generate_walls(goal_pos):
    walls = defaultdict(set)
    goal_r, goal_c = goal_pos
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # W, N, E, S

    # Determine if goal is in a corner
    is_corner = (goal_r in {0, GRID_SIZE - 1}) and (goal_c in {0, GRID_SIZE - 1})
    num_walls = 1 if is_corner else 2  # Only 1 wall for corners

    used_directions = set()

    while len(used_directions) < num_walls:
        dir_idx = np.random.randint(4)
        if dir_idx in used_directions:
            continue

        dr, dc = directions[dir_idx]
        nr, nc = goal_r + dr, goal_c + dc

        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            walls[(goal_r, goal_c)].add(dir_idx)
            opposite_dir = (dir_idx + 2) % 4
            walls[(nr, nc)].add(opposite_dir)
            used_directions.add(dir_idx)

    return walls


def get_state_representation(agent_pos, goal_pos, discovered_walls):
    # Channels: [agent, goal, W_walls, N_walls, E_walls, S_walls]
    state = np.zeros((6, 6, 6), dtype=np.float32)
    state[agent_pos[0], agent_pos[1], 0] = 1.0  # Agent channel
    state[goal_pos[0], goal_pos[1], 1] = 1.0  # Goal channel

    # Wall channels now match move order: W=0, N=1, E=2, S=3
    for (r, c), directions in discovered_walls.items():
        for dir in directions:
            state[r, c, 2 + dir] = 1.0  # Channel 2+0=W, 2+1=N, etc.

    return torch.tensor(state).permute(2, 0, 1).unsqueeze(0)


def move(agent_pos, action, walls):
    moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # W, N, E, S
    dr, dc = moves[action]
    new_pos = (agent_pos[0] + dr, agent_pos[1] + dc)

    # Check if new position is within grid boundaries
    if not (0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE):
        print(f"Out of bounds at {new_pos}")
        return agent_pos, False  # Stay in current position, no wall hit

    # Check for walls
    if action in walls.get(agent_pos, set()):
        print(f"Hit wall at {agent_pos} moving {['W', 'N', 'E', 'S'][action]}")
        return agent_pos, True  # Blocked by wall

    return new_pos, False


def get_path(model, start, goal, walls):
    path = [start]
    current = start
    discovered_walls = defaultdict(set)
    visited = set()  # Track visited positions

    while current != goal and len(path) < GRID_SIZE * 2:  # Max steps
        if current in visited:
            print(f"Stuck in loop at {current}")
            break
        visited.add(current)

        with torch.no_grad():
            state = get_state_representation(current, goal, discovered_walls)
            probs, _ = model(state)
            action = torch.argmax(probs).item()

        new_pos, hit_wall = move(current, action, walls)
        if hit_wall:
            discovered_walls[current].add(action)
        else:
            current = new_pos
        path.append(current)

    return path


def plot(start, goal, path, walls):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Create grid
    grid = np.zeros((6, 6))
    grid[start[0], start[1]] = 1  # Start position
    grid[goal[0], goal[1]] = 2    # Goal position
    plt.imshow(grid, cmap='viridis', origin='upper', extent=[0, 6, 6, 0])

    # Configure ticks to center labels on cells
    ax.set_xticks(np.arange(0.5, 6, 1))  # Column centers
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])
    ax.set_yticks(np.arange(5.5, -0.5, -1))  # Row centers (top to bottom)
    ax.set_yticklabels(['0', '1', '2', '3', '4', '5'])

    # Add grid lines between cells
    ax.set_xticks(np.arange(0, 7, 1), minor=True)
    ax.set_yticks(np.arange(0, 7, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='both', length=0)  # Hide tick marks

    # Plot walls (existing code remains unchanged)
    for (r, c), directions in walls.items():
        for dir in directions:
            if dir == 0:    # West wall (left edge)
                plt.plot([c, c], [r, r+1], 'k-', lw=3)
            elif dir == 1:  # North wall (top edge)
                plt.plot([c, c+1], [r, r], 'k-', lw=3)
            elif dir == 2:  # East wall (right edge)
                plt.plot([c+1, c+1], [r, r+1], 'k-', lw=3)
            elif dir == 3:  # South wall (bottom edge)
                plt.plot([c, c+1], [r+1, r+1], 'k-', lw=3)

    # Plot path (existing code remains unchanged)
    for i in range(1, len(path)):
        y1, x1 = path[i-1]
        y2, x2 = path[i]
        plt.plot([x1+0.5, x2+0.5], [y1+0.5, y2+0.5], 'r-', lw=2)
        plt.plot(x2+0.5, y2+0.5, 'ro', markersize=8)

    plt.title(f"Path from {start} to {goal} with Walls")
    plt.show()


def main():
    model = load_model('ppo_wall_model.pt')  # Use the wall-trained model

    start = (np.random.randint(0, 6), np.random.randint(0, 6))
    goal = start
    while goal == start:
        goal = (np.random.randint(0, 6), np.random.randint(0, 6))

    walls = generate_walls(goal)
    print(f"Testing path from {start} to {goal}")
    path = get_path(model, start, goal, walls)
    print("Path taken:", path)
    plot(start, goal, path, walls)


if __name__ == "__main__":
    main()