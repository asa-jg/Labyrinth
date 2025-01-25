import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn


# Define the network architecture (must match training)
class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return nn.functional.softmax(self.actor(x), dim=-1), self.critic(x)


def load_model(path):
    model = PPONetwork()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_state_representation(agent_pos, goal_pos):
    state = np.zeros((6, 6, 2), dtype=np.float32)
    state[agent_pos[0], agent_pos[1], 0] = 1.0
    state[goal_pos[0], goal_pos[1], 1] = 1.0
    return torch.tensor(state).permute(2, 0, 1).unsqueeze(0)


def move(agent_pos, action):
    moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    new_pos = (agent_pos[0] + moves[action][0],
               agent_pos[1] + moves[action][1])
    if 0 <= new_pos[0] < 6 and 0 <= new_pos[1] < 6:
        return new_pos
    return agent_pos


def get_path(model, start, goal):
    path = [start]
    current = start
    while current != goal and len(path) < 12:
        with torch.no_grad():
            state = get_state_representation(current, goal)
            probs, _ = model(state)
            action = torch.argmax(probs).item()
        current = move(current, action)
        path.append(current)
    return path


def plot_grid_and_path(start, goal, path):
    grid = np.zeros((6, 6))
    grid[start[0], start[1]] = 1
    grid[goal[0], goal[1]] = 2

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='viridis', origin='upper')

    for i in range(1, len(path)):
        plt.plot([path[i - 1][1], path[i][1]],
                 [path[i - 1][0], path[i][0]],
                 'r-', linewidth=2)
        plt.plot(path[i][1], path[i][0], 'ro', markersize=8)

    plt.colorbar(ticks=[0, 1, 2], label='Start (1) / Goal (2)')
    plt.title(f"Path from {start} to {goal}")
    plt.show()


def main():
    # Load trained model
    model = load_model('ppo_model.pt')

    # Test with random positions
    start = (np.random.randint(0, 6), np.random.randint(0, 6))
    goal = start
    while goal == start:
        goal = (np.random.randint(0, 6), np.random.randint(0, 6))

    print(f"Testing path from {start} to {goal}")
    path = get_path(model, start, goal)
    print("Path taken:", path)
    plot_grid_and_path(start, goal, path)


if __name__ == "__main__":
    main()
