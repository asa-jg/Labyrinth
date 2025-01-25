import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

# Grid parameters
GRID_SIZE = 6
CHANNELS = 6  # Agent, goal + 4 wall directions (W, N, E, S)

# Hyperparameters (⚠️ adjusted values)
EPISODES = 30000
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01  # ⚠️ Increased for better exploration
BATCH_SIZE = 64
EPOCHS = 10 # ⚠️ Reduced for more stable updates
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPONetwork(nn.Module):
    def __init__(self):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(GRID_SIZE * GRID_SIZE * CHANNELS, 128),  # ⚠️ Increased capacity
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return nn.functional.softmax(self.actor(x), dim=-1), self.critic(x)


class PPOAgent:
    def __init__(self):
        self.network = PPONetwork().to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def get_state_representation(self, agent_pos, goal_pos, discovered_walls):
        state = np.zeros((GRID_SIZE, GRID_SIZE, CHANNELS), dtype=np.float32)
        state[agent_pos[0], agent_pos[1], 0] = 1.0
        state[goal_pos[0], goal_pos[1], 1] = 1.0

        for (r, c), directions in discovered_walls.items():
            for dir in directions:
                state[r, c, 2 + dir] = 1.0
        return torch.tensor(state).permute(2, 0, 1).unsqueeze(0).to(device)

    def get_action(self, agent_pos, goal_pos, discovered_walls):
        with torch.no_grad():
            state = self.get_state_representation(agent_pos, goal_pos, discovered_walls)
            raw_probs, _ = self.network(state)

            # Generate action mask
            mask = self._get_action_mask(agent_pos, discovered_walls)
            masked_probs = raw_probs * torch.tensor(mask, dtype=torch.float32).to(device)

            # Check if all actions are masked (invalid)
            if masked_probs.sum().item() == 0:
                # If all actions are invalid, fall back to uniform distribution
                masked_probs = torch.ones_like(raw_probs) / raw_probs.size(-1)
            else:
                # Normalize probabilities
                masked_probs /= masked_probs.sum()

            dist = torch.distributions.Categorical(masked_probs)
            action = dist.sample()
            return action.item(), raw_probs.cpu().numpy()[0]

    def _get_action_mask(self, pos, discovered_walls):
        mask = [1, 1, 1, 1]
        r, c = pos
        if c == 0 or 0 in discovered_walls.get((r, c), []): mask[0] = 0
        if r == 0 or 1 in discovered_walls.get((r, c), []): mask[1] = 0
        if c == GRID_SIZE - 1 or 2 in discovered_walls.get((r, c), []): mask[2] = 0
        if r == GRID_SIZE - 1 or 3 in discovered_walls.get((r, c), []): mask[3] = 0
        return mask

    def train(self, states, actions, old_probs, rewards, dones, next_states):
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        next_states = torch.stack(next_states).to(device)

        with torch.no_grad():
            values = self.network(states)[1].squeeze()
            next_values = self.network(next_states)[1].squeeze()
            deltas = rewards + GAMMA * next_values * (1 - dones) - values
            advantages = self.calculate_advantages(deltas.cpu().numpy())
            returns = torch.tensor(advantages, dtype=torch.float32).to(device) + values

        dataset = TensorDataset(states, actions, old_probs,
                                torch.tensor(advantages).to(device), returns)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(EPOCHS):
            for batch in loader:
                s_batch, a_batch, op_batch, adv_batch, ret_batch = batch

                new_probs, value_pred = self.network(s_batch)
                dist = torch.distributions.Categorical(new_probs)
                log_probs = dist.log_prob(a_batch)

                ratio = (log_probs - torch.log(op_batch + 1e-8)).exp()
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy = dist.entropy().mean()
                critic_loss = torch.mean((ret_batch - value_pred.squeeze()) ** 2)

                total_loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)  # ⚠️ Gradient clipping
                self.optimizer.step()

    def calculate_advantages(self, deltas):
        advantages = []
        advantage = 0.0
        for delta in reversed(deltas):
            advantage = delta + GAMMA * LAMBDA * advantage
            advantages.insert(0, advantage)
        return np.array(advantages, dtype=np.float32)


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


def move(agent_pos, action, walls):
    moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    dr, dc = moves[action]
    new_pos = (agent_pos[0] + dr, agent_pos[1] + dc)
    return (agent_pos, True) if action in walls.get(agent_pos, set()) else (new_pos, False)


def train_ppo(load_path=None):
    agent = PPOAgent()
    if load_path:
        print(f"Loading model from {load_path}...")
        agent.load(load_path)  # ⚠️ Using new load method

    episode_rewards = []
    best_avg = float('-inf')

    for episode in range(EPISODES):
        start = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        goal = start
        while goal == start:
            goal = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))

        walls = generate_walls(goal)
        discovered_walls = defaultdict(set)
        state = start
        trajectory = {"states": [], "actions": [], "old_probs": [],
                      "rewards": [], "dones": [], "next_states": []}

        total_reward = 0
        done = False
        max_steps = (abs(start[0] - goal[0]) + abs(start[1] - goal[1])) * 2 + 10  # ⚠️ Dynamic step limit
        steps = 0

        while not done and steps < max_steps:
            action, raw_probs = agent.get_action(state, goal, discovered_walls)  # ⚠️ Get raw_probs
            new_pos, hit_wall = move(state, action, walls)

            if hit_wall:
                discovered_walls[state].add(action)
                new_pos = state
                reward = -0.3  # ⚠️ Stronger wall penalty
            else:
                dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
                reward = (-0.05 + (steps * -0.01) +  # ⚠️ Adjusted rewards
                          (0.5 if dist < 3 else 0))
                if new_pos == goal:
                    reward = 5.0  # ⚠️ Increased goal reward
                    done = True

            trajectory["states"].append(agent.get_state_representation(state, goal, discovered_walls))
            trajectory["actions"].append(action)
            trajectory["old_probs"].append(raw_probs[action])  # ⚠️ Store raw probabilities
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)
            trajectory["next_states"].append(agent.get_state_representation(new_pos, goal, discovered_walls))

            state = new_pos
            total_reward += reward
            steps += 1

        agent.train(**trajectory)
        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

            if avg_reward > best_avg:  # ⚠️ Save best model
                best_avg = avg_reward
                agent.save("ppo_wall_model.pt")
                print(f"New best model saved with avg reward {best_avg:.2f}")

    return agent


def main():
    agent = train_ppo(load_path="ppo_model.pt" if True else None)  # Set True to load
    agent.save("ppo_final_model.pt")
    print("Training complete and model saved!")


if __name__ == "__main__":
    main()