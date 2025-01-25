import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Grid parameters
GRID_SIZE = 6
CHANNELS = 2  # Agent and goal positions

# Hyperparameters
EPISODES = 10000
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPONetwork(nn.Module):
    def __init__(self):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(GRID_SIZE * GRID_SIZE * CHANNELS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
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

    def get_state_representation(self, agent_pos, goal_pos):
        state = np.zeros((GRID_SIZE, GRID_SIZE, CHANNELS), dtype=np.float32)
        state[agent_pos[0], agent_pos[1], 0] = 1.0
        state[goal_pos[0], goal_pos[1], 1] = 1.0
        return torch.tensor(state).permute(2, 0, 1).unsqueeze(0).to(device)

    def get_action(self, agent_pos, goal_pos):
        with torch.no_grad():
            state = self.get_state_representation(agent_pos, goal_pos)
            probs, _ = self.network(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), probs.cpu().numpy()[0]

    def train(self, states, actions, old_probs, rewards, dones, next_states):
        # Convert to tensors
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        next_states = torch.stack(next_states).to(device)

        # Calculate advantages
        with torch.no_grad():
            values = self.network(states)[1].squeeze()
            next_values = self.network(next_states)[1].squeeze()
            deltas = rewards + GAMMA * next_values * (1 - dones) - values
            advantages = self.calculate_advantages(deltas.cpu().numpy())
            returns = torch.tensor(advantages, dtype=torch.float32).to(device) + values

        # Create dataset
        dataset = TensorDataset(states, actions, old_probs,
                                torch.tensor(advantages).to(device), returns)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(EPOCHS):
            for batch in loader:
                s_batch, a_batch, op_batch, adv_batch, ret_batch = batch

                # Forward pass
                new_probs, value_pred = self.network(s_batch)
                dist = torch.distributions.Categorical(new_probs)
                new_probs = dist.logits.gather(1, a_batch.unsqueeze(1)).exp().squeeze()

                # Calculate losses
                ratio = new_probs / (op_batch + 1e-8)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy = -dist.entropy().mean()
                critic_loss = torch.mean((ret_batch - value_pred.squeeze()) ** 2)

                total_loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def calculate_advantages(self, deltas):
        advantages = []
        advantage = 0.0
        for delta in reversed(deltas):
            advantage = delta + GAMMA * LAMBDA * advantage
            advantages.insert(0, advantage)
        return np.array(advantages, dtype=np.float32)


def move(agent_pos, action):
    moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    new_pos = (agent_pos[0] + moves[action][0],
               agent_pos[1] + moves[action][1])
    if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE:
        return new_pos
    return agent_pos


def train_ppo():
    agent = PPOAgent()
    episode_rewards = []

    for episode in range(EPISODES):
        start = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        goal = start
        while goal == start:
            goal = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))

        state = start
        trajectory = {
            'states': [],
            'actions': [],
            'old_probs': [],
            'rewards': [],
            'dones': [],
            'next_states': []
        }

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < GRID_SIZE * 2:
            action, probs = agent.get_action(state, goal)
            new_state = move(state, action)

            prev_dist = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
            new_dist = abs(new_state[0] - goal[0]) + abs(new_state[1] - goal[1])
            reward = -0.03 + (prev_dist - new_dist) * 0.1

            if new_state == goal:
                reward = 1.0
                done = True

            trajectory['states'].append(agent.get_state_representation(state, goal))
            trajectory['actions'].append(action)
            trajectory['old_probs'].append(probs[action])
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            trajectory['next_states'].append(agent.get_state_representation(new_state, goal))

            state = new_state
            total_reward += reward
            steps += 1

        agent.train(
            trajectory['states'],
            trajectory['actions'],
            np.array(trajectory['old_probs']),
            np.array(trajectory['rewards']),
            np.array(trajectory['dones']),
            trajectory['next_states']
        )

        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    return agent


def main():
    agent = train_ppo()
    torch.save(agent.network.state_dict(), 'ppo_model.pt')
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
