import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =====================
# Maze Environment
# =====================
class MazeEnv:
    def __init__(self):
        self.grid = np.array([
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 2]
        ])
        self.start = (0, 0)
        self.goal = (4, 4)
        self.reset()

    def reset(self):
        self.pos = self.start
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.grid.shape)
        state[self.pos] = 1
        return state.flatten()

    def step(self, action):
        x, y = self.pos

        if action == 0: x -= 1  # up
        elif action == 1: x += 1  # down
        elif action == 2: y -= 1  # left
        elif action == 3: y += 1  # right

        # boundaries
        if x < 0 or x >= 5 or y < 0 or y >= 5 or self.grid[x, y] == 1:
            return self._get_state(), -1, False

        self.pos = (x, y)

        if self.pos == self.goal:
            return self._get_state(), 10, True

        return self._get_state(), -0.1, False


# =====================
# Policy Network
# =====================
class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# =====================
# Training (REINFORCE)
# =====================
def train():
    env = MazeEnv()
    state_size = 25
    action_size = 4

    policy = PolicyNet(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    gamma = 0.99
    episodes = 500

    for ep in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False

        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy loss
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)

        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Total Reward: {sum(rewards):.2f}")

    return policy


# =====================
# Test Agent
# =====================
def test(policy):
    env = MazeEnv()
    state = env.reset()

    done = False
    steps = 0

    print("\nPath:")
    while not done and steps < 50:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)

        action = torch.argmax(probs).item()
        state, _, done = env.step(action)

        print(env.pos)
        steps += 1


# =====================
# Run
# =====================
if __name__ == "__main__":
    policy = train()
    test(policy)