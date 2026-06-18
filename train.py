import torch
import torch.nn as nn
import torch.optim as optim
from envs.maze_env import MazeEnv
from utils.maze_generator import MazeConfig

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

def train(episodes=500):
    config = MazeConfig(width=5, height=5)
    env = MazeEnv(config)
    
    state_size = env.grid.size
    action_size = env.action_space_n
    
    policy = PolicyNet(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    gamma = 0.99
    
    print(f"Training agent for {episodes} episodes...")
    
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
        if len(returns) > 1:
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
            
    print("Training complete.")
    return policy

if __name__ == "__main__":
    train()
