import torch
from envs.maze_env import MazeEnv
from utils.maze_generator import MazeConfig
from train import PolicyNet

def evaluate(policy, episodes=1):
    config = MazeConfig(width=5, height=5)
    env = MazeEnv(config)
    
    print("\nEvaluating trained agent...")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        
        print(f"Episode {ep+1}:")
        while not done and steps < 50:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            action = torch.argmax(probs).item()
            
            state, reward, done = env.step(action)
            steps += 1
            print(f"Step {steps}: Position {env.pos}, Reward {reward}")
            
        if done:
            print("Goal reached!")
        else:
            print("Failed to reach goal.")

if __name__ == "__main__":
    # In a real scenario, you would load a saved model here.
    # For now, we will assume the policy object is passed from a train call
    # if this were part of a larger CLI.
    # Since we can't save/load easily yet, we train a quick one.
    from train import train
    policy = train(episodes=100)
    evaluate(policy)
