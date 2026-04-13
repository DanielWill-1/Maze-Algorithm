import numpy as np
import random

# ==========================================
# 1. DEFINE THE ENVIRONMENT (The Maze)
# ==========================================
# 0 = Safe path, 1 = Obstacle (wall/hole), 2 = Goal
maze = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 0, 2]
])

START = (0, 0) # Top-left
GOAL = (3, 3)  # Bottom-right

# Define possible actions: 0: Up, 1: Right, 2: Down, 3: Left
actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def step(state, action_idx):
    """Takes a state and an action, and returns the next state, reward, and if it's done."""
    move = actions[action_idx]
    next_state = (state[0] + move[0], state[1] + move[1])

    # Check if the agent hits the outer boundary
    if next_state[0] < 0 or next_state[0] >= 4 or next_state[1] < 0 or next_state[1] >= 4:
        return state, -1, False # Penalty for hitting a wall, stay in same state

    # Check what is in the next state cell
    cell = maze[next_state]
    if cell == 1:
        return state, -5, False # Big penalty for hitting an obstacle
    elif cell == 2:
        return next_state, 10, True # Reached the goal! 
    else:
        return next_state, -0.1, False # Small penalty for a normal step to encourage speed

# ==========================================
# 2. INITIALIZE THE Q-TABLE
# ==========================================
# The Q-table stores the "quality" of taking a specific action in a specific state.
# Size: 4 rows x 4 columns x 4 possible actions. Initialized to zeros.
q_table = np.zeros((4, 4, 4))

# ==========================================
# 3. HYPERPARAMETERS
# ==========================================
alpha = 0.1      # Learning rate: How much new info overrides old info
gamma = 0.9      # Discount factor: How much the agent cares about future rewards vs immediate ones
epsilon = 0.2    # Exploration rate: 20% chance to take a random action to explore the maze
episodes = 1000  # Number of times the agent attempts the maze

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("Training agent...")
for episode in range(episodes):
    state = START
    done = False
    
    while not done:
        # Epsilon-Greedy Strategy: Decide whether to explore or exploit
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3) # Explore: Pick a random action
        else:
            action = np.argmax(q_table[state[0], state[1]]) # Exploit: Pick the best known action
            
        next_state, reward, done = step(state, action)
        
        # Update the Q-Value using the Bellman Equation
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        
        # The core math of Q-Learning
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state[0], state[1], action] = new_value
        
        state = next_state

print("Training complete!\n")

# ==========================================
# 5. TESTING THE TRAINED AGENT
# ==========================================
print("Testing the trained agent's path:")
state = START
path = [state]
done = False
steps = 0

# Give the agent a max of 20 steps to prove it knows the way
while not done and steps < 20:
    # Always exploit the best path now (no random exploration)
    action = np.argmax(q_table[state[0], state[1]])
    state, _, done = step(state, action)
    path.append(state)
    steps += 1

for p in path:
    print(f"-> {p}")

if path[-1] == GOAL:
    print("\nSuccess! The agent found the goal.")
else:
    print("\nThe agent got lost. It might need more training episodes.")