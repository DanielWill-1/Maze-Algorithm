import numpy as np
from typing import Tuple, List
from utils.maze_generator import MazeGenerator, MazeConfig

class MazeEnv:
    def __init__(self, config: MazeConfig = MazeConfig()):
        self.generator = MazeGenerator(config)
        self.grid = np.array(self.generator.generate())
        self.start = (1, 0)
        self.goal = (self.grid.shape[0] - 2, self.grid.shape[1] - 1)
        self.pos = self.start
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space_n = 4
        self.state_shape = self.grid.shape

    def reset(self):
        self.grid = np.array(self.generator.generate())
        self.pos = self.start
        return self._get_state()

    def _get_state(self):
        # Return flattened grid with agent position marked as 2
        state = self.grid.copy()
        state[self.pos] = 2
        return state.flatten()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        x, y = self.pos
        
        if action == 0: x -= 1 # Up
        elif action == 1: x += 1 # Down
        elif action == 2: y -= 1 # Left
        elif action == 3: y += 1 # Right
        
        # Check bounds
        if not (0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]):
            return self._get_state(), -1.0, False
            
        # Check walls
        if self.grid[x, y] == 1:
            return self._get_state(), -1.0, False
            
        self.pos = (x, y)
        
        if self.pos == self.goal:
            return self._get_state(), 10.0, True
            
        return self._get_state(), -0.1, False
