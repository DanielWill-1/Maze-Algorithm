import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class MazeConfig:
    width: int = 5
    height: int = 5
    seed: int = 42

class MazeGenerator:
    def __init__(self, config: MazeConfig):
        self.config = config
        random.seed(config.seed)

    def generate(self) -> List[List[int]]:
        width = self.config.width
        height = self.config.height
        
        # Initialize grid with walls (1)
        grid = [[1 for _ in range(width * 2 + 1)] for _ in range(height * 2 + 1)]
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        
        def carve_passages_from(cx, cy):
            grid[cy][cx] = 0
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < width * 2 and 1 <= ny < height * 2 and grid[ny][nx] == 1:
                    grid[cy + dy//2][cx + dx//2] = 0
                    carve_passages_from(nx, ny)

        carve_passages_from(1, 1)
        
        # Set Start and End points
        grid[1][0] = 0
        grid[height * 2 - 1][width * 2] = 0
        
        return grid
