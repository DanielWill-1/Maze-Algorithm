import random

def generate_maze(width, height):
    # Initialize a grid full of walls (represented by 1s)
    # The actual paths will be 0s. 
    # We make the grid 2x+1 to account for walls between cells.
    grid = [[1 for _ in range(width * 2 + 1)] for _ in range(height * 2 + 1)]
    
    # Directions: (dx, dy) -> moves 2 steps to jump over the "wall" 
    directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    
    def carve_passages_from(cx, cy):
        # Mark current cell as empty path (0)
        grid[cy][cx] = 0
        
        # Shuffle directions to ensure randomized paths
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # Check if the next cell is within bounds and hasn't been visited (is still a wall)
            if 1 <= nx < width * 2 and 1 <= ny < height * 2 and grid[ny][nx] == 1:
                # Knock down the wall between current and next cell
                grid[cy + dy//2][cx + dx//2] = 0
                # Recursively carve from the next cell
                carve_passages_from(nx, ny)

    # Start the maze generation from the top-left cell (1, 1)
    carve_passages_from(1, 1)
    
    # Set Start and End points
    grid[1][0] = 0 # Entrance
    grid[height * 2 - 1][width * 2] = 0 # Exit
    
    return grid

# Generate and print a small 5x5 maze
maze = generate_maze(5, 5)
for row in maze:
    print("".join(["██" if cell == 1 else "  " for cell in row]))