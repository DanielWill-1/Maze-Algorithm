# Project Plan: RL Maze Solver

## 1. Current Repository Assessment

### Existing Files (Moved to `backup/`)
- `maze.py`: Basic DFS-based maze generator.
- `rl.py`: Q-Learning implementation on a hardcoded 4x4 grid.
- `gradient.py`: Deep Policy Gradient (REINFORCE) implementation on a hardcoded 5x5 grid.

### Strengths
- Functional prototypes for both procedural content generation and RL agents exist.
- Basic logic for maze generation and RL training is established.

### Weaknesses / Technical Debt
- **Fragmented Codebase:** No shared structure or common interface.
- **Hardcoding:** RL environments are hardcoded grids, not dynamic.
- **Lack of Abstraction:** RL algorithms and environment logic are tightly coupled.
- **Scalability:** The current setup is not designed for different algorithms, larger mazes, or experiment tracking.

## 2. Target Architecture

- **`envs/`**: Contains environment wrappers that adhere to a standardized interface.
- **`agents/`**: Contains agent policy networks/strategies.
- **`algorithms/`**: Contains RL algorithm implementations (DQN, PPO, etc.).
- **`utils/`**: Shared utilities (maze generator, logging, metrics).
- **`configs/`**: Configuration files for experiments.
- **`outputs/`**: Training logs, model checkpoints, and visualizations.

## 3. Maze Generation Improvements (`utils/maze_generator.py`)
- **Configurability:** Use dataclasses for grid size, difficulty, and seeding.
- **Validation:** Add a post-generation pathfinding check (e.g., BFS) to ensure a guaranteed solution.
- **Quality:** Improve code readability, type hinting, and add logging.

## 4. RL Roadmap
- **Interface:** Standardize `reset()`, `step()`, and `render()` for all environments.
- **State Representation:** Decouple environment state from agent input.
- **Pipeline:** Build a robust `train.py` that loads configs, instantiates an environment and an agent, and runs the training loop with logging.

## 5. Implementation Roadmap
1. **Phase 1:** Scaffold structure, create `MazeEnv`, and refine `maze_generator`.
2. **Phase 2:** Implement a unified `train.py` pipeline.
3. **Phase 3:** Introduce advanced algorithms (DQN/PPO) and logging.
4. **Phase 4:** Evaluation framework and metric tracking.
