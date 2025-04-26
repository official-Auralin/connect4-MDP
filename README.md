# Connect4 MDP - Solving Connect Four with Markov Decision Processes

<div align="center">
<img src="./images/logo/c4.gif" alt="Connect Four Logo">
</div>

## About

This project implements a Connect Four game with an AI agent that uses Markov Decision Processes (MDPs) and linear algebra to make optimal decisions. The AI uses value iteration and direct linear system solving to calculate the optimal policy, making it a powerful opponent that can see several moves ahead.

The original Connect Four game was created by [Mayank Singh (code-monk08)](https://github.com/code-monk08/connect-four). This project extends the original by adding an MDP-based AI opponent using dynamic programming and linear algebra techniques.

## Mathematical Foundation

### Markov Decision Processes (MDPs)

An MDP is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. Formally, an MDP consists of:

- **State space (S)**: All possible game configurations
- **Action space (A)**: Legal moves (columns) for each state
- **Transition function (T)**: Deterministic in Connect Four - placing a piece results in a specific new state
- **Reward function (R)**: Values assigned to states (+200 for win, -200 for loss, 0 for draw)
- **Discount factor (γ)**: Values future rewards less than immediate ones (default: 0.95)

### The Bellman Equation

The value of a state is defined by the Bellman equation:

```
V(s) = max_a [ R(s,a) + γ * V(T(s,a)) ]
```

Where:
- V(s) is the value of state s
- R(s,a) is the reward for taking action a in state s
- T(s,a) is the next state after taking action a in state s
- γ is the discount factor

### Linear Algebra Formulation

For finite MDPs, we can represent the Bellman equation as a system of linear equations:

```
V = R + γPV
```

Which can be rearranged as:

```
(I - γP)V = R
```

Where:
- V is the vector of state values
- R is the vector of rewards
- P is the transition probability matrix
- I is the identity matrix

The solution is:

```
V = (I - γP)⁻¹R
```

This direct matrix inversion is more efficient than iterative methods for certain problem sizes and allows for exact solutions to the MDP.

### Value Iteration vs. Linear System Solving

This project implements both classic value iteration (an iterative method) and direct linear system solving:

1. **Value Iteration**: Iteratively updates state values until convergence
   - Pros: Works well for large state spaces, low memory requirements
   - Cons: May require many iterations to converge

2. **Linear System Solving**: Directly solves (I - γP)V = R
   - Pros: Gets exact solution in one step, faster for small to medium problems
   - Cons: Requires more memory, less practical for very large state spaces

## Features

- Full Connect Four game implementation with customizable board sizes
- Dynamic Programming MDP agent with configurable parameters
- Value iteration and linear algebra solving approaches
- Interactive game modes: Player vs Player, Player vs Agent, Agent vs Agent
- Supports multiple board sizes (standard 7×6 Connect 4 or smaller variants)
- Detailed Bellman equation visualization for educational purposes
- Unit tests and parameter sweep scripts for validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/official-Auralin/connect4-MDP.git
cd connect4-MDP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Game

Launch the game with the GUI interface:
```bash
python game.py
```

### Testing the MDP Agent

Test the agent in isolation:
```bash
python -c "from dp_agent import DPAgent; agent = DPAgent(); agent.run_toy_problem(rows=3, cols=4, horizon=6)"
```

Analyze a specific position:
```bash
python -c "from dp_agent import DPAgent, GameState, GameBoard; import numpy as np; board = np.zeros((3, 4)); game_board = GameBoard(rows=3, cols=4); state = GameState(board, 0, game_board); agent = DPAgent(); agent.analyze_position(state)"
```

### Running Tests

Run the unit tests to verify the MDP implementation:
```bash
pytest tests/test_dp_agent_tiny.py
```

### Parameter Sweep

Run the parameter sweep script to analyze performance with different settings:
```bash
python scripts/param_sweep.py
```

## Implementation Details

### MDP Formulation for Connect Four

In our implementation, the Connect Four MDP is defined as:

- **State space (S)**: Each `GameState` encodes:
  - An `r × c` board (r∈[2,6], c∈[3,7]) with 0 = empty, 1 = Player1 (P1) piece, 2 = Player2 (P2)
  - `turn ∈ {0,1}` (0 → P1 to play, 1 → P2)
  - A reference to the `GameBoard` object

- **Action space (A(s))**: Legal columns that are not full in state s

- **Transition (T)**: Deterministic:
  `s' = s.apply_action(a)` drops the current player's piece in column a

- **Reward (R)**: Deterministic, zero-sum:
  - +200 if P2 wins in s'
  - -200 if P1 wins in s'
  - 0 if draw
  - -0.01 step cost otherwise (when use_heuristics=False)

- **Discount factor (γ)**: Configurable (default 0.95)

### DP Agent Pipeline

1. **Enumerate** reachable states up to horizon H
2. **Set global index** for states
3. **Initialize** value function
4. **Value-iteration** until convergence
5. **Greedy policy extraction**
6. **Output** state values and optimal actions

## Differences from Original Project

Our project extends the original Connect Four implementation in several key ways:


1. **AI Opponent**: Added an MDP-based AI that uses dynamic programming for optimal play
2. **Mathematical Framework**: Implemented the Bellman equation and linear system solving
3. **Configurable Parameters**: Added tunable discount factor, horizon, and other MDP parameters
4. **Theoretical Foundation**: Provided rigorous mathematical basis for AI decision-making
5. **Educational Value**: Added visualization of Bellman backups for educational purposes

**To see the original README.md**: view [README_old.md](./README_old.md) or visit the original repo at [code-monk08/connect-four](https://github.com/code-monk08/connect-four) for the latest version.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Connect Four implementation by [Mayank Singh (code-monk08)](https://github.com/code-monk08/connect-four)
- The MDP framework is inspired by classical works in reinforcement learning and dynamic programming by Richard Bellman and other pioneers in the field 