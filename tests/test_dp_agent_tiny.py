import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dp_agent import DPAgent, GameState, GameBoard

def test_dp_agent_tiny_board():
    """
    Sanity-check: on a 2×3 board with horizon 2 and γ = 0.9, the value vector V
    returned by DPAgent must satisfy (I − γP) V  ≈  R for the greedy policy.
    """
    # Build agent in DP-only mode
    agent = DPAgent(discount_factor=0.9,
                    use_heuristics=False,
                    use_search=False)

    # Minimal 2×3 Connect-Four board
    board = np.zeros((2, 3))
    game_board = GameBoard(rows=2, cols=3)
    root = GameState(board, 0, game_board)

    # Run plain DP planning with horizon 2
    agent.horizon = 2
    agent._dp_plan_simple(root)

    # Collect state set and corresponding V vector
    states = agent.all_states
    V = np.array([agent.values[s] for s in states])

    # Build transition matrix P and reward vector R for the extracted policy
    P, R = agent.build_PR_matrices(agent.policy, states)

    # Verify Bellman consistency: (I − γP) V ≈ R
    lhs = (np.eye(len(states)) - agent.gamma * P) @ V
    assert np.allclose(lhs, R, atol=1e-6), "Bellman equation not satisfied on tiny board"