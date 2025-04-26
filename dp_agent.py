from typing import Any, Dict, List, Tuple, Set, Optional
import numpy as np
import copy
import random
import time
import math
from game_board import GameBoard
from game_state import GameState

"""
--------------------------------------------------------------------------
Connect‑4 MDP  —  Formal definition & DP‑only pipeline
--------------------------------------------------------------------------

Markov Decision Process
-----------------------
• **State space  (S)**  –  Each `GameState` encodes:
    – an `r × c` board (r∈[2,6], c∈[3,7]) with 0 = empty, 1 = P1 piece, 2 = P2  
    – `turn ∈ {0,1}`  (0 → P1 to play, 1 → P2)  
    – a reference to the `GameBoard` object (rows, cols, win_condition).

• **Action space  (A(s))**  –  Legal columns that are not full in state *s*.

• **Transition  (T)**  –  Deterministic.
    `s' = s.apply_action(a)` drops the current player’s piece in column *a*.

• **Reward  (R)**  –  Deterministic, zero‑sum:  
    *  +200 if P2 wins in *s'*,  
    *  –200 if P1 wins in *s'*,  
    *    0  if draw,  
    *  –0.01 step cost otherwise (when `use_heuristics=False`).  

• **Discount factor  (γ)**  –  Configurable (default 0.95 in DP‑only mode).

Finite‑horizon truncation
-------------------------
Because Connect‑4 can last up to 42 plies on a 6×7 board, we approximate the
infinite‑horizon MDP by **breadth‑first enumeration up to depth *H*** (`self.horizon`)
from the current root.  All states beyond depth *H* are ignored; this yields a
finite state set |S| that scales roughly O(b^H) with average branching factor *b*.

DP‑only evaluation pipeline
---------------------------
1. **Enumerate** reachable states ≤ *H*  →  `self.enumerate_reachable_states`.  
2. **Set global index**               →  `_set_global_state_index`.  
3. **Initialize** `V(s)=0`, lock terminal rewards.  
4. **Value‑iteration** over `states` until  Δ < ε (stores `vi_sweeps`, `last_vi_delta`).  
5. **Greedy policy extraction**       (stores `policy_updates_last`).  
6. **Instrumentation** print:  |S|, sweeps, final Δ, policy updates.

Unit test  &  sweep scripts
---------------------------
* `tests/test_dp_agent_tiny.py`  verifies that the computed *V* satisfies  
  `(I − γP)V = R` on a 2×3 board, horizon 2.
* `scripts/param_sweep.py`  logs scaling of |S|, run‑time, and convergence stats
  for γ ∈ {0.7,0.8,0.9,0.95}, H ∈ {2..6} on a 3×4 board.

Set `use_search=True` / `use_heuristics=True` to re‑enable progressive beam
search and positional bonuses for strong play; leave them **False** for pure
linear‑algebra experiments.
--------------------------------------------------------------------------
"""

# TODO: add an initial state setting, so we can test the agent in terminal and near terminal states with fewer available moves
# TODO: figure out if the recursive nature of the bellman equation is supposed to reduce to a smaller system for each turn. (what we have seems correct)
# TODO: fill compute_bellman_equation with the correct equations, currently just returns a placeholder - this will let us see the linear systems for the 7 available moves. 
# TODO: imshow in matplotlib can be used to visualize the board takes in a numpy array and displays it as a grid, will pull up a secondary GUI. 

class DPAgent:
    """
    Dynamic Programming agent for Connect4.
    Uses online policy iteration with limited horizon and beam search
    to compute optimal policies for the current game state.
    """
    
    def __init__(self, discount_factor: float = 0.9995, epsilon: float = 0.001, horizon: int = 18, beam_width: int = 800,
                 use_heuristics: bool = True, use_search: bool = True, verbose: bool = True):
        """
        Initialize the DP agent.
        
        Args:
            discount_factor: The discount factor for future rewards (gamma)
            epsilon: The convergence threshold for value iteration
            horizon: The maximum depth to explore from current state
            beam_width: The maximum number of states to consider at each depth
            use_heuristics: Toggle for positional‑pattern heuristic rewards
        """
        self.use_search = use_search
        self.gamma = discount_factor
        if not use_heuristics and discount_factor > 0.99:
            print("Warning: High γ combined with simple rewards may slow convergence; "
                  "consider setting γ≈0.9.")
        self.epsilon = epsilon
        self.horizon = horizon
        self.beam_width = beam_width
        self.use_heuristics = use_heuristics  # toggle for positional‑pattern rewards
        self.V0 = 0.0  # Initial value for all states
        self.values = {}  # State -> value mapping (V(s))
        self.policy = {}  # State -> action mapping
        self.linear_systems = {}  # State -> linear system mapping

        # Cache for transposition table
        self.eval_cache = {}  # State hash -> reward value
        self.cache_hits = 0
        self.cache_misses = 0

        # Statistics for analysis
        self.states_explored = 0
        self.iterations_performed = 0
        self.visits = {}  # Count state visits for improved exploration

        # ------------------------------------------------------------------
        # Instrumentation counters
        # ------------------------------------------------------------------
        self.vi_sweeps: int = 0           # value-iteration sweeps in last run
        self.last_vi_delta: float = 0.0   # final delta from last value_iteration
        self.policy_updates_last: int = 0 # how many states changed action last extraction

        # ------------------------------------------------------------------
        # Global state bookkeeping (used in DP‑only mode)
        # ------------------------------------------------------------------
        self.all_states: Set[GameState] = set()
        self.state_index: Dict[GameState, int] = {}

        self.verbose = verbose         # master flag for console output

        # Initialize the agent
        self.reset()
        print(f"Agent initialized. Ready for online learning with horizon={horizon}, beam_width={beam_width}, gamma={discount_factor}.")
        
    def set_epsilon(self, epsilon: float) -> None:
        """Set the convergence threshold for value iteration."""
        self.epsilon = epsilon
        
    def set_discount_factor(self, discount_factor: float) -> None:
        """Set the discount factor for future rewards."""
        self.gamma = discount_factor
        
    def set_horizon(self, horizon: int) -> None:
        """Set the maximum depth to explore from current state."""
        self.horizon = horizon
        
    def set_beam_width(self, beam_width: int) -> None:
        """Set the maximum number of states to consider at each depth."""
        self.beam_width = beam_width

    def set_use_heuristics(self, flag: bool) -> None:
        """Enable or disable positional‑pattern heuristic rewards."""
        self.use_heuristics = flag
    
    def set_use_search(self, flag: bool) -> None:
        """Enable/disable progressive beam search and defensive overrides."""
        self.use_search = flag

    def set_verbose(self, flag: bool) -> None:
        """Enable or disable most console printing."""
        self.verbose = flag

    def _vprint(self, *args, **kwargs):
        """Verbose‑controlled print."""
        if self.verbose:
            print(*args, **kwargs)

    def _initialize_state(self, state: GameState) -> None:
        """Initialize a new state with default values and policy."""
        if state not in self.values:
            self.values[state] = self.V0
            self.policy[state] = None  # No policy yet for this state
            
    def print_linear_system(self, game_state: Dict) -> None:
        """
        Compute and print the Bellman candidates for the given game state using the Bellman optimality backup.
        This can be called regardless of whose turn it is.
        
        Args:
            game_state: The current state of the game
        """
        try:
            # Convert dictionary game state to GameState
            state = self._convert_to_game_state(game_state)
            current_player = state.turn + 1
            player_perspective = "MAXIMIZE" if current_player == 2 else "MINIMIZE"

            print(f"\n=== BELLMAN CANDIDATES FOR PLAYER {current_player} ({player_perspective}) ===")

            candidates = self.get_bellman_candidates(state)
            if not candidates:
                print("No valid actions.")
                return

            for action in sorted(candidates):
                c = candidates[action]
                print(f"Column {action+1}: "
                      f"R={c['reward']:+6.2f}  "
                      f"+ γ·V(s')={self.gamma:.4f}·{c['future_value']:+6.2f}  "
                      f"⇒ Q={c['q_value']:+7.2f}"
                      f"{'  (terminal)' if c['is_terminal'] else ''}")

            # Pick best/min action purely from these Q values
            if current_player == 2:     # maximize
                best = max(candidates.items(), key=lambda kv: kv[1]['q_value'])[0]
            else:                       # minimize
                best = min(candidates.items(), key=lambda kv: kv[1]['q_value'])[0]

            print(f"→ Best action under one‑step backup: Column {best+1}")
            print("=== END CANDIDATES ===\n")
        except Exception as e:
            # If there's an error, print a more graceful message
            print(f"\n=== BELLMAN CANDIDATES FOR PLAYER {state.turn + 1} ===")
            print(f"Unable to generate Bellman candidates: {str(e)}")
            print(f"=== END CANDIDATES ===\n")
        
    def choose_action(self, game_state: Dict) -> int:
        """Choose an action based on the current state."""
        # Convert dictionary game state to our GameState object
        state = self._convert_to_game_state(game_state)
        
        # Check if this is a small board (toy problem)
        num_rows, num_cols = state.board.shape
        is_toy_problem = (num_rows <= 3 and num_cols <= 4)
        
        if is_toy_problem:
            print("Detected small board - using linear algebra approach")
            policy, values = self.run_toy_problem(num_rows, num_cols, horizon=3)
            if state in policy:
                return policy[state]
            # Fall back to regular method if policy doesn't have this state
        
        # Existing choose_action logic...
        # (rest of the method unchanged)
        start_time = time.time()
        
        valid_actions = state.get_valid_actions()
        current_player = state.turn + 1  # Convert from 0/1 to 1/2
        player_perspective = "MAXIMIZE" if current_player == 2 else "MINIMIZE"
        
        self._vprint(f"\nAgent is Player {current_player} (perspective: {player_perspective})")
        if not self.use_search:
            self._vprint("Â  [search extras DISABLED â€“ DP-only mode]")
        
        # If no valid actions, return -1 (should never happen in a normal game)
        if not valid_actions:
            return -1
            
        # IMPORTANT: We no longer skip the MDP for hardcoded openings or defensive moves
        # This ensures the mathematical structure of the MDP is preserved
        
        # Comment out hardcoded opening moves to ensure MDP is always used
        # empty_count = np.count_nonzero(state.board == 0)
        # if empty_count >= 41:  # First move or nearly first move
        #     # If center is available, always take it
        #     if 3 in valid_actions:
        #         print("Opening move: Taking center column")
        #         return 3
        #     # If center is taken, take adjacent column
        #     elif 2 in valid_actions:
        #         print("Opening move: Taking column adjacent to center")
        #         return 2
                
        # PHASE 1: STRATEGIC SEARCH - Always perform full policy iteration first
        if self.use_search:
            print("Performing online policy iteration with progressive beam widening...")
            self.online_policy_iteration_progressive(state)
        else:
            print("Performing pure DP planning...")
            self._dp_plan_simple(state)
        
        # Get the best action from the policy
        mdp_action = self.policy.get(state, None)
        
        # Print linear system for this state - now using the separate method
        self.print_linear_system(game_state)
        
        # If no policy available, evaluate actions directly
        if mdp_action is None or mdp_action not in valid_actions:
            print("Policy not available for current state. Evaluating actions directly...")
            mdp_action = self._evaluate_actions(state, valid_actions)
        else:
            print(f"MDP policy chose column {mdp_action+1}")
            
        # PHASE 2: DEFENSIVE CHECK - Validate the MDP's decision
        # This is now a safety check AFTER the MDP has run, not a replacement for it
        defensive_action = self._defensive_search(state) if self.use_search else None
        final_action = defensive_action if defensive_action is not None else mdp_action
        
        # If the defensive action overrides the MDP's choice, log this
        if defensive_action is not None and defensive_action != mdp_action:
            print(f"MDP chose column {mdp_action+1}, but defensive check overrode with column {defensive_action+1}")
        else:
            print(f"Final decision: column {final_action+1}")
        
        end_time = time.time()
        print(f"Decision took {end_time - start_time:.3f} seconds. Explored {self.states_explored} states.")
        
        # Reset cache stats for next move
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
        print(f"Cache performance: {self.cache_hits} hits, {self.cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")
        self.cache_hits = 0
        self.cache_misses = 0
        
        return final_action
    
    def _defensive_search(self, state: GameState) -> Optional[int]:
        """
        Perform a shallow defensive search to find immediate tactical moves.
        This is now ONLY a safety check that runs AFTER the MDP process,
        not a replacement for it.
        
        Args:
            state: The current game state
            
        Returns:
            Optional[int]: Critical action to take, or None if no critical action found
        """
        current_player = state.turn + 1
        opponent = 3 - current_player
        
        # 1. Check if we can win immediately
        winning_moves = state.check_for_immediate_threat(current_player)
        if winning_moves:
            print(f"Found immediate winning move at column {winning_moves[0]+1}")
            return winning_moves[0]
            
        # 2. Check if opponent can win next move and block
        blocking_moves = state.check_for_immediate_threat(opponent)
        if blocking_moves:
            print(f"Blocking opponent's immediate win at column {blocking_moves[0]+1}")
            return blocking_moves[0]
            
        # 3. Check for traps and advanced patterns
        trap_moves = state.check_for_traps(current_player)
        if trap_moves:
            print(f"Setting up trap at column {trap_moves[0]+1}")
            return trap_moves[0]
            
        # 4. Check for opponent traps to block
        opponent_traps = state.check_for_traps(opponent)
        if opponent_traps:
            print(f"Blocking opponent's trap setup at column {opponent_traps[0]+1}")
            return opponent_traps[0]
            
        # 5. Check for advanced patterns
        advanced_moves, pattern_score = state.detect_advanced_patterns(current_player)
        if advanced_moves and pattern_score > 10:  # Only use if pattern score is significant
            print(f"Found advanced pattern, playing column {advanced_moves[0]+1} (score: {pattern_score})")
            return advanced_moves[0]
        
        # No critical defensive action found - use the MDP's decision
        return None
        
    def online_policy_iteration_progressive(self, state: GameState) -> None:
        """
        Perform online policy iteration from the current state with progressive beam widening.
        Uses a wider beam for shallow depths and narrows it as depth increases.
        
        Args:
            state: The current game state
        """
        start_time = time.time()
        self._initialize_state(state)
        
        # Track this state as visited
        self.visits[state] = self.visits.get(state, 0) + 1
        
        print(f"Starting progressive beam search from state: {state.get_key()}")
        
        # Create a set to track all explored states
        all_states = {state}
        
        # Store states by depth for beam search
        states_by_depth = {0: [state]}
        
        # Track total states explored for debugging
        total_states_at_depth = {0: 1}
        
        # Configure progressive beam widths - wider at shallower depths
        progressive_beam_widths = {}
        for d in range(1, self.horizon + 1):
            # Start with full beam width and gradually reduce
            if d <= 4:
                progressive_beam_widths[d] = self.beam_width  # Full width for early depths
            elif d <= 10:
                progressive_beam_widths[d] = int(self.beam_width * 0.75)  # 75% for medium depths
            else:
                progressive_beam_widths[d] = int(self.beam_width * 0.5)  # 50% for deep searches
        
        # Explore up to horizon depth
        for depth in range(1, self.horizon + 1):
            current_beam_width = progressive_beam_widths[depth]
            states_by_depth[depth] = []
            total_states_at_depth[depth] = 0
            
            # Consider all states from previous depth
            parent_count = 0
            for parent_state in states_by_depth[depth-1]:
                parent_count += 1
                # Skip if this is a terminal state
                if parent_state.is_terminal():
                    continue
                
                # Get valid actions for this state
                valid_actions = parent_state.get_valid_actions()
                
                # Try all valid actions
                for action in valid_actions:
                    # Get resulting state
                    next_state = parent_state.apply_action(action)
                    
                    # Initialize state if new
                    if next_state not in all_states:
                        self._initialize_state(next_state)
                        all_states.add(next_state)
                        self.states_explored += 1
                    
                    # Calculate immediate reward for this state
                    reward = self._get_reward(next_state)
                    
                    # For terminal states, just set the value and don't explore further
                    if next_state.is_terminal():
                        # Terminal states get their direct reward value
                        self.values[next_state] = reward
                    else:
                        # Add to next depth states
                        states_by_depth[depth].append(next_state)
                        total_states_at_depth[depth] += 1
                        
                        # Ensure value is initialized (will be updated in value iteration)
                        if next_state not in self.values:
                            self.values[next_state] = self.V0
            
            if parent_count == 0:
                print(f"Warning: No parent states at depth {depth-1}")
                
            # Apply beam search - keep only the best beam_width states
            if len(states_by_depth[depth]) > current_beam_width:
                # Calculate UCB-style values for better exploration
                exploration_values = {}
                for state in states_by_depth[depth]:
                    base_value = self.values.get(state, self.V0)
                    
                    # Add exploration bonus for less-visited states
                    visit_count = self.visits.get(state, 0)
                    if visit_count == 0:
                        exploration_bonus = 2.0  # High bonus for never-visited states
                    else:
                        exploration_bonus = 1.0 / math.sqrt(visit_count)
                    
                    # Check if this state contains immediate threats
                    current_player = state.turn + 1
                    opponent = 3 - current_player
                    
                    # CRITICAL IMMEDIATE THREATS - never prune these
                    if state.check_for_immediate_threat(current_player):
                        exploration_bonus += 10000.0  # Extremely high bonus for immediate wins
                    
                    if state.check_for_immediate_threat(opponent):
                        exploration_bonus += 5000.0  # Very high bonus for blocking opponent wins
                    
                    # Additional patterns - high bonus but not as critical
                    # Strategically important states get a significant bonus
                    
                    # Add bonus for center control
                    num_rows, num_cols = state.board.shape
                    center_col = num_cols // 2
                    center_pieces = sum(1 for row in range(num_rows) if row < num_rows and state.board[row][center_col] == current_player)
                    exploration_bonus += center_pieces * 50.0
                    
                    # Add diagonal pattern detection
                    diagonal_score = state.check_diagonal_connectivity(current_player)
                    if diagonal_score > 0:
                        exploration_bonus += diagonal_score * 20.0
                    
                    # Moves that set up forks (multiple threats)
                    trap_moves = state.check_for_traps(current_player)
                    if trap_moves:
                        exploration_bonus += 100.0
                    
                    # Combined value for sorting
                    exploration_values[state] = base_value + exploration_bonus
                
                # Sort states by exploration-adjusted value
                sorted_states = sorted(
                    states_by_depth[depth],
                    key=lambda x: exploration_values.get(x, float('-inf')),
                    reverse=True
                )
                
                # Print some top and bottom values for debugging
                if len(sorted_states) > 5:
                    top_states = sorted_states[:3]
                    bottom_states = sorted_states[-2:]
                    print(f"  Top states: {[(s.get_key(), exploration_values[s]) for s in top_states]}")
                    print(f"  Bottom states: {[(s.get_key(), exploration_values[s]) for s in bottom_states]}")
                
                # Keep only current_beam_width best states
                states_by_depth[depth] = sorted_states[:current_beam_width]
                
                # Mark these states as visited for future exploration
                for state in states_by_depth[depth]:
                    self.visits[state] = self.visits.get(state, 0) + 1
            
            print(f"Depth {depth}: Exploring {len(states_by_depth[depth])} states (beam width: {current_beam_width}, total: {self.states_explored})")
            
            # If we didn't add any new states at this depth, we can stop exploring
            if len(states_by_depth[depth]) == 0:
                print(f"No new states to explore at depth {depth}, stopping exploration")
                break
        
        # Combine all explored states for value iteration
        states_to_evaluate = set()
        for depth in states_by_depth:
            states_to_evaluate.update(states_by_depth[depth])
        
        # Run value iteration on all explored states
        print(f"Running value iteration on {len(states_to_evaluate)} states")
        self.value_iteration(states_to_evaluate)
        
        # Extract policy for all explored states
        self.policy_extraction(states_to_evaluate)
        
        end_time = time.time()
        print(f"Progressive beam search complete. Explored {self.states_explored} states in {end_time - start_time:.2f} seconds. Policy size: {len(self.policy)}")
    
    def _evaluate_actions(self, state: GameState, valid_actions: List[int]) -> int:
        """
        Evaluate each valid action and choose the best one.
        
        Args:
            state: The current game state
            valid_actions: List of valid actions
            
        Returns:
            int: The best action
        """
        best_action = None
        current_player = state.turn + 1  # Convert from 0/1 to 1/2
        
        # Initialize best value based on player perspective
        if current_player == 2:  # Player 2 maximizes
            best_value = float('-inf')
        else:  # Player 1 minimizes
            best_value = float('inf')
            
        action_values = {}  # For debugging
        
        # Check for immediate winning move
        for action in valid_actions:
            # Simulate the move
            next_state = state.apply_action(action)
            
            # Check if this move results in a win for current player
            # Need to check if previous player (who just played) won
            if next_state.game_board.winning_move(current_player):
                print(f"Found winning move at column {action+1}")
                return action  # Immediate return for winning moves
                
        # Check for opponent's potential win to block
        opponent = 3 - current_player  # Convert from 1/2 to 2/1
        for action in valid_actions:
            # Create a copy of the game board to simulate opponent's move
            temp_board = state.board.copy()
            # Need to create a new GameBoard with the correct dimensions and win condition
            rows, cols = state.board.shape
            win_condition = state.game_board.win_condition
            temp_game_board = GameBoard(rows=rows, cols=cols, win_condition=win_condition)
            temp_game_board.board = temp_board
            
            # Find the next open row in the chosen column
            row = temp_game_board.get_next_open_row(action)
            
            # Place the opponent's piece
            temp_board[row][action] = opponent
            
            # Check if opponent would win with this move
            if temp_game_board.winning_move(opponent):
                print(f"Blocking opponent's win at column {action+1}")
                return action  # Block opponent win
        
        # Check fork creation - look for moves that create multiple threats
        fork_actions = []
        for action in valid_actions:
            next_state = state.apply_action(action)
            forks = self._count_forks(next_state.board, current_player, next_state.game_board.win_condition)
            if forks > 0:
                print(f"Creating fork at column {action+1} with {forks} potential threats")
                fork_actions.append((action, forks))
                
        # If we found fork-creating moves, choose the one with the most forks
        if fork_actions:
            best_fork_action = max(fork_actions, key=lambda x: x[1])[0]
            return best_fork_action
        
        # Check threat creation - look for moves that create win-minus-one-in-a-row
        threat_actions = []
        for action in valid_actions:
            next_state = state.apply_action(action)
            # Get the win condition from the game board
            win_condition = next_state.game_board.win_condition
            # Count threats with win_condition - 1 pieces in a row
            threats = self._count_threats(next_state.board, current_player, win_condition - 1, win_condition)
            if threats > 0:
                print(f"Creating threat at column {action+1} with {threats} potential winning positions")
                threat_actions.append((action, threats))
                
        # If we found threat-creating moves, choose the one with the most threats
        if threat_actions:
            best_threat_action = max(threat_actions, key=lambda x: x[1])[0]
            return best_threat_action
        
        # If we didn't find a winning move, evaluate based on state values
        for action in valid_actions:
            next_state = state.apply_action(action)
            
            # Get reward for this action
            reward = self._get_reward(next_state)
            
            # Calculate value using reward and estimated future value
            if next_state.is_terminal():
                value = reward  # For terminal states, just use reward
            else:
                # For non-terminal states, use reward plus discounted future value
                future_value = self.values.get(next_state, self.V0)
                value = reward + self.gamma * future_value
            
            action_values[action] = value
            
            # Update best action based on player perspective
            if current_player == 2:  # Player 2 maximizes
                if value > best_value:
                    best_value = value
                    best_action = action
            else:  # Player 1 minimizes
                if value < best_value:
                    best_value = value
                    best_action = action
        
        # Log the action evaluations
        print(f"Action values: {', '.join([f'{a+1}: {v:.2f}' for a, v in sorted(action_values.items())])}")
        
        # If still no best action, prefer center columns
        if best_action is None:
            # Get the center column based on number of columns
            num_cols = state.board.shape[1]
            center_col = num_cols // 2
            
            # Center column preference - prefer center, then adjacent columns
            center_preference = [center_col]
            # Add columns radiating outward from center
            for offset in range(1, num_cols):
                if center_col - offset >= 0:
                    center_preference.append(center_col - offset)
                if center_col + offset < num_cols:
                    center_preference.append(center_col + offset)
                    
            # Choose the first valid action from our preference list
            for col in center_preference:
                if col in valid_actions:
                    best_action = col
                    break
        
        # If still no best action, choose randomly
        if best_action is None:
            best_action = random.choice(valid_actions)
            print(f"Choosing random action: {best_action+1}")
        else:
            perspective = "maximize" if current_player == 2 else "minimize"
            print(f"Choosing best action: column {best_action+1} with value {action_values.get(best_action, 'N/A'):.2f} ({perspective})")
        
        return best_action
    
    def update(self, game_state: Dict, reward: float) -> None:
        """Update the value function for the current state."""
        # Convert external reward scale to internal reward scale
        if reward > 0:  # Win
            reward = 200.0
        elif reward < 0:  # Loss
            reward = -200.0
            
        state = self._convert_to_game_state(game_state)
        self.values[state] = reward
        print(f"Updating final state value to {reward}")
    
    def reset(self) -> None:
        """Reset the agent's state for a new game."""
        # Keep values and policy but reset statistics
        self.states_explored = 0
        self.iterations_performed = 0
        self.eval_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def value_iteration(self, states: Set[GameState]) -> None:
        """
        Evaluate the current policy by computing V(s) for all states in the set.
        
        Args:
            states: Set of states to evaluate
        """
        # Reset sweep counter for this run
        self.vi_sweeps = 0
        self.iterations_performed += 1
        iteration = 0
        max_iterations = 100  # Allow more iterations for better convergence
        
        # Initialize debug information
        last_deltas = []
        
        while True:
            iteration += 1
            # Count each full sweep through all states
            self.vi_sweeps += 1
            delta = 0
            
            # Copy values for synchronous updates
            old_values = self.values.copy()
            
            # Update each state's value
            for state in states:
                # Skip terminal states (they already have fixed values)
                if state.is_terminal():
                    continue
                
                # Get valid actions
                valid_actions = state.get_valid_actions()
                if not valid_actions:
                    continue
                
                # Initialize optimal value based on player perspective
                current_player = state.turn + 1  # Convert from 0/1 to 1/2
                
                if current_player == 2:  # Player 2 maximizes
                    optimal_value = float('-inf')
                else:  # Player 1 minimizes
                    optimal_value = float('inf')
                
                # Try each action and find the best one
                for action in valid_actions:
                    next_state = state.apply_action(action)
                    
                    # Get reward and next state value
                    reward = self._get_reward(next_state)
                    
                    # Use fixed reward for terminal states, otherwise use value function
                    if next_state.is_terminal():
                        next_value = reward
                    else:
                        next_value = old_values.get(next_state, self.V0)
                    
                    # Compute Q-value
                    value = reward + self.gamma * next_value
                    
                    # Update optimal value based on player perspective
                    if current_player == 2:  # Player 2 maximizes
                        if value > optimal_value:
                            optimal_value = value
                    else:  # Player 1 minimizes
                        if value < optimal_value:
                            optimal_value = value
                
                # Update state value if we found a better value
                if (current_player == 2 and optimal_value != float('-inf')) or \
                   (current_player == 1 and optimal_value != float('inf')):
                    old_value = old_values.get(state, self.V0)
                    self.values[state] = optimal_value
                    value_change = abs(old_value - optimal_value)
                    delta = max(delta, value_change)
            
            # Save delta for convergence tracking
            last_deltas.append(delta)
            if len(last_deltas) > 5:
                last_deltas.pop(0)
            
            # Check for convergence - only if we've done enough iterations
            if iteration > 10 and delta < self.epsilon:
                break
                
            # Limit iterations
            if iteration >= max_iterations:
                print(f"Value iteration stopped after {iteration} iterations (delta={delta:.6f})")
                break
            
            # Print progress periodically
            if iteration % 10 == 0:
                self._vprint(f"Value iteration: {iteration} iterations, delta={delta:.6f}")
        
        # Save final delta for stats
        self.last_vi_delta = delta
        # Print some debugging info about convergence
        if len(last_deltas) > 1:
            avg_delta = sum(last_deltas) / len(last_deltas)
            self._vprint(f"Value iteration converged after {iteration} iterations. Final delta={delta:.6f}, avg={avg_delta:.6f}")
    
    def policy_extraction(self, states: Set[GameState]) -> None:
        """
        Extract the optimal policy from the current value function.
        
        Args:
            states: Set of states to extract policy for
        """
        # Reset counter for this run
        self.policy_updates_last = 0
        policy_updates = 0
        
        # Update policy for all states
        for state in states:
            # Skip terminal states
            if state.is_terminal():
                continue
            
            # Get valid actions
            valid_actions = state.get_valid_actions()
            if not valid_actions:
                continue
            
            # Find the best action
            best_action = None
            current_player = state.turn + 1  # Convert from 0/1 to 1/2
            
            # Initialize best value differently based on player
            if current_player == 2:  # Player 2 maximizes
                best_value = float('-inf')
            else:  # Player 1 minimizes
                best_value = float('inf')
                
            action_values = {}  # For debugging
            
            for action in valid_actions:
                next_state = state.apply_action(action)
                
                # Get reward for the next state
                reward = self._get_reward(next_state)
                
                # Calculate value differently for terminal vs. non-terminal states
                if next_state.is_terminal():
                    value = reward  # Just use reward for terminal states
                else:
                    # For non-terminal states, use reward + discounted future value
                    value = reward + self.gamma * self.values.get(next_state, self.V0)
                
                # Store this action's value for debugging
                action_values[action] = value
                
                # Update best action if this is better, based on player perspective
                if current_player == 2:  # Player 2 maximizes
                    if value > best_value:
                        best_value = value
                        best_action = action
                else:  # Player 1 minimizes
                    if value < best_value:
                        best_value = value
                        best_action = action
            
            # Update policy for this state
            old_action = self.policy.get(state)
            if best_action is not None and best_action != old_action:
                self.policy[state] = best_action
                policy_updates += 1
                self.policy_updates_last += 1
                
                # Debug output for significant policy changes
                if old_action is not None:
                    print(f"Policy updated for state: turn={state.turn+1}, " 
                          f"old={old_action+1} (value={action_values.get(old_action, 'N/A')}), "
                          f"new={best_action+1} (value={action_values.get(best_action, 'N/A')})")
        
        self._vprint(f"Policy extraction complete. Updated {policy_updates} states out of {len(states)}.")
    
    def _get_reward(self, state: GameState) -> float:
        """
        Calculate the reward for a game state.
        Enhanced with better strategic evaluation for Connect Four patterns.
        
        Args:
            state: The current game state
            
        Returns:
            float: Reward value (positive for win, negative for loss)
        """
        # Check cache first
        state_hash = hash(state)
        if state_hash in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[state_hash]
            
        self.cache_misses += 1
        
        board = state.board
        num_rows, num_cols = board.shape
        current_player = state.turn + 1  # Player 1 or 2
        # Note: current_player here is who will move next,
        # but for terminal checks we look at absolute winners (1 or 2).
        
        # Get win condition from the game board
        win_condition = state.game_board.win_condition

        # ------------------------------------------------------------------
        # Terminal‑state checks – symmetric, zero‑sum
        #   • Player 2 (the maximizer) wins  →  +200
        #   • Player 1 (the minimizer) wins  →  −200
        #   • Draw                            →   0
        # ------------------------------------------------------------------
        if state.game_board.winning_move(2):
            reward = 200.0
            self.eval_cache[state_hash] = reward
            return reward

        if state.game_board.winning_move(1):
            reward = -200.0
            self.eval_cache[state_hash] = reward
            return reward

        if state.game_board.tie_move():
            reward = 0.0
            self.eval_cache[state_hash] = reward
            return reward

        # If heuristics are disabled, return a small step cost to encourage
        # faster wins but keep the scale modest.
        if not self.use_heuristics:
            reward = -0.01
            self.eval_cache[state_hash] = reward
            return reward

        # Calculate positional reward based on pieces and threats
        reward = 0.0
        
        # Check for potential winning positions for the current player
        three_in_a_row = self._count_threats(board, current_player, win_condition-1, win_condition)
        two_in_a_row = self._count_threats(board, current_player, win_condition-2, win_condition)
        
        # Check for opponent threats
        last_player = 3 - current_player
        opponent_three = self._count_threats(board, last_player, win_condition-1, win_condition)
        opponent_two = self._count_threats(board, last_player, win_condition-2, win_condition)
        
        # Count forks (multiple threats)
        fork_positions = self._count_forks(board, current_player, win_condition)
        opponent_forks = self._count_forks(board, last_player, win_condition)
        
        # Get diagonal connectivity score - not using this for smaller boards
        diagonal_score = 0
        if win_condition >= 4:
            diagonal_score = state.check_diagonal_connectivity(current_player)
        
        # REWARD STRUCTURE - BALANCED FOR BOTH OFFENSE AND DEFENSE
        
        # Immediate threats - highest rewards/penalties
        # Winning threats are extremely valuable 
        reward += three_in_a_row * 30.0
        
        # Building threats is good
        reward += two_in_a_row * 4.0
        
        # Forks are extremely valuable
        reward += fork_positions * 50.0
        
        # Add diagonal score 
        reward += diagonal_score * 5.0
        
        # DEFENSIVE REWARDS - must be strong enough to actually block opponent threats
        # Opponent threats need to be countered - negative value
        reward -= opponent_three * 50.0  # Even higher penalty - must be higher than our reward
        reward -= opponent_two * 4.0  
        reward -= opponent_forks * 75.0  # Critical to block opponent forks
        
        # Prefer center control - use appropriate center column based on board size
        center_col = num_cols // 2  # Middle column
        center_control = sum(1 for row in range(num_rows) if board[row][center_col] == current_player)
        reward += center_control * 5.0
        
        # Opponent center control is dangerous
        opponent_center = sum(1 for row in range(num_rows) if board[row][center_col] == last_player)
        reward -= opponent_center * 4.0
        
        # Adjacent columns are next most valuable if available
        adjacent_columns = []
        if center_col > 0:
            adjacent_columns.append(center_col - 1)
        if center_col < num_cols - 1:
            adjacent_columns.append(center_col + 1)
            
        if adjacent_columns:
            adjacent_control = sum(1 for row in range(num_rows) for col in adjacent_columns if col < num_cols and board[row][col] == current_player)
            reward += adjacent_control * 2.0
        
        # Add a small penalty to encourage faster wins
        reward -= 0.01
        
        # Cache the reward
        self.eval_cache[state_hash] = reward
        return reward
    
    def _count_connected_pieces(self, board, player):
        """Count the number of our pieces that are adjacent to other pieces of the same player."""
        connected = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]  # horizontal, vertical, diagonal
        num_rows, num_cols = board.shape
        
        for row in range(num_rows):
            for col in range(num_cols):
                if board[row][col] == player:
                    # Check all directions
                    for dr, dc in directions:
                        r2, c2 = row + dr, col + dc
                        if 0 <= r2 < num_rows and 0 <= c2 < num_cols and board[r2][c2] == player:
                            connected += 1
        
        return connected
        
    def _count_threats(self, board, player, count, win_condition=4):
        """
        Count the number of potential threats with 'count' pieces in a row
        and at least one empty space to complete it.
        
        Args:
            board: The game board
            player: The player to check threats for
            count: How many pieces in a row to look for
            win_condition: Number of pieces in a row needed to win
            
        Returns:
            int: Number of threats found
        """
        threats = 0
        num_rows, num_cols = board.shape
        
        # Horizontal threats
        for row in range(num_rows):
            for col in range(num_cols - (win_condition - 1)):
                window = [board[row][col+i] for i in range(win_condition)]
                if window.count(player) == count and window.count(0) == win_condition - count:
                    threats += 1
        
        # Vertical threats
        for row in range(num_rows - (win_condition - 1)):
            for col in range(num_cols):
                window = [board[row+i][col] for i in range(win_condition)]
                if window.count(player) == count and window.count(0) == win_condition - count:
                    threats += 1
        
        # Positive diagonal threats
        for row in range(num_rows - (win_condition - 1)):
            for col in range(num_cols - (win_condition - 1)):
                window = [board[row+i][col+i] for i in range(win_condition)]
                if window.count(player) == count and window.count(0) == win_condition - count:
                    threats += 1
        
        # Negative diagonal threats
        for row in range(win_condition - 1, num_rows):
            for col in range(num_cols - (win_condition - 1)):
                window = [board[row-i][col+i] for i in range(win_condition)]
                if window.count(player) == count and window.count(0) == win_condition - count:
                    threats += 1
                    
        return threats
        
    def _count_forks(self, board, player, win_condition=4):
        """
        Count fork positions - positions where multiple winning threats exist.
        
        Args:
            board: The game board
            player: The player to check for
            win_condition: Number of pieces in a row needed to win
            
        Returns:
            int: Number of fork positions
        """
        forks = 0
        num_rows, num_cols = board.shape
        
        # For each empty position, check if placing a piece creates multiple threats
        for col in range(num_cols):
            for row in range(num_rows):
                # Skip non-empty positions
                if board[row][col] != 0:
                    continue
                    
                # Skip positions that aren't accessible yet
                if row > 0 and board[row-1][col] == 0:
                    continue
                    
                # Make a temporary move
                board[row][col] = player
                
                # Count threats at this position
                threats = self._count_threats(board, player, win_condition-1, win_condition)
                
                # A fork has at least 2 threats
                if threats >= 2:
                    forks += 1
                    
                # Undo the move
                board[row][col] = 0
                
        return forks
        
    def _convert_to_game_state(self, game_state: Dict) -> GameState:
        """
        Convert a dictionary game state to a GameState object.
        
        Args:
            game_state: The dictionary game state from the game
            
        Returns:
            GameState: The converted GameState object
        """
        board = game_state['board']
        turn = game_state['turn']
        game_board = game_state.get('game_board')
        
        return GameState(board, turn, game_board)

    def compute_bellman_equation(self, state: GameState) -> Dict:
        """
        Compute the complete Bellman equations for a state, including full action values.
        This shows exactly how the value of each action is calculated.
        
        Args:
            state: The current game state
            
        Returns:
            Dict: Dictionary with action values and their components
        """
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return {}
            
        result = {}
        current_player = state.turn + 1  # 1 or 2
        
        # For each action, compute value components
        for action in valid_actions:
            next_state = state.apply_action(action)
            
            # Get immediate reward
            immediate_reward = self._get_reward(next_state)
            
            # Get future value
            if next_state.is_terminal():
                future_value = 0.0  # Terminal states have no future
            else:
                future_value = self.values.get(next_state, self.V0)
                
            # Calculate total value
            total_value = immediate_reward + self.gamma * future_value
            
            # Store all components
            result[action] = {
                'immediate_reward': immediate_reward,
                'future_value': future_value,
                'discount_factor': self.gamma,
                'total_value': total_value,
                'perspective': 'MAXIMIZE' if current_player == 2 else 'MINIMIZE'
            }
            
        return result
        
    def analyze_linear_system(self, state: GameState) -> None:
        """Analyze the linear system for a state."""
        # This method can be implemented later for linear system analysis
        pass
        
    def get_linear_system(self, state: GameState) -> np.ndarray:
        """Get the linear system for a state."""
        valid_actions = state.get_valid_actions()
        num_actions = len(valid_actions)
        
        # Handle case where there are no valid actions
        if num_actions == 0:
            # Return a 1x1 matrix with a 0
            return np.zeros((1, 1))
        
        # Ensure we have at least num_actions+1 columns (one for each action plus reward)
        min_columns = max(num_actions, 1) + 1
        
        # map all known states to a unique index
        state_values = list(self.values.keys())
        state_ind = {s: idx for idx, s in enumerate(state_values)}
        
        # Make sure the coefficient matrix has enough columns
        # Either the number of states in values + 1, or min_columns, whichever is larger
        coeff_columns = max(len(self.values) + 1, min_columns)
        coeff = np.zeros((num_actions, coeff_columns))
        
        for i, action in enumerate(valid_actions):
            next_state = state.apply_action(action)
            reward = self._get_reward(next_state)
            
            # Set diagonal element to 1.0
            coeff[i, i] = 1.0
            
            if next_state.is_terminal():
                coeff[i, -1] = reward
            else:
                # If next_state is in our value function mapping, include it in equation
                if next_state in state_ind:
                    coeff[i, state_ind[next_state]] = -self.gamma
                
                coeff[i, -1] = reward
                
        return coeff

    def enumerate_reachable_states(self, start_state, horizon=3):
        """Enumerate all states reachable from start_state within horizon moves."""
        all_states = set([start_state])
        frontier = [start_state]
        
        for depth in range(horizon):
            new_frontier = []
            for state in frontier:
                if state.is_terminal():
                    continue
                    
                for action in state.get_valid_actions():
                    next_state = state.apply_action(action)
                    if next_state not in all_states:
                        all_states.add(next_state)
                        new_frontier.append(next_state)
            
            frontier = new_frontier
            if not frontier:  # No more states to explore
                break
            
        return all_states

    # ------------------------------------------------------------------
    # Build / refresh a canonical ordering of states for DP helpers
    # ------------------------------------------------------------------
    def _set_global_state_index(self, states: Set[GameState]) -> None:
        """
        Record a stable mapping from each state to a column index.
        All DP helpers should reference `self.state_index` instead of
        building their own local dictionaries.
        """
        self.all_states = set(states)
        self.state_index = {s: i for i, s in enumerate(states)}

    # ------------------------------------------------------------------
    # Pure dynamic‑programming planner (no beam search, no defensive extras)
    # ------------------------------------------------------------------
    def _dp_plan_simple(self, root: GameState) -> None:
        """Populate self.values and self.policy using plain DP only."""
        # Enumerate all states reachable within the given horizon
        states = self.enumerate_reachable_states(root, self.horizon)

        # Record a global ordering for later helpers
        self._set_global_state_index(states)

        # Initialize value table and seed terminal‑state rewards
        for s in states:
            self._initialize_state(s)
            if s.is_terminal():
                self.values[s] = self._get_reward(s)

        # Classic value‑iteration followed by greedy policy extraction
        self.value_iteration(states)
        self.policy_extraction(states)
        # Show instrumentation summary
        self.print_stats("DP‑only summary")
    # ------------------------------------------------------------------
    # Pretty‑print instrumentation after a DP run
    # ------------------------------------------------------------------
    def print_stats(self, label: str = "DP run stats") -> None:
        """Print key instrumentation counters in a single line."""
        total_states = len(self.all_states)
        print(f"{label}: "
              f"|S|={total_states}, "
              f"VI sweeps={self.vi_sweeps}, "
              f"final Δ={self.last_vi_delta:.6f}, "
              f"policy updates={self.policy_updates_last}")

    def visualize_policy_matrices(self, policy, states):
        """Visualize transition and reward matrices for a given policy."""
        n = len(states)
        index = {s:i for i,s in enumerate(states)}
        P = np.zeros((n,n))
        R = np.zeros(n)
        
        # Build matrices
        for s in states:
            i = index[s]
            if s in policy and policy[s] is not None:
                a = policy[s]
                next_state = s.apply_action(a)
                R[i] = self._get_reward(next_state)
                if not next_state.is_terminal():
                    if next_state in index:  # Only include states in our set
                        j = index[next_state]
                        P[i,j] = 1.0
        
        # Print matrices in a readable format
        print(f"\nTransition matrix P (size: {P.shape}):")
        print(P)
        print(f"\nReward vector R (size: {R.shape}):")
        print(R)
        
        # Calculate and display V = (I - γP)^-1 R
        try:
            I = np.eye(n)
            V = np.linalg.solve(I - self.gamma*P, R)
            print("\nValue vector V:")
            print(V)
        except np.linalg.LinAlgError as e:
            print(f"Error solving linear system: {e}")

    def policy_iteration_linear(self, start_state, horizon=3):
        """
        Perform policy iteration using direct linear algebra.
        
        Args:
            start_state: Starting state
            horizon: Maximum depth to explore
        
        Returns:
            Tuple of (policy, values)
        """
        # Step 1: Enumerate all reachable states
        states = self.enumerate_reachable_states(start_state, horizon)
        print(f"Enumerated {len(states)} states within horizon {horizon}")
        
        # Step 2: Initialize policy randomly
        policy = {}
        for s in states:
            if not s.is_terminal():
                valid_actions = s.get_valid_actions()
                if valid_actions:
                    policy[s] = random.choice(valid_actions)
        
        # Step 3: Policy iteration
        stable = False
        iteration = 0
        while not stable and iteration < 20:  # Limit iterations
            iteration += 1
            
            # Policy evaluation using linear algebra
            values = self.policy_evaluate_linear(policy, states)
            
            # Policy improvement
            stable = True
            for s in states:
                if s.is_terminal() or s not in policy:
                    continue
                    
                old_action = policy[s]
                
                # Find best action
                best_action = None
                current_player = s.turn + 1  # Convert from 0/1 to 1/2
                
                if current_player == 2:  # Maximize
                    best_value = float('-inf')
                else:  # Minimize
                    best_value = float('inf')
                    
                for a in s.get_valid_actions():
                    next_s = s.apply_action(a)
                    reward = self._get_reward(next_s)
                    
                    if next_s.is_terminal():
                        value = reward
                    else:
                        value = reward + self.gamma * values.get(next_s, 0.0)
                    
                    if (current_player == 2 and value > best_value) or \
                       (current_player == 1 and value < best_value):
                        best_value = value
                        best_action = a
                
                if best_action != old_action:
                    policy[s] = best_action
                    stable = False
            
            print(f"Iteration {iteration}: {'Stable' if stable else 'Changed'}")
        
        # Visualize final matrices
        self.visualize_policy_matrices(policy, states)
        
        return policy, values

    def policy_evaluate_linear(self, policy, states):
        """Evaluate a policy using direct linear algebra (solving V = (I-γP)^(-1)R)."""
        # Prefer the global mapping if we're evaluating that exact set
        if set(states) == self.all_states:
            index = self.state_index
        else:
            index = {s: i for i, s in enumerate(states)}
        n = len(states)
        P = np.zeros((n, n))
        R = np.zeros(n)

        for s in states:
            i = index[s]
            if s in policy and policy[s] is not None:
                a = policy[s]
                sprime = s.apply_action(a)
                R[i] = self._get_reward(sprime)
                if not sprime.is_terminal() and sprime in index:
                    j = index[sprime]
                    P[i, j] = 1.0   # deterministic

        # Solve V = (I - γP)^(-1)R directly
        V = np.linalg.solve(np.eye(n) - self.gamma * P, R)
        return {s: V[index[s]] for s in states}

    # ------------------------------------------------------------------
    # Utility: deterministic transition matrix Pπ and reward vector Rπ
    # ------------------------------------------------------------------
    def build_PR_matrices(self, policy: Dict['GameState', int], states: Set['GameState']):
        """
        Return (P, R) for a deterministic policy π restricted to `states`.

        • P is |S|×|S| with 1.0 in column j if T(s,π(s)) = sʹ_j  
        • R is length‑|S|, the immediate reward of taking π(s) in s.
        """
        # Re‑use the global mapping when applicable
        if set(states) == self.all_states:
            index = self.state_index
        else:
            index = {s: i for i, s in enumerate(states)}

        n = len(states)
        P = np.zeros((n, n))
        R = np.zeros(n)

        for s in states:
            i = index[s]
            if s in policy and policy[s] is not None:
                a = policy[s]
                sprime = s.apply_action(a)
                R[i] = self._get_reward(sprime)
                if sprime in index:
                    P[i, index[sprime]] = 1.0
        return P, R

    def run_toy_problem(self, rows=3, cols=4, horizon=3):
        """Run a small toy problem using linear algebra approach."""
        # --- Temporarily turn off positional heuristics for this clean experiment ---
        original_heuristic_flag = self.use_heuristics
        self.use_heuristics = False
        # Create a small initial board
        board = np.zeros((rows, cols))
        game_board = GameBoard(rows=rows, cols=cols)
        start_state = GameState(board, 0, game_board)
        
        print(f"\n=== RUNNING TOY PROBLEM: {rows}x{cols} board with horizon {horizon} ===")
        print("Initial board:")
        print(board)
        
        # Completely disable beam search, caching, and other optimizations
        original_beam = self.beam_width
        original_horizon = self.horizon
        self.beam_width = float('inf')  # No beam search limitation
        self.horizon = horizon
        
        # Clear existing values and policy
        self.values = {}
        self.policy = {}
        
        # Run our linear algebra policy iteration
        policy, values = self.policy_iteration_linear(start_state, horizon)
        
        # Print the policy for the starting state
        if start_state in policy:
            best_action = policy[start_state]
            print(f"\nBest action for starting state: {best_action+1}")
            print(f"Value: {values.get(start_state, 'Unknown')}")
        else:
            print("\nNo policy found for starting state")

        # Register the full state set for later helpers
        self._set_global_state_index(set(values.keys()))
        
        # ---------------------------------------------------------------------------
        # Restore original heuristic setting, beam_width, and horizon
        self.beam_width = original_beam
        self.horizon = original_horizon
        self.use_heuristics = original_heuristic_flag
        
        return policy, values

    def compare_with_minimax(self, state, depth=3):
        """Compare our linear algebra solution with minimax."""
        print("\n=== COMPARING WITH MINIMAX ===")
        
        # Run minimax
        minimax_value, minimax_action = self._minimax(state, depth, True)
        
        # Run our linear policy iteration
        policy, values = self.policy_iteration_linear(state, depth)
        linear_value = values.get(state, 0.0)
        linear_action = policy.get(state, None)
        
        print(f"Minimax: action={minimax_action+1}, value={minimax_value}")
        print(f"Linear: action={linear_action+1 if linear_action is not None else None}, value={linear_value}")
        
        return minimax_action == linear_action
        
    def _minimax(self, state, depth, maximizing):
        """Simple minimax implementation for comparison."""
        if depth == 0 or state.is_terminal():
            return self._get_reward(state), None
        
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return 0, None
            
        best_action = None
        if maximizing:
            value = float('-inf')
            for action in valid_actions:
                next_state = state.apply_action(action)
                child_value, _ = self._minimax(next_state, depth-1, False)
                if child_value > value:
                    value = child_value
                    best_action = action
        else:
            value = float('inf')
            for action in valid_actions:
                next_state = state.apply_action(action)
                child_value, _ = self._minimax(next_state, depth-1, True)
                if child_value < value:
                    value = child_value
                    best_action = action
                    
        return value, best_action
    def get_bellman_candidates(self, state: GameState) -> Dict[int, Dict[str, float]]:
        """
        For each valid action a in state s, return a dictionary with the pieces
        needed for the Bellman optimality backup

            Q(s,a) = R(s,a) + gamma * V(s')

        where s' is the successor reached by taking action a.

        The returned mapping is:
            action_index -> {
                'reward':          R(s,a),
                'future_value':    V(s'),
                'q_value':         R(s,a) + gamma * V(s'),
                'is_terminal':     bool
            }
        """
        candidates: Dict[int, Dict[str, float]] = {}
        valid_actions = state.get_valid_actions()
        if not valid_actions:           # no legal moves
            return candidates

        for action in valid_actions:
            next_state = state.apply_action(action)

            # Ensure the global index contains this successor
            if next_state not in self.state_index:
                self.state_index[next_state] = len(self.state_index)
                self.all_states.add(next_state)

            # immediate reward
            reward = self._get_reward(next_state)

            # look‑ahead value
            if next_state.is_terminal():
                future_v = 0.0
            else:
                future_v = self.values.get(next_state, self.V0)

            q_val = reward + self.gamma * future_v

            candidates[action] = {
                'reward': reward,
                'future_value': future_v,
                'q_value': q_val,
                'is_terminal': next_state.is_terminal()
            }

        return candidates