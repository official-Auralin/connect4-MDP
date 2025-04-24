from typing import Any, Dict, List, Tuple, Set, Optional
import numpy as np
import copy
import random
import time
import math
from game_board import GameBoard

# TODO: figure out why the game is not printing a linear system for Player 1
# TODO: modify game board to have a size setting and a win condition setting e.g., 4x3 and 3 in a row
# TODO: add an initial state setting, so we can test the agent in terminal and near terminal states with fewer available moves
# TODO: figure out if the recursive nature of the bellman equation is supposed to reduce to a smaller system for each turn. (what we have seems correct)
# TODO: fill compute_bellman_equation with the correct equations, currently just returns a placeholder - this will let us see the linear systems for the 7 available moves. 
# TODO: imshow in matplotlib can be used to visualize the board takes in a numpy array and displays it as a grid, will pull up a secondary GUI. 

class GameState:
    """
    A wrapper class for game states that supports hashing and comparison.
    This enables using GameState objects as dictionary keys for the MDP value function.
    """
    
    def __init__(self, board: np.ndarray, turn: int, game_board: GameBoard = None):
        """
        Initialize a game state.
        
        Args:
            board: The game board as a numpy array
            turn: The player's turn (0 or 1)
            game_board: Reference to GameBoard object (if available)
        """
        self.board = board.copy()  # Make a copy to ensure independence
        self.turn = turn
        
        # Create a new GameBoard if none provided
        if game_board is None:
            # Get board dimensions from the array
            rows, cols = board.shape
            self.game_board = GameBoard(rows=rows, cols=cols)
            self.game_board.board = board.copy()
        else:
            self.game_board = game_board
            
    def __hash__(self):
        """
        Generate a hash for the game state based on board configuration and turn.
        This allows GameState objects to be used as dictionary keys.
        """
        # Convert board to tuple for hashing
        board_tuple = tuple(map(tuple, self.board))
        return hash((board_tuple, self.turn))
        
    def __eq__(self, other):
        """Check if two game states are equal."""
        if not isinstance(other, GameState):
            return False
        return (np.array_equal(self.board, other.board) and 
                self.turn == other.turn)
                
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (win or draw)."""
        # Check if previous player won
        last_player = 3 - (self.turn + 1)  # Convert from 0/1 to 1/2
        if self.game_board.winning_move(last_player):
            return True
            
        # Check for a draw
        if self.game_board.tie_move():
            return True
            
        return False
        
    def get_valid_actions(self) -> List[int]:
        """Get valid actions (columns) for this state."""
        # Use game_board's columns count instead of hardcoded 7
        return [col for col in range(self.game_board.cols) if self.game_board.is_valid_location(col)]
    
    def apply_action(self, action: int) -> 'GameState':
        """
        Apply an action to this state and return the resulting state.
        
        Args:
            action: Column to drop piece in (0-6)
            
        Returns:
            GameState: The new state after action
        """
        # Create a new game board for the next state
        new_board = self.board.copy()
        
        # Create a new game board object with the same dimensions and win condition
        rows, cols = self.board.shape
        win_condition = getattr(self.game_board, 'win_condition', 4)  # Default to 4 if not available
        new_game_board = GameBoard(rows=rows, cols=cols, win_condition=win_condition)
        new_game_board.board = new_board
        
        # Find the next open row in the chosen column
        row = new_game_board.get_next_open_row(action)
        
        # Place the piece
        new_board[row][action] = self.turn + 1  # Convert from 0/1 to 1/2
        
        # Create and return the new state with updated turn
        return GameState(new_board, (self.turn + 1) % 2, new_game_board)
        
    def get_key(self) -> str:
        """
        Get a string key representation for this state.
        Used for debugging and display purposes only.
        """
        # Convert the board to a string representation
        cols = []
        num_rows, num_cols = self.board.shape
        for col in range(num_cols):
            column = ''.join(str(int(self.board[row][col])) for row in range(num_rows))
            cols.append(column)
        
        # Join columns with '|' separator and combine with turn
        return f"{self.turn}:{':'.join(cols)}"
        
    def check_for_immediate_threat(self, player: int) -> List[int]:
        """
        Check if there are any immediate threats (opponent can win next move).
        
        Args:
            player: The player to check threats for
            
        Returns:
            List[int]: List of columns where the player can win immediately
        """
        winning_moves = []
        board = self.board
        num_rows, num_cols = board.shape
        win_condition = self.game_board.win_condition
        
        # Check each column
        for col in range(num_cols):
            # Skip if column is full
            if not self.game_board.is_valid_location(col):
                continue
                
            # Create a temporary board with correct dimensions and win condition
            temp_board = board.copy()
            temp_game_board = GameBoard(rows=num_rows, cols=num_cols, win_condition=win_condition)
            temp_game_board.board = temp_board
            
            # Find the next open row in this column
            row = temp_game_board.get_next_open_row(col)
            
            # Place the piece
            temp_board[row][col] = player
            
            # Check if this creates a win
            if temp_game_board.winning_move(player):
                winning_moves.append(col)
                
        return winning_moves
        
    def check_for_traps(self, player: int) -> List[int]:
        """
        Check for common Connect Four trap setups that lead to forced wins.
        IMPROVED to be more selective and accurate in trap detection.
        
        Args:
            player: The player to check traps for
            
        Returns:
            List[int]: List of columns to play to set up or block traps
        """
        trap_moves = []
        opponent = 3 - player
        board = self.board
        num_rows, num_cols = board.shape
        win_condition = self.game_board.win_condition  # Get win condition from game board
        
        # Special handling for early game center control
        empty_count = np.count_nonzero(board == 0)
        total_slots = num_rows * num_cols
        is_early_game = empty_count > total_slots * 0.8  # First few moves (80% empty)
        
        # In early game, prioritize center and adjacent columns
        if is_early_game:
            # Center column is highly valuable
            center_col = num_cols // 2
            if self.game_board.is_valid_location(center_col):
                if center_col not in trap_moves:
                    trap_moves.append(center_col)
            
            # If opponent has center, control adjacent columns
            if center_col < num_cols and board[0][center_col] == opponent:
                for col in [center_col-1, center_col+1]:
                    if 0 <= col < num_cols and self.game_board.is_valid_location(col) and col not in trap_moves:
                        trap_moves.append(col)
        
        # Find moves that create TWO threats simultaneously (true forks)
        for col in range(num_cols):
            if not self.game_board.is_valid_location(col):
                continue
                
            # Simulate placing a piece in this column
            row = self.game_board.get_next_open_row(col)
            temp_board = board.copy()
            temp_game_board = GameBoard(rows=num_rows, cols=num_cols, win_condition=win_condition)
            temp_game_board.board = temp_board
            temp_board[row][col] = player
            
            # Count potential winning lines after this move
            threats = 0
            
            # Check horizontal threats
            for c in range(max(0, col-(win_condition-1)), min(col+1, num_cols-(win_condition-1))):
                if c + win_condition <= num_cols:
                    window = [temp_board[row][c+i] for i in range(win_condition)]
                    if window.count(player) == win_condition - 1 and window.count(0) == 1:
                        threats += 1
                    
            # Check vertical threats
            if row >= win_condition - 1:
                window = [temp_board[row-i][col] for i in range(win_condition)]
                if window.count(player) == win_condition - 1 and window.count(0) == 1:
                    threats += 1
                    
            # Check diagonal threats
            for i in range(win_condition):
                # Positive diagonal
                r = row - i
                c = col - i
                if r >= 0 and r <= num_rows - win_condition and c >= 0 and c <= num_cols - win_condition:
                    window = [temp_board[r+j][c+j] for j in range(win_condition)]
                    if window.count(player) == win_condition - 1 and window.count(0) == 1:
                        threats += 1
                
                # Negative diagonal
                r = row - i
                c = col + i
                if r >= 0 and r <= num_rows - win_condition and c >= win_condition - 1 and c < num_cols:
                    if all(0 <= r+j < num_rows and 0 <= c-j < num_cols for j in range(win_condition)):
                        window = [temp_board[r+j][c-j] for j in range(win_condition)]
                        if window.count(player) == win_condition - 1 and window.count(0) == 1:
                            threats += 1
            
            # Only consider as trap if it creates MULTIPLE threats
            if threats >= 2 and col not in trap_moves:
                trap_moves.append(col)
                
        return trap_moves
        
    def check_diagonal_connectivity(self, player: int) -> int:
        """
        Specifically check for diagonal connections and potential winning patterns.
        
        Args:
            player: The player to check for
            
        Returns:
            int: Score representing strength of diagonal connections
        """
        board = self.board
        num_rows, num_cols = board.shape
        score = 0
        opponent = 3 - player
        win_condition = self.game_board.win_condition
        
        # Check all possible diagonal directions
        # Positive diagonals (/)
        for row in range(num_rows - (win_condition - 1)):
            for col in range(num_cols - (win_condition - 1)):
                window = [board[row+i][col+i] for i in range(win_condition)]
                # Give points for our pieces, subtract for opponent pieces
                player_count = window.count(player)
                opponent_count = window.count(opponent)
                empty_count = window.count(0)
                
                # Only consider if there are no opponent pieces (can't win otherwise)
                if opponent_count == 0:
                    if player_count == win_condition - 1 and empty_count == 1:
                        score += 5  # Near win
                    elif player_count == win_condition - 2 and empty_count == 2:
                        score += 2  # Building threat
                    elif player_count == 1 and empty_count == win_condition - 1:
                        score += 0.5  # Starting position
                
                # Also check opponent's diagonal threats
                if player_count == 0:
                    if opponent_count == win_condition - 1 and empty_count == 1:
                        score -= 6  # Near loss - weigh higher than our threats
                    elif opponent_count == win_condition - 2 and empty_count == 2:
                        score -= 3  # Opponent building threat
        
        # Negative diagonals (\)
        for row in range(win_condition - 1, num_rows):
            for col in range(num_cols - (win_condition - 1)):
                window = [board[row-i][col+i] for i in range(win_condition)]
                # Give points for our pieces, subtract for opponent pieces
                player_count = window.count(player)
                opponent_count = window.count(opponent)
                empty_count = window.count(0)
                
                # Only consider if there are no opponent pieces (can't win otherwise)
                if opponent_count == 0:
                    if player_count == win_condition - 1 and empty_count == 1:
                        score += 5  # Near win
                    elif player_count == win_condition - 2 and empty_count == 2:
                        score += 2  # Building threat
                    elif player_count == 1 and empty_count == win_condition - 1:
                        score += 0.5  # Starting position
                
                # Also check opponent's diagonal threats
                if player_count == 0:
                    if opponent_count == win_condition - 1 and empty_count == 1:
                        score -= 6  # Near loss - weigh higher than our threats
                    elif opponent_count == win_condition - 2 and empty_count == 2:
                        score -= 3  # Opponent building threat
        
        return score
        
    def detect_advanced_patterns(self, player: int) -> Tuple[List[int], float]:
        """
        Detect advanced Connect Four patterns beyond basic threats.
        
        Args:
            player: The player to check patterns for
            
        Returns:
            Tuple[List[int], float]: List of recommended moves and pattern score
        """
        opponent = 3 - player
        moves = []
        pattern_score = 0
        board = self.board
        num_rows, num_cols = board.shape
        win_condition = self.game_board.win_condition  # Get win condition from game board
        
        # Check for double-threat creation (placing a piece that creates TWO three-in-a-rows)
        for col in range(num_cols):
            if not self.game_board.is_valid_location(col):
                continue
                
            # Find where the piece would land
            row = self.game_board.get_next_open_row(col)
            
            # Create a temporary board with this move
            temp_board = board.copy()
            temp_board[row][col] = player
            
            # Count threats in all directions
            threat_count = 0
            
            # Check horizontal threats
            for c in range(max(0, col-(win_condition-1)), min(col+1, num_cols-(win_condition-1))):
                if c + win_condition <= num_cols:
                    window = [temp_board[row][c+i] for i in range(win_condition)]
                    if window.count(player) == win_condition - 1 and window.count(0) == 1:
                        threat_count += 1
            
            # Check vertical threats
            if row >= win_condition - 1:
                window = [temp_board[row-i][col] for i in range(win_condition)]
                if window.count(player) == win_condition - 1 and window.count(0) == 1:
                    threat_count += 1
            
            # Check diagonal threats
            # Positive diagonal
            for i in range(win_condition):
                r = row - i
                c = col - i
                if r >= 0 and r <= num_rows - win_condition and c >= 0 and c <= num_cols - win_condition:
                    window = [temp_board[r+j][c+j] for j in range(win_condition)]
                    if window.count(player) == win_condition - 1 and window.count(0) == 1:
                        threat_count += 1
            
            # Negative diagonal
            for i in range(win_condition):
                r = row - i
                c = col + i
                if r >= 0 and r <= num_rows - win_condition and c >= win_condition - 1 and c < num_cols:
                    if all(0 <= r+j < num_rows and 0 <= c-j < num_cols for j in range(win_condition)):
                        window = [temp_board[r+j][c-j] for j in range(win_condition)]
                        if window.count(player) == win_condition - 1 and window.count(0) == 1:
                            threat_count += 1
            
            # If this creates multiple threats, it's a very strong move
            if threat_count >= 2:
                moves.append(col)
                pattern_score += threat_count * 7  # Valuable move
        
        return moves, pattern_score

class DPAgent:
    """
    Dynamic Programming agent for Connect4.
    Uses online policy iteration with limited horizon and beam search
    to compute optimal policies for the current game state.
    """
    
    def __init__(self, discount_factor: float = 0.9995, epsilon: float = 0.001, horizon: int = 18, beam_width: int = 800):
        """
        Initialize the DP agent.
        
        Args:
            discount_factor: The discount factor for future rewards (gamma)
            epsilon: The convergence threshold for value iteration
            horizon: The maximum depth to explore from current state
            beam_width: The maximum number of states to consider at each depth
        """
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.horizon = horizon
        self.beam_width = beam_width
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
    
    def _initialize_state(self, state: GameState) -> None:
        """Initialize a new state with default values and policy."""
        if state not in self.values:
            self.values[state] = self.V0
            self.policy[state] = None  # No policy yet for this state
            
    def choose_action(self, game_state: Dict) -> int:
        """
        Choose an action based on online policy iteration from the current state.
        Always runs the MDP process first, then validates the decision with defensive checks.
        
        Args:
            game_state: The current state of the game
            
        Returns:
            int: The column index where the agent wants to place its piece
        """
        start_time = time.time()
        
        # Convert dictionary game state to our GameState object
        state = self._convert_to_game_state(game_state)
        valid_actions = state.get_valid_actions()
        
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
        print("Performing online policy iteration with progressive beam widening...")
        self.online_policy_iteration_progressive(state)
        
        # Get the best action from the policy
        mdp_action = self.policy.get(state, None)
        
        # Print linear system for this state
        print(f"\n=== LINEAR SYSTEM FOR PLAYER {state.turn + 1} ===")
        coeff = self.get_linear_system(state)
        print("Coefficient matrix:")
        print(coeff)
        print(f"=== END LINEAR SYSTEM FOR PLAYER {state.turn + 1} ===\n")
        
        # If no policy available, evaluate actions directly
        if mdp_action is None or mdp_action not in valid_actions:
            print("Policy not available for current state. Evaluating actions directly...")
            mdp_action = self._evaluate_actions(state, valid_actions)
            
        # PHASE 2: DEFENSIVE CHECK - Validate the MDP's decision
        # This is now a safety check AFTER the MDP has run, not a replacement for it
        defensive_action = self._defensive_search(state)
        final_action = defensive_action if defensive_action is not None else mdp_action
        
        # If the defensive action overrides the MDP's choice, log this
        if defensive_action is not None and defensive_action != mdp_action:
            print(f"MDP chose column {mdp_action+1}, but defensive check overrode with column {defensive_action+1}")
        
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
        best_value = float('-inf')
        action_values = {}  # For debugging
        
        current_player = state.turn + 1  # Convert from 0/1 to 1/2
        
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
            
            if value > best_value:
                best_value = value
                best_action = action
        
        # Apply a small random perturbation to the action values to create variety
        if random.random() < 0.03:  # Reduced exploration probability from 5% to 3%
            exploration_coef = 0.05  # Reduced from 0.1 to 0.05
            exploration_values = {}
            for action in valid_actions:
                if action in action_values:
                    # Add random noise to value
                    noise = random.uniform(-exploration_coef, exploration_coef)
                    exploration_values[action] = action_values[action] + noise
                    
            # Find best action after adding noise
            if exploration_values:
                best_action_with_noise = max(exploration_values, key=exploration_values.get)
                if best_action_with_noise != best_action:
                    print(f"Exploration: changing action from {best_action+1} to {best_action_with_noise+1}")
                    best_action = best_action_with_noise
        
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
            print(f"Choosing best action: column {best_action+1} with value {action_values.get(best_action, 'N/A'):.2f}")
        
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
        self.iterations_performed += 1
        iteration = 0
        max_iterations = 100  # Allow more iterations for better convergence
        
        # Initialize debug information
        last_deltas = []
        
        while True:
            iteration += 1
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
                
                # Find the max Q-value for this state
                max_value = float('-inf')
                
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
                    
                    # Update max value
                    if value > max_value:
                        max_value = value
                
                # Update state value if we found a better value
                if max_value != float('-inf'):
                    old_value = old_values.get(state, self.V0)
                    self.values[state] = max_value
                    value_change = abs(old_value - max_value)
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
                print(f"Value iteration: {iteration} iterations, delta={delta:.6f}")
        
        # Print some debugging info about convergence
        if len(last_deltas) > 1:
            avg_delta = sum(last_deltas) / len(last_deltas)
            print(f"Value iteration converged after {iteration} iterations. Final delta={delta:.6f}, avg={avg_delta:.6f}")
    
    def policy_extraction(self, states: Set[GameState]) -> None:
        """
        Extract the optimal policy from the current value function.
        
        Args:
            states: Set of states to extract policy for
        """
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
            best_value = float('-inf')
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
                
                # Update best action if this is better
                if value > best_value:
                    best_value = value
                    best_action = action
            
            # Update policy for this state
            old_action = self.policy.get(state)
            if best_action is not None and best_action != old_action:
                self.policy[state] = best_action
                policy_updates += 1
                
                # Debug output for significant policy changes
                if old_action is not None:
                    print(f"Policy updated for state: turn={state.turn+1}, " 
                          f"old={old_action+1} (value={action_values.get(old_action, 'N/A')}), "
                          f"new={best_action+1} (value={action_values.get(best_action, 'N/A')})")
        
        print(f"Policy extraction complete. Updated {policy_updates} states out of {len(states)}.")
    
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
        last_player = 3 - current_player  # Previous player
        
        # Get win condition from the game board
        win_condition = state.game_board.win_condition
        
        # First check if last player won (current player loses)
        if state.game_board.winning_move(last_player):
            reward = -200.0  # Very strong negative reward for losing
            self.eval_cache[state_hash] = reward
            return reward
        
        # Check for draw
        if state.game_board.tie_move():
            reward = 0.0  # Neutral reward for draw
            self.eval_cache[state_hash] = reward
            return reward
        
        # Calculate positional reward based on pieces and threats
        reward = 0.0
        
        # Check for potential winning positions for the current player
        three_in_a_row = self._count_threats(board, current_player, win_condition-1, win_condition)
        two_in_a_row = self._count_threats(board, current_player, win_condition-2, win_condition)
        
        # Check for opponent threats
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

    # Linear system methods - preserved for future implementation
    def compute_bellman_equation(self, state: GameState) -> Dict:
        """Compute the Bellman equation for a state."""
        # This method can be implemented later for linear system analysis
        return {}
        
    def analyze_linear_system(self, state: GameState) -> None:
        """Analyze the linear system for a state."""
        # This method can be implemented later for linear system analysis
        pass
        
    def get_linear_system(self, state: GameState) -> np.ndarray:
        """Get the linear system for a state."""
        valid_actions = state.get_valid_actions()
        num_actions = len(valid_actions)
        
        # map all known states to a unique index
        coeff = np.zeros((num_actions, len(self.values) + 1))
        
        for i, action in enumerate(valid_actions):
            next_state = state.apply_action(action)
            reward = self._get_reward(next_state)
            
            coeff[i, i] = 1.0
            
            if next_state.is_terminal():
                coeff[i, -1] = reward
            else:
                state_ind = {state: idx for idx, state in enumerate(self.values.keys())}
                if next_state not in state_ind:
                    coeff[i, state_ind[next_state]] = -self.gamma
                    
                coeff[i, -1] = reward
                
        return coeff