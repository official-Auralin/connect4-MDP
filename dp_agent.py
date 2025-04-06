from typing import Any, Dict, List, Tuple
import numpy as np
import copy

class DPAgent:
    """
    Dynamic Programming agent for Connect4.
    Uses value iteration to compute optimal policy and maintains linear systems
    for state transitions.
    """
    
    def __init__(self, discount_factor: float = 0.9, epsilon: float = 0.01):
        """
        Initialize the DP agent.
        
        Args:
            discount_factor: The discount factor for future rewards (gamma)
            epsilon: The convergence threshold for value iteration
        """
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.V0 = 0.0  # Initial value for all states
        self.states = set()  # Set of all possible states
        self.values = {}  # State -> value mapping (V(s))
        self.policy = {}  # State -> action mapping
        self.linear_systems = {}  # State -> linear system mapping
        
    def set_epsilon(self, epsilon: float) -> None:
        """
        Set the convergence threshold for value iteration.
        
        Args:
            epsilon: The new convergence threshold
        """
        self.epsilon = epsilon
        
    def set_discount_factor(self, discount_factor: float) -> None:
        """
        Set the discount factor for future rewards.
        
        Args:
            discount_factor: The new discount factor (gamma)
        """
        self.gamma = discount_factor
        
    def _initialize_state(self, state: str) -> None:
        """
        Initialize a new state with default values and policy.
        
        Args:
            state: The state to initialize
        """
        if state not in self.states:
            self.states.add(state)
            self.values[state] = self.V0
            self.policy[state] = None  # No policy yet for this state
            
    def choose_action(self, game_state: Any) -> int:
        """
        Choose an action based on the current policy.
        
        Args:
            game_state: The current state of the game
            
        Returns:
            int: The column index where the agent wants to place its piece
        """
        state = self._get_state_representation(game_state)
        return self.policy.get(state, 0)  # Default to column 0 if no policy exists
    
    def update(self, game_state: Any, reward: float) -> None:
        """
        Update the value function and policy based on the game outcome.
        
        Args:
            game_state: The current state of the game
            reward: The reward received
        """
        state = self._get_state_representation(game_state)
        self.values[state] = reward if reward != 0 else self.V0  # Use V0 for non-terminal states
    
    def reset(self) -> None:
        """Reset the agent's state for a new game."""
        self.states = set()
        self.values = {}
        self.policy = {}
        self.linear_systems = {}
    
    def policy_evaluation(self) -> None:
        """
        Evaluate the current policy by computing V(s) for all states.
        Uses iterative policy evaluation algorithm with synchronous updates.
        """
        while True:
            delta = 0
            # Make a copy of all values to use for this iteration
            old_values = self.values.copy()
            
            # Update each state's value using OLD values
            for state in self.states:
                if self.policy[state] is None:
                    continue
                
                # Get next state and reward using our granular functions
                game_state = self._state_to_game_state(state)
                action = self.policy[state]
                next_game_state = self._get_next_state(game_state, action)
                reward = self._get_reward(next_game_state)
                next_state = self._get_state_representation(next_game_state)
                
                # Update value using Bellman equation and OLD values
                self.values[state] = reward + self.gamma * old_values.get(next_state, self.V0)
                
                # Track maximum change
                delta = max(delta, abs(old_values[state] - self.values[state]))
            
            # Check for convergence
            if delta < self.epsilon:
                break
    
    def policy_extraction(self) -> None:
        """
        Extract the optimal policy from the current value function.
        Uses one-step lookahead to find the best action for each state.
        """
        # TODO: Implement policy extraction
        pass
    
    def policy_iteration(self) -> None:
        """
        Perform policy iteration to find the optimal policy.
        Alternates between policy evaluation and policy improvement until convergence.
        """
        # TODO: Implement policy iteration
        pass
    
    # Connect4-specific methods
    def _get_state_representation(self, game_state: Any) -> str:
        """
        Convert Connect4 board state to a hashable representation.
        
        Args:
            game_state: The current Connect4 board state
            
        Returns:
            str: A string representation of the board state
        """
        # TODO: Implement board state to string conversion
        pass
    
    def _get_valid_actions(self, game_state: Any) -> List[int]:
        """
        Get all valid column moves for the current Connect4 board state.
        
        Args:
            game_state: The current Connect4 board state
            
        Returns:
            List[int]: List of valid column indices (0-6)
        """
        board = game_state['board']
        return [col for col in range(7) if board[5][col] == 0]  # Check top row
    
    def _get_next_state(self, game_state: Any, action: int) -> Any:
        """
        Simulate placing a piece in the given column and return the resulting board state.
        
        Args:
            game_state: The current Connect4 board state
            action: The column index where to place the piece
            
        Returns:
            Any: The resulting board state after placing the piece
        """
        # Create a deep copy of the board to simulate the move
        next_state = copy.deepcopy(game_state)
        board = next_state['board']
        
        # Find the next open row in the chosen column
        for row in range(6):  # Connect4 board is 6x7
            if board[row][action] == 0:  # Empty spot
                board[row][action] = next_state['turn'] + 1  # Player 1 or 2
                break
                
        # Update turn
        next_state['turn'] = (next_state['turn'] + 1) % 2
        return next_state
    
    def _get_reward(self, game_state: Any) -> float:
        """
        Get the reward for the current Connect4 board state.
        
        Args:
            game_state: The current Connect4 board state
            
        Returns:
            float: Reward value (+1 for win, -1 for loss, 0 for draw/ongoing)
        """
        board = game_state['board']
        current_player = game_state['turn'] + 1  # Player 1 or 2
        
        # Use game's built-in win checking for the previous player
        last_player = 3 - current_player  # Previous player
        if game_state['game_board'].winning_move(last_player):
            return -1.0 if last_player == current_player else 1.0
            
        # Check for draw (full board)
        if game_state['game_board'].tie_move():
            return 0.0
            
        return 0.0  # Non-terminal state
    
    # Linear system methods
    def _compute_linear_system(self, state: str) -> np.ndarray:
        """
        Compute the linear system for a given Connect4 state.
        The linear system represents transition probabilities and expected rewards.
        
        Args:
            state: The state to compute the linear system for
            
        Returns:
            np.ndarray: The linear system matrix
        """
        # TODO: Implement linear system computation
        pass
    
    def get_linear_system(self, state: str) -> np.ndarray:
        """
        Get the linear system for a given state.
        
        Args:
            state: The state to get the linear system for
            
        Returns:
            np.ndarray: The linear system matrix
        """
        if state not in self.linear_systems:
            self.linear_systems[state] = self._compute_linear_system(state)
        return self.linear_systems[state]
    
    def _state_to_game_state(self, state: str) -> Dict:
        """
        Convert state string representation back to game state dictionary.
        
        Args:
            state: String representation of state
            
        Returns:
            Dict: Game state dictionary with board and turn information
        """
        # TODO: Implement conversion from state string to game state
        # This should be the inverse of _get_state_representation
        pass 