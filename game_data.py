from typing import Tuple, Optional, Any

from game_board import GameBoard
from dp_agent import DPAgent


class GameData:
    """
    The game data class contains all of the data for the game.
    """

    radius: int
    height: int
    width: int
    sq_size: int
    size: Tuple[int, int]
    game_over: bool
    turn: int
    last_move_row: [int]
    last_move_col: [int]
    game_board: GameBoard
    
    # Agent-related fields
    game_mode: str  # 'pvp', 'pva', 'ava'
    agent1: Optional[DPAgent]
    agent2: Optional[DPAgent]

    def __init__(self):
        self.game_over = False
        self.turn = 0
        self.last_move_row = []
        self.last_move_col = []
        self.game_board = GameBoard()
        self.action = None
        self.panel_size = 400
        self.sq_size: int = 100
        self.width: int = 7 * self.sq_size + self.panel_size
        self.height: int = 7 * self.sq_size
        self.size: Tuple[int, int] = (self.width, self.height)
        self.radius: int = int(self.sq_size / 2 - 5)
        
        # Initialize agent-related fields
        self.game_mode = 'pvp'  # Default to player vs player
        self.agent1 = None
        self.agent2 = None

    def set_game_mode(self, mode: str) -> None:
        """
        Set the game mode and initialize agents if needed.
        
        Args:
            mode: 'pvp' for player vs player, 'pva' for player vs agent,
            'ava' for agent vs agent
        """
        self.game_mode = mode
        if mode in ['pva', 'ava']:
            # Create a new agent - no pre-training needed since it uses online learning
            if self.agent1 is None:
                print("Initializing agent...")
                self.agent1 = DPAgent()
            else:
                # Reset the agent for a new game but preserve its learned values
                print("Resetting agent for new game...")
                self.agent1.reset()
                
        if mode == 'ava':
            # For agent vs agent, we'll use the same agent for both
            self.agent2 = self.agent1

    def get_state_for_agent(self) -> Any:
        """
        Convert the current game state to a format suitable for the agent.
        
        Returns:
            Any: The game state in agent-readable format
        """
        return {
            'board': self.game_board.board,
            'turn': self.turn,
            'game_board': self.game_board,  # Include the game board reference
            'last_move': (self.last_move_row[-1] if self.last_move_row else None, 
                          self.last_move_col[-1] if self.last_move_col else None)
        }
