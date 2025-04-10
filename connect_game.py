import math
import os
import sys
import random

import pygame

from config import BLACK
from events import GameOver, MouseClickEvent, PieceDropEvent, bus
from game_data import GameData
from game_renderer import GameRenderer


class ConnectGame:
    """
    Holds all of the game logic and game data.
    """

    game_data: GameData
    renderer: GameRenderer

    def __init__(self, game_data: GameData, renderer: GameRenderer):
        """
        Initializes the connect game.
        :param game_data: A reference to the game data object.
        :param renderer: A reference to the game renderer.
        """
        self.game_data = game_data
        self.renderer = renderer

    def quit(self):
        """
        Exits the game.
        """
        sys.exit()

    def make_move(self, col: int) -> bool:
        """
        Make a move in the specified column.
        
        Args:
            col: The column to make the move in
            
        Returns:
            bool: True if the move was successful, False otherwise
        """
        if self.game_data.game_board.is_valid_location(col):
            row = self.game_data.game_board.get_next_open_row(col)
            
            self.game_data.last_move_row.append(row)
            self.game_data.last_move_col.append(col)
            self.game_data.game_board.drop_piece(row, col, self.game_data.turn + 1)
            
            self.draw()
            bus.emit("piece:drop", PieceDropEvent(self.game_data.game_board.board[row][col]))
            self.print_board()
            
            if self.game_data.game_board.winning_move(self.game_data.turn + 1):
                # Determine winning player and update agent reward if needed
                winning_player = self.game_data.turn + 1
                self.update_agent_reward(winning_player)
                
                bus.emit("game:over", self.renderer, GameOver(False, winning_player))
                self.game_data.game_over = True
                
            pygame.display.update()
            self.game_data.turn += 1
            self.game_data.turn = self.game_data.turn % 2
            return True
        return False
        
    def update_agent_reward(self, winning_player=None):
        """
        Update agent with reward based on game outcome.
        
        Args:
            winning_player: The player who won (1 or 2), or None if tie
        """
        if self.game_data.game_mode not in ['pva', 'ava']:
            return
            
        game_state = self.game_data.get_state_for_agent()
        
        # Determine reward based on outcome
        if winning_player is None:  # Tie
            reward = 0.0
            print("Game ended in a tie. Agent reward: 0.0")
        elif (winning_player == 2 and self.game_data.game_mode == 'pva') or \
             (self.game_data.game_mode == 'ava'):  # Agent win
            reward = 10.0
            print("Agent won! Reward: 10.0")
        else:  # Agent loss
            reward = -10.0
            print("Agent lost. Reward: -10.0")
            
        # Update agent with final reward
        if self.game_data.agent1:
            self.game_data.agent1.update(game_state, reward)
        
    @bus.on("mouse:click")
    def mouse_click(self, event: MouseClickEvent):
        """
        Handles a mouse click event.
        :param event: Data about the mouse click
        """
        pygame.draw.rect(
            self.renderer.screen,
            BLACK,
            (0, 0, self.game_data.width, self.game_data.sq_size),
        )
        
        col = int(math.floor(event.posx / self.game_data.sq_size))
        # Add bounds checking to ensure column is valid (0-6)
        if 0 <= col < self.game_data.game_board.cols:
            self.make_move(col)
        # If col is outside valid range, ignore the click
        
    def handle_agent_move(self) -> None:
        """
        Handle agent moves when it's their turn.
        """
        if self.game_data.game_over:
            return
            
        current_agent = None
        if self.game_data.game_mode == 'pva' and self.game_data.turn == 1:
            current_agent = self.game_data.agent1
        elif self.game_data.game_mode == 'ava':
            current_agent = self.game_data.agent1 if self.game_data.turn == 0 else self.game_data.agent2
            
        if current_agent:
            game_state = self.game_data.get_state_for_agent()
            col = current_agent.choose_action(game_state)
            # Validate column before making move
            if 0 <= col < self.game_data.game_board.cols:
                self.make_move(col)
            else:
                print(f"Agent tried to make an invalid move: column {col}")
                # Choose a random valid column instead
                valid_cols = [c for c in range(self.game_data.game_board.cols) 
                             if self.game_data.game_board.is_valid_location(c)]
                if valid_cols:
                    col = random.choice(valid_cols)
                    self.make_move(col)

    def update(self):
        """
        Checks the game state, dispatching events as needed.
        """
        if self.game_data.game_board.tie_move():
            # Update agent with tie reward
            self.update_agent_reward(None)
            
            bus.emit("game:over", self.renderer, GameOver(was_tie=True))
            self.game_data.game_over = True
            
        if not self.game_data.game_over:
            self.handle_agent_move()
            
        if self.game_data.game_over:
            print(os.getpid())
            pygame.time.wait(1000)
            
            # Use the correct path to the game.py file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            game_path = os.path.join(script_dir, "game.py")
            
            # Use python to run the game script
            if os.path.exists(game_path):
                os.system(f"python {game_path}")
            else:
                print(f"Error: Could not find {game_path}")
                print(f"Current directory: {os.getcwd()}")

    def draw(self):
        """
        Directs the game renderer to 'render' the game state to the audio and video devices.
        """
        self.renderer.draw(self.game_data)
        

    def print_board(self):
        """
        Prints the state of the board to the console.
        """
        self.game_data.game_board.print_board()
