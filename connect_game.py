import math
import os
import sys

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
                bus.emit("game:over", self.renderer, GameOver(False, self.game_data.turn + 1))
                self.game_data.game_over = True
                
            pygame.display.update()
            self.game_data.turn += 1
            self.game_data.turn = self.game_data.turn % 2
            return True
        return False
        
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
        self.make_move(col)
        
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
            self.make_move(col)
            
    def update(self):
        """
        Checks the game state, dispatching events as needed.
        """
        if self.game_data.game_board.tie_move():
            bus.emit("game:over", self.renderer, GameOver(was_tie=True))
            self.game_data.game_over = True
            
        if not self.game_data.game_over:
            self.handle_agent_move()
            
        if self.game_data.game_over:
            print(os.getpid())
            pygame.time.wait(1000)
            os.system("game.py")

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
