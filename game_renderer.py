import os
from typing import Any, Optional, Union

import pygame
from pygame import mixer
from pygame.font import FontType
from pygame.ftfont import Font
from pygame.gfxdraw import aacircle, filled_circle

from assets import (black_coin, disc_drop_1, disc_drop_2, event_sound,
                    red_coin, yellow_coin)
from config import BLACK, BLUE, RED, WHITE, YELLOW
from events import GameOver, MouseHoverEvent, PieceDropEvent, bus
from game_data import GameData


@bus.on("piece:drop")
def on_piece_drop(event: PieceDropEvent):
    """
    Plays a sound when a piece is dropped over an empty slot.
    :param event: Information about the drop, namely the slot where the piece was dropped.
    """
    if event.side == 1:
        mixer.music.load(disc_drop_1)
        mixer.music.play(0)

    if event.side == 2:
        mixer.music.load(disc_drop_2)
        mixer.music.play(0)


class GameRenderer:
    """
    Renders the current game state to the screen and the speakers.
    """

    game_data: GameData
    label: Optional[Any]
    myfont: Union[None, Font, FontType]

    def __init__(self, screen, game_data: GameData):
        """
        Initializes the game renderer.
        :param screen: The screen.
        :param game_data: All of the data for the game.
        """
        self.myfont = pygame.font.SysFont("monospace", 75)
        self.label = self.myfont.render("CONNECT FOUR!!", 1, WHITE)
        screen.blit(self.label, (40, 10))
        self.screen = screen
        self.game_data = game_data
        self.stats = {}

        pygame.display.set_caption("Connect Four | Mayank Singh")
        pygame.display.update()

    def draw_stats_panel(self, stats):
        import game_data
        font = pygame.font.SysFont(None, 24)
        x_offset = self.game_data.width - self.game_data.panel_size+ 20
        y = 20

        def render_line(label, value):
            nonlocal y
            text_surface = font.render(f"{label}: {value}", True, (255, 255, 255))
            self.screen.blit(text_surface, (x_offset, y))
            y += 28

        render_line("State ID", stats.get("state_id", "-"))
        render_line("Action", stats.get("action", "-"))
        render_line("Reward", stats.get("reward", "-"))

        V = stats.get("V", [])
        if V:
            render_line("V[:5]", ", ".join(f"{v:.2f}" for v in V[:5]))

        eigenvalues = stats.get("eigenvalues", [])
        if eigenvalues:
            render_line("λ[0]", f"{eigenvalues[0]:.4f}")

    @bus.on("mouse:hover")
    def on_mouse_move(self, event: MouseHoverEvent):
        """
        Draws a coin over the slot that the mouse is positioned.
        :param event: Information about the hover, namely the x position
        """
        posx = event.posx

        pygame.draw.rect(
            self.screen, BLACK, (0, 0, self.game_data.width, self.game_data.sq_size)
        )
        self.draw_coin(
            self.game_data,
            posx - (self.game_data.sq_size / 2),
            int(self.game_data.sq_size) - self.game_data.sq_size + 5,
        )

    def draw_red_coin(self, x, y):
        """
        Draws a red coin.
        :param x: The x position to draw the coin.
        :param y: The y position to draw the coin.
        """
        self.screen.blit(red_coin, (x, y))

    def draw_yellow_coin(self, x, y):
        """
        Draws a yellow coin.
        :param x: The x position to draw the coin.
        :param y: The y position to draw the coin.
        """
        self.screen.blit(yellow_coin, (x, y))

    def draw_black_coin(self, x, y):
        """
        Draws a black coin.
        :param x: The x position to draw the coin.
        :param y: The y position to draw the coin.
        """
        self.screen.blit(black_coin, (x, y))

    def draw_coin(self, game_data, x, y):
        """
        Draws a coin to the specified position
        using the color of the current player.

        :param game_data: All of the data for the game.
        :param x: The x position for the coin to be drawn.
        :param y: The y position for the coin to be drawn.
        """
        if game_data.turn == 0:
            self.screen.blit(red_coin, (x, y))
        else:
            self.screen.blit(yellow_coin, (x, y))

    def draw(self, game_data: GameData):
        """
        Draws the game state, including the board and the pieces.
        :param game_data: All of the data associated with the game.
        """
        if game_data.action == "undo":
            filled_circle(
                self.screen,
                game_data.last_move_row,
                game_data.last_move_col,
                self.game_data.radius,
                BLACK,
            )

            aacircle(
                self.screen,
                game_data.last_move_row,
                game_data.last_move_col,
                self.game_data.radius,
                BLACK,
            )

            self.draw_black_coin(
                game_data.last_move_col * self.game_data.sq_size + 5,
                self.game_data.height
                - (
                    game_data.last_move_row * self.game_data.sq_size
                    + self.game_data.sq_size
                    - 5
                ),
            )

            game_data.game_board.print_board()
            game_data.action = None

        self.draw_board(game_data.game_board)

    @bus.on("game:over")
    def on_game_over(self, event: GameOver):
        """
        Handles a game over event.
        :param event: Data about how the game ended.
        """
        color = None

        if event.winner == 1:
            color = RED
        if event.winner == 2:
            color = YELLOW

        if not event.was_tie:
            self.label = self.myfont.render(f"PLAYER {event.winner} WINS!", 1, color)
            self.screen.blit(self.label, (40, 10))

            mixer.music.load(event_sound)
            mixer.music.play(0)
        else:
            mixer.music.load(os.path.join("sounds", "event.ogg"))
            mixer.music.play(0)
            self.myfont = pygame.font.SysFont("monospace", 75)
            self.label = self.myfont.render("GAME DRAW !!!!", 1, WHITE)
            self.screen.blit(self.label, (40, 10))

    def draw_board(self, board):
        """
        Draws the game board to the screen.
        :param board: The game board.
        """
        sq_size = 100
        height = 700
        radius = int(sq_size / 2 - 5)

        for c in range(board.cols):
            for r in range(board.rows):
                pygame.draw.rect(
                    self.screen,
                    BLUE,
                    (c * sq_size, (r + 1) * sq_size, sq_size, sq_size),
                )
                aacircle(
                    self.screen,
                    int(c * sq_size + sq_size / 2),
                    int((r + 1) * sq_size + sq_size / 2),
                    radius,
                    BLACK,
                )
                filled_circle(
                    self.screen,
                    int(c * sq_size + sq_size / 2),
                    int((r + 1) * sq_size + sq_size / 2),
                    radius,
                    BLACK,
                )

        for c in range(board.cols):
            for r in range(board.rows):
                if board.board[r][c] == 1:
                    self.draw_red_coin(
                        int(c * sq_size) + 5, height - int(r * sq_size + sq_size - 5)
                    )

                elif board.board[r][c] == 2:
                    self.draw_yellow_coin(
                        int(c * sq_size) + 5, height - int(r * sq_size + sq_size - 5)
                    )
        self.draw_stats_panel(self.stats)
        pygame.display.update()
