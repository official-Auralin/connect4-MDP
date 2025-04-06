import sys

import pygame
from pygame.locals import KEYDOWN

from config import BLACK, BLUE, WHITE, RED
from connect_game import ConnectGame
from events import MouseClickEvent, MouseHoverEvent, bus
from game_data import GameData
from game_renderer import GameRenderer


def quit():
    sys.exit()


def start(mode: str = 'pvp'):
    data = GameData()
    data.set_game_mode(mode)
    screen = pygame.display.set_mode(data.size)
    game = ConnectGame(data, GameRenderer(screen, data))

    game.print_board()
    game.draw()

    pygame.display.update()
    pygame.time.wait(1000)

    # Processes mouse and keyboard events, dispatching events to the event bus.
    # The events are handled by the ConnectGame and GameRenderer classes.
    while not game.game_data.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.quit()

            if event.type == pygame.MOUSEMOTION:
                bus.emit("mouse:hover", game.renderer, MouseHoverEvent(event.pos[0]))

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                bus.emit("mouse:click", game, MouseClickEvent(event.pos[0]))

            if event.type == KEYDOWN:
                if event.key == pygame.K_z:
                    mods: int = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL:
                        bus.emit("game:undo", game)

            game.update()
            game.draw()


def text_objects(text, font, color):
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()


def message_display(text, color, p, q, v):
    largeText = pygame.font.SysFont("monospace", v)
    TextSurf, TextRect = text_objects(text, largeText, color)
    TextRect.center = (p, q)
    screen.blit(TextSurf, TextRect)


pygame.init()
screen = pygame.display.set_mode(GameData().size)
pygame.display.set_caption("Connect Four | Mayank Singh")
message_display("CONNECT FOUR!!", WHITE, 350, 150, 75)
message_display("HAVE FUN!", (23, 196, 243), 350, 300, 75)

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    def button(msg, x, y, w, h, ic, ac, action=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if x + w > mouse[0] > x and y + h > mouse[1] > y:
            pygame.draw.rect(screen, ac, (x, y, w, h))
            # Draw slightly smaller black rectangle inside
            pygame.draw.rect(screen, BLACK, (x+2, y+2, w-4, h-4))
            if click[0] == 1 and action != None:
                action()
        else:
            pygame.draw.rect(screen, ic, (x, y, w, h))
            # Draw slightly smaller black rectangle inside
            pygame.draw.rect(screen, BLACK, (x+2, y+2, w-4, h-4))

        smallText = pygame.font.SysFont("monospace", 30)
        textSurf, textRect = text_objects(msg, smallText, WHITE)
        textRect.center = ((x + (w / 2)), (y + (h / 2)))
        screen.blit(textSurf, textRect)

    # Game mode buttons
    button_width = 300
    button_height = 50
    button_x = (700 - button_width) // 2  # Center horizontally (screen width is 700)
    
    # Main menu buttons
    button("Player vs Player", button_x, 400, button_width, button_height, WHITE, BLUE, lambda: start('pvp'))
    button("Player vs Agent", button_x, 470, button_width, button_height, WHITE, BLUE, lambda: start('pva'))
    button("Agent vs Agent", button_x, 540, button_width, button_height, WHITE, BLUE, lambda: start('ava'))
    
    # Quit button - centered and below other buttons
    quit_width = 150
    quit_x = (700 - quit_width) // 2
    button("QUIT", quit_x, 610, quit_width, button_height, WHITE, RED, quit)
    pygame.display.update()
