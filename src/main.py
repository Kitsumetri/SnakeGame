import pygame as pg
import sys
from sprites import Snake, Apple
from typing import Tuple

class Game:
    def __init__(self) -> None:
        pg.init()
        self.snake = None
        self.apple = None

        self.sprite_group = pg.sprite.Group()
        self.WIDTH = 800
        self.HEIGHT = 800
        self.start_speed = 150
        self.screen: pg.surface.Surface = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        pg.display.set_caption('Snake')
        self.clock = pg.time.Clock()

        self.BLACK: Tuple = (0, 0, 0)
        self.HITBOX_SIZE = 50

        self.WHITE: Tuple = (255, 255, 255)
        self.score_font: pg.font.Font = pg.font.Font(pg.font.get_default_font(), 24)

        self.grid_color: Tuple = (50, 50, 50)

        self.start_new_game()

    def start_new_game(self) -> None:
        self.sprite_group.empty()
        self.snake = Snake(self)
        self.apple = Apple(self)
        self.sprite_group.add(self.apple)

    def check_game_over_condition(self):
        return self.snake.check_boarders() or self.snake.check_tail_eating()

    def draw_grid(self) -> None:
        for i in range(0, self.WIDTH, self.HITBOX_SIZE):
            pg.draw.line(self.screen, self.grid_color, (i, 0), (i, self.WIDTH))

        for i in range(0, self.HEIGHT, self.HITBOX_SIZE):
            pg.draw.line(self.screen, self.grid_color, (0, i), (self.HEIGHT, i))

    def update(self) -> None:
        self.snake.update()
        pg.display.flip()
        self.clock.tick(60)

    def draw(self) -> None:
        self.screen.fill(self.BLACK)
        score_text = self.score_font.render(f"Score: {self.apple.count: <8}"
                                            f"Speed: {self.snake.tick}",
                                            True, self.WHITE)
        score_text_rect = score_text.get_rect(x=10, y=10)
        self.draw_grid()
        self.sprite_group.draw(self.screen)
        self.snake.draw()
        self.screen.blit(score_text, score_text_rect)

    def check_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            self.snake.get_input(event)

    def run(self) -> None:
        while not self.check_events():
            if self.check_game_over_condition():
                self.start_new_game()
            self.update()
            self.draw()

if __name__ == '__main__':
    game = Game()
    game.run()
