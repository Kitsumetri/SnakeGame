import pygame as pg
import sys
from sprites import Snake, Apple, Blueberry, Lemon
from typing import List, Tuple

class Game:
    def __init__(self) -> None:
        pg.init()
        self.snake = None
        self.apple = None
        self.blueberry = None
        self.lemon = None

        self.sprite_group = pg.sprite.Group()
        self.WIDTH = 800
        self.HEIGHT = 800
        self.screen: pg.surface.Surface = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        pg.display.set_caption('Snake')
        self.clock = pg.time.Clock()

        self.BLACK: Tuple = (0, 0, 0)
        self.HITBOX_SIZE = 50

        self.WHITE: Tuple = (255, 255, 255)
        self.start_font: pg.font.Font = pg.font.SysFont('arial', 40)

        self.grid_color: Tuple = (50, 50, 50)

        self.start_new_game()

    def start_new_game(self) -> None:
        self.sprite_group.empty()
        self.show_go_screen()
        self.snake = Snake(self)
        self.apple = Apple(self)
        self.lemon = Lemon(self)
        self.blueberry = Blueberry(self)
        self.sprite_group.add((self.apple, self.blueberry, self.lemon))

    def draw_grid(self) -> None:
        for i in range(0, self.WIDTH, self.HITBOX_SIZE):
            pg.draw.line(self.screen, self.grid_color, (i, 0), (i, self.WIDTH))

        for i in range(0, self.HEIGHT, self.HITBOX_SIZE):
            pg.draw.line(self.screen, self.grid_color, (0, i), (self.HEIGHT, i))

    def show_go_screen(self) -> None:
        self.screen.fill(self.BLACK)
        texts: List[str] = ["Press space for starting new game!"]

        if self.snake: texts.append(f"Score: {self.apple.count + self.blueberry.count}")

        surfaces = list()
        for text in texts: surfaces.append(self.start_font.render(text,True, self.WHITE))

        new_line = 0
        for surface in surfaces:
            self.screen.blit(surface, (self.WIDTH // 6, self.HEIGHT // 2 + new_line))
            new_line += 50

        pg.display.flip()
        done = False
        while not done:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        done = True
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

    def update(self) -> None:
        self.snake.update()
        pg.display.flip()
        self.clock.tick(60)

    def draw(self) -> None:
        self.screen.fill(self.BLACK)
        self.draw_grid()
        self.sprite_group.draw(self.screen)
        self.snake.draw()

    def check_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            self.snake.get_input(event)

    def run(self) -> None:
        while True:
            self.check_events()
            self.update()
            self.draw()


if __name__ == '__main__':
    game = Game()
    game.run()
