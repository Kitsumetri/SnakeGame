import pygame as pg

from pygame.math import Vector2 as Point
from random import randint
from enum import Enum, unique
import numpy as np
from typing import Final, Tuple, NoReturn


@unique
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class GameEnv:
    WHITE: Final = pg.Color(255, 255, 255)
    RED: Final = pg.Color(200, 0, 0)
    GREEN: Final = pg.Color(0, 255, 0)
    BLACK: Final = pg.Color(0, 0, 0)
    GRAY: Final = pg.Color(50, 50, 50)
    BLOCK_SIZE: Final = 50
    SPEED: Final = 150

    def __init__(self) -> NoReturn:
        self.snake = None
        self.head = None
        self.direction = None
        self.frame_iteration = None
        self.food = None
        self.score = None

        self.epoch = None

        pg.init()
        self.font = pg.font.Font(pg.font.get_default_font(), 25)

        self.w = 800
        self.h = 800

        self.display = pg.display.set_mode((self.w, self.h))
        pg.display.set_caption("SnakeAI")
        self.clock = pg.time.Clock()

        self.start_new_game()

    def manhattan_distance(self):
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

    def draw_grid(self) -> NoReturn:
        for i in range(0, self.w, self.BLOCK_SIZE):
            pg.draw.line(self.display, GameEnv.GRAY, (i, 0), (i, self.w))

        for i in range(0, self.h, self.BLOCK_SIZE):
            pg.draw.line(self.display, GameEnv.GRAY, (0, i), (self.h, i))

    def start_new_game(self) -> NoReturn:
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - GameEnv.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * GameEnv.BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None

        self.place_food()
        self.frame_iteration = 0

    def place_food(self) -> NoReturn:
        x = randint(0, (self.w - GameEnv.BLOCK_SIZE) // GameEnv.BLOCK_SIZE) * GameEnv.BLOCK_SIZE
        y = randint(0, (self.h - GameEnv.BLOCK_SIZE) // GameEnv.BLOCK_SIZE) * GameEnv.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def play_step(self, action) -> Tuple[int, bool, int]:
        self.frame_iteration += 1

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit()

        self.move_snake(action)
        reward = 0

        if self.is_collision():
            gameOver = True
            reward -= 25
            return reward, gameOver, self.score

        if self.frame_iteration > 75 * len(self.snake):
            gameOver = True
            reward -= 50
            return reward, gameOver, self.score

        if self.head == self.food:
            self.score += 1
            reward += 100
            self.place_food()
        else:
            self.snake.pop()

        self.update()
        self.clock.tick(GameEnv.SPEED)
        gameOver = False

        return reward, gameOver, self.score

    def is_collision(self, p: Point = None) -> bool:
        if p is None:
            p = self.head

        if p.x > self.w - GameEnv.BLOCK_SIZE or p.x < 0:
            return True

        if p.y > self.h - GameEnv.BLOCK_SIZE or p.y < 0:
            return True

        if p in self.snake[1:]:
            return True
        return False

    def move_snake(self, action) -> NoReturn:
        # action -> [straight, right, left]
        dirs = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        cur_direction_index = dirs.index(self.direction)

        newDirection = self.direction

        if np.array_equal(action, [0, 1, 0]):
            newDirection = dirs[(cur_direction_index + 1) % 4]
        elif np.array_equal(action, [0, 1, 0]):
            newDirection = dirs[(cur_direction_index - 1) % 4]

        self.direction = newDirection

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += GameEnv.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= GameEnv.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += GameEnv.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= GameEnv.BLOCK_SIZE

        self.head = Point(x, y)

        self.snake.insert(0, self.head)

    def draw(self) -> NoReturn:
        self.display.fill(GameEnv.BLACK)
        self.draw_grid()
        [pg.draw.rect(self.display,
                      GameEnv.GREEN,
                      pg.Rect(p.x, p.y, GameEnv.BLOCK_SIZE-2, GameEnv.BLOCK_SIZE-2))
         for p in self.snake]

        pg.draw.rect(self.display,
                     GameEnv.RED,
                     pg.Rect(self.food.x, self.food.y, GameEnv.BLOCK_SIZE-2, GameEnv.BLOCK_SIZE-2))

        score_text = self.font.render(f"Score: {self.score: <8}"
                                      f"Speed: {GameEnv.SPEED: <8}"
                                      f"Move count: {self.frame_iteration: <8}"
                                      f"Epoch: {self.epoch}",
                                      True, self.WHITE)

        self.display.blit(score_text, score_text.get_rect(x=10, y=10))

    def update(self) -> NoReturn:
        self.draw()
        pg.display.flip()
