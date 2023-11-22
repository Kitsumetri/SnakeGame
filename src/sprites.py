import pygame as pg
from random import randrange
from typing import Tuple
from pygame.math import Vector2 as vec2


class Snake(pg.sprite.Sprite):
    def __init__(self, game) -> None:
        super().__init__()
        self.game = game
        self.size: int = game.HITBOX_SIZE
        self.GREEN: pg.Color = pg.Color(0, 255, 0)
        self.image = pg.Surface((self.size - 2, self.size - 2))
        self.image.fill(self.GREEN)
        self.rect = self.image.get_rect(center=self.get_random_pos())

        self.direction: pg.math.Vector2 = vec2(0, 0)

        self.tick = self.game.start_speed
        self.time = 0

        self.length = 1
        self.body = list()

        self.keys_state: dict[int, bool] = {pg.K_w: True, pg.K_s: True, pg.K_a: True, pg.K_d: True}

    def check_move(self) -> bool:
        time_now: int = pg.time.get_ticks()
        if time_now - self.time > self.tick:
            self.time = time_now
            return True
        return False

    def get_input(self, event: pg.event.Event) -> None:
        if event.type == pg.KEYDOWN:
            match event.key:
                case pg.K_w:
                    if self.keys_state[pg.K_w]:
                        self.direction = vec2(0, -self.size)
                        self.keys_state = {pg.K_w: True, pg.K_s: False, pg.K_a: True, pg.K_d: True}
                case pg.K_s:
                    if self.keys_state[pg.K_s]:
                        self.direction = vec2(0, self.size)
                        self.keys_state = {pg.K_w: False, pg.K_s: True, pg.K_a: True, pg.K_d: True}
                case pg.K_a:
                    if self.keys_state[pg.K_a]:
                        self.direction = vec2(-self.size, 0)
                        self.keys_state = {pg.K_w: True, pg.K_s: True, pg.K_a: True, pg.K_d: False}
                case pg.K_d:
                    if self.keys_state[pg.K_d]:
                        self.direction = vec2(self.size, 0)
                        self.keys_state = {pg.K_w: True, pg.K_s: True, pg.K_a: False, pg.K_d: False}

    def get_random_pos(self) -> Tuple[int, int]:
        pos_x: int = randrange(start=self.size // 2,
                               stop=self.game.WIDTH - self.size // 2,
                               step=self.size)
        pos_y: int = randrange(start=self.size // 2,
                               stop=self.game.HEIGHT - self.size // 2,
                               step=self.size)
        return pos_x, pos_y

    def check_food(self) -> None:
        if self.game.snake.rect.center == self.game.apple.rect.center:
            self.game.apple.rect.center = self.game.apple.get_random_pos()
            self.length += 1
            self.game.apple.count += 1

    def check_boarders(self) -> bool:
        if self.rect.left < 0 or self.rect.right > self.game.WIDTH:
            return True

        if self.rect.top < 0 or self.rect.bottom > self.game.HEIGHT:
            return True

    def check_tail_eating(self) -> bool:
        for body in self.body:
            if self.body.count(body) > 1:
                return True

    def move(self) -> None:
        if self.check_move():
            self.rect.move_ip(self.direction)
            self.body.append(self.rect.copy())
            self.body: list[pg.Rect] = self.body[-self.length:]

    def update(self) -> None:
        self.check_food()
        self.move()

    def draw(self) -> None:
        [self.game.screen.blit(self.image, hitbox) for hitbox in self.body]


class Apple(pg.sprite.Sprite):
    def __init__(self, game) -> None:
        super().__init__()
        self.game = game
        self.size: int = game.HITBOX_SIZE
        self.image: pg.surface.Surface = pg.Surface((50, 50))
        self.image.fill(pg.Color(255, 0, 0))
        self.rect: pg.Rect = self.image.get_rect()
        self.count = 0

        self.rect.center = self.get_random_pos()

    def get_random_pos(self) -> Tuple[int, int]:
        pos_x: int = randrange(start=self.size // 2,
                               stop=self.game.WIDTH - self.size // 2,
                               step=self.size)
        pos_y: int = randrange(start=self.size // 2,
                               stop=self.game.HEIGHT - self.size // 2,
                               step=self.size)

        for body in self.game.snake.body:
            while ((pos_x, pos_y) == body.center or
                   (pos_x, pos_y) == self.game.apple.rect.center):
                pos_x: int = randrange(start=self.size // 2,
                                       stop=self.game.WIDTH - self.size // 2,
                                       step=self.size)
                pos_y: int = randrange(start=self.size // 2,
                                       stop=self.game.HEIGHT - self.size // 2,
                                       step=self.size)
        return pos_x, pos_y

    def draw(self) -> None:
        self.game.screen.blit(self.image, self.rect)