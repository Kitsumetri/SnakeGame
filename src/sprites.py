import pygame as pg
from random import randrange
from typing import Tuple
from pygame.math import Vector2 as vec2


class Snake:
    def __init__(self, game) -> None:
        self.game = game
        self.size: int = game.HITBOX_SIZE
        self.rect = pg.rect.Rect(0, 0, self.size - 2, self.size - 2)

        self.rect.center = self.get_random_pos()
        self.GREEN: Tuple = (0, 255, 0)

        self.direction: pg.math.Vector2 = vec2(0, 0)

        self.tick = 150
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

    def check_speed_food(self) -> None:
        if self.game.snake.rect.center == self.game.blueberry.rect.center:
            self.game.blueberry.rect.center = self.game.blueberry.get_random_pos()
            self.tick -= 4
            self.game.blueberry.count += 2

    def check_lemon_eating(self) -> None:
        if self.game.snake.rect.center == self.game.lemon.rect.center:
            self.game.lemon.rect.center = self.game.lemon.get_random_pos()
            if self.length > 1:
                self.length -= 1

    def check_boarders(self) -> None:
        if self.rect.left < 0 or self.rect.right > self.game.WIDTH:
            self.game.start_new_game()

        if self.rect.top < 0 or self.rect.bottom > self.game.HEIGHT:
            self.game.start_new_game()

    def check_tail_eating(self) -> None:
        if len(self.body) != len(set([body.center for body in self.body])):
            self.game.start_new_game()

    def move(self) -> None:
        if self.check_move():
            self.rect.move_ip(self.direction)
            self.body.append(self.rect.copy())
            self.body: list[pg.Rect] = self.body[-self.length:]

    def update(self) -> None:
        self.check_tail_eating()
        self.check_food()
        self.check_speed_food()
        self.check_boarders()
        self.check_lemon_eating()
        self.move()

    def draw(self) -> None:
        for hitbox in self.body:
            pg.draw.rect(self.game.screen, self.GREEN, hitbox)


class Apple(pg.sprite.Sprite):
    def __init__(self, game) -> None:
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.size: int = game.HITBOX_SIZE
        self.image: pg.surface.Surface = pg.image.load('sprites/apple_sprite.png').convert_alpha()
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

        while (pos_x, pos_y) in self.game.snake.body:
            pos_x: int = randrange(start=self.size // 2,
                                   stop=self.game.WIDTH - self.size // 2,
                                   step=self.size)
            pos_y: int = randrange(start=self.size // 2,
                                   stop=self.game.HEIGHT - self.size // 2,
                                   step=self.size)
        return pos_x, pos_y

    def update(self) -> None: pass

    def draw(self) -> None:
        self.game.screen.blit(self.image, self.rect)


class Blueberry(pg.sprite.Sprite):
    def __init__(self, game) -> None:
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.size: int = game.HITBOX_SIZE
        self.image: pg.surface.Surface = pg.image.load('sprites/blueberry_sprite.png').convert_alpha()
        self.rect: pg.Rect = self.image.get_rect()

        self.rect.center = self.get_random_pos()

        self.count = 0

    def get_random_pos(self) -> Tuple[int, int]:
        pos_x: int = randrange(start=self.size // 2,
                               stop=self.game.WIDTH - self.size // 2,
                               step=self.size)
        pos_y: int = randrange(start=self.size // 2,
                               stop=self.game.HEIGHT - self.size // 2,
                               step=self.size)

        while (pos_x, pos_y) in self.game.snake.body:
            pos_x: int = randrange(start=self.size // 2,
                                   stop=self.game.WIDTH - self.size // 2,
                                   step=self.size)
            pos_y: int = randrange(start=self.size // 2,
                                   stop=self.game.HEIGHT - self.size // 2,
                                   step=self.size)
        return pos_x, pos_y

    def update(self) -> None: pass

    def draw(self) -> None:
        self.game.screen.blit(self.image, self.rect)


class Lemon(pg.sprite.Sprite):
    def __init__(self, game) -> None:
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.size: int = game.HITBOX_SIZE
        self.image: pg.surface.Surface = pg.image.load('sprites/lemon_sprite.png').convert_alpha()
        self.rect: pg.Rect = self.image.get_rect()

        self.rect.center = self.get_random_pos()
        self.count = 0

    def get_random_pos(self) -> Tuple[int, int]:
        pos_x: int = randrange(start=self.size // 2,
                               stop=self.game.WIDTH - self.size // 2,
                               step=self.size)
        pos_y: int = randrange(start=self.size // 2,
                               stop=self.game.HEIGHT - self.size // 2,
                               step=self.size)

        while (pos_x, pos_y) in self.game.snake.body:
            pos_x: int = randrange(start=self.size // 2,
                                   stop=self.game.WIDTH - self.size // 2,
                                   step=self.size)
            pos_y: int = randrange(start=self.size // 2,
                                   stop=self.game.HEIGHT - self.size // 2,
                                   step=self.size)
        return pos_x, pos_y

    def update(self) -> None: pass

    def draw(self) -> None:
        self.game.screen.blit(self.image, self.rect)
