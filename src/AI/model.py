import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from collections import deque
from random import sample, randint
from src.AI.game_env import GameEnv, Point, Direction
from torchsummary import summary

# HyperParams
##########################################
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
NUM_EPOCHS = 200
GAMMA = 0.1
##########################################

Losses = []

class QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, X):
        out = self.network(X)
        return out

class QTrainner:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(model.parameters(), self.lr)
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, new_state, done):

        states = torch.tensor(np.array(state), dtype=torch.float)
        actions = torch.tensor(np.array(action), dtype=torch.long)
        rewards = torch.tensor(np.array(reward), dtype=torch.float)
        new_states = torch.tensor(np.array(new_state), dtype=torch.float)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            new_states = torch.unsqueeze(new_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            done = (done, )

        prediction = self.model(states)

        # Q_new = reward + gamma * max(q)
        target = prediction.clone()

        for i in range(len(done)):
            Q_new = rewards[i]
            if not done[i]:
                Q_new = rewards[i] + self.gamma * torch.max(self.model(new_states[i]))

            target[i][torch.argmax(actions).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_function(target, prediction)
        Losses.append(loss)
        loss.backward()

        self.optimizer.step()

class Agent:
    def __init__(self):
        self.numberOfGames = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = QNet(11, [256, 128], 3)
        self.trainner = QTrainner(self.model, lr=LR, gamma=self.gamma)

    @staticmethod
    def get_state(game):
        head = game.head

        point_left = Point(head.x - GameEnv.BLOCK_SIZE, head.y)
        point_right = Point(head.x + GameEnv.BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - GameEnv.BLOCK_SIZE)
        point_down = Point(head.x, head.y + GameEnv.BLOCK_SIZE)

        point_up_left = Point(head.x - GameEnv.BLOCK_SIZE, head.y - GameEnv.BLOCK_SIZE)
        point_up_right = Point(head.x + GameEnv.BLOCK_SIZE, head.y - GameEnv.BLOCK_SIZE)
        point_down_left = Point(head.x - GameEnv.BLOCK_SIZE, head.y + GameEnv.BLOCK_SIZE)
        point_down_right = Point(head.x + GameEnv.BLOCK_SIZE, head.y + GameEnv.BLOCK_SIZE)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),

            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_right and game.is_collision(point_down)),

            (direction_down and game.is_collision(point_right)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)),

            # game.is_collision(point_up_left),
            # game.is_collision(point_down_left),
            # game.is_collision(point_up_right),
            # game.is_collision(point_down_right),

            direction_left,
            direction_right,
            direction_up,
            direction_down,

            # direction_up and game.is_collision(point_up),
            # direction_down and game.is_collision(point_down),
            # direction_right and game.is_collision(point_right),
            # direction_left and game.is_collision(point_left),

            # game.is_collision(point_up_right),
            # game.is_collision(point_up_left),
            # game.is_collision(point_down_right),
            # game.is_collision(point_down_left),

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,

            # game.food.x < game.head.x and game.food.y < game.head.y,
            # game.food.x < game.head.x and game.food.y > game.head.y,
            # game.food.x > game.head.x and game.food.y < game.head.y,
            # game.food.x > game.head.x and game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            sample_mem = self.memory
        else:
            sample_mem = sample(self.memory, BATCH_SIZE)

        states, actions, rewards, nextStates, dones = zip(*sample_mem)
        self.trainner.train_step(states, actions, rewards, nextStates, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainner.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.numberOfGames
        finalMove = [0, 0, 0]

        if randint(0, 200) < self.epsilon:
            move = randint(0, 2)
            finalMove[move] = 1
        else:
            stateTensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(stateTensor)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove


all_scores = []
average_scores = []


def train():
    totalScore = 0
    bestScore = 0

    agent = Agent()
    game = GameEnv()

    while agent.numberOfGames < NUM_EPOCHS:
        game.epoch = agent.numberOfGames
        old_state = agent.get_state(game)
        final_move = agent.get_action(old_state)
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            game.start_new_game()
            agent.numberOfGames += 1
            agent.train_long_memory()

            if score > bestScore:
                bestScore = score

            totalScore += score
            meanScore = (totalScore / agent.numberOfGames)

            all_scores.append(score)
            average_scores.append(meanScore)

            if score == bestScore or agent.numberOfGames % 10 == 0:
                print(f"Game number: {agent.numberOfGames: <5}|"
                      f"Score: {score: <5}|"
                      f"Best Score:{bestScore: <5}|"
                      f"Mean scores: {meanScore: <5}")


if __name__ == '__main__':
    train()
    # summary(QNet(11, [256, 128], 3), (11,), device='cpu')
    # print([x for x in LinearQNet(11, 256, 3).parameters()])




