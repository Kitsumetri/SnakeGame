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

class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        layers = [nn.Linear(inputSize, hiddenSize), nn.ReLU(), nn.Linear(hiddenSize, outputSize)]
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, X):
        out = self.hidden_layers(X)

        return out


class QTrainner:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(model.parameters(), self.lr)
        self.lossFunction = nn.MSELoss()

    def train_step(self, state, action, reward, newState, done):

        states = torch.tensor(np.array(state), dtype=torch.float)
        actions = torch.tensor(np.array(action), dtype=torch.long)
        rewards = torch.tensor(np.array(reward), dtype=torch.float)
        new_states = torch.tensor(np.array(newState), dtype=torch.float)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            new_states = torch.unsqueeze(new_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            done = (done, )

        # 1. predicted q values with current state
        prediction = self.model(states)

        # Q_new = reward + gamma * max(next predicted q value)
        target = prediction.clone()

        for i in range(len(done)):
            Q_new = rewards[i]
            if not done[i]:
                Q_new = rewards[i] + self.gamma * torch.max(self.model(new_states[i]))

            target[i][torch.argmax(actions).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.lossFunction(target, prediction)
        Losses.append(loss)
        loss.backward()

        self.optimizer.step()

class Agent:
    def __init__(self):
        self.numberOfGames = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = LinearQNet(11, 256, 3)
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

    def train_short_memory(self, state, action, reward, nextState, done):
        self.trainner.train_step(state, action, reward, nextState, done)

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
        oldState = agent.get_state(game)
        finalMove = agent.get_action(oldState)
        reward, done, score = game.play_step(finalMove)
        newState = agent.get_state(game)

        agent.train_short_memory(oldState, finalMove, reward, newState, done)
        agent.remember(oldState, finalMove, reward, newState, done)

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
    # summary(LinearQNet(11, 256, 3), (11, ), device='cpu')
    # print([x for x in LinearQNet(11, 256, 3).parameters()])




