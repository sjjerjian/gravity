

import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import json

# create game
class Gravity:
    def __init__(self, size):  # self references object itself, size is an attribute passed to the object
        self.size = size
        block_y = 0                                 # block starts at y = 0
        block_x = random.randint(0, self.size - 1)   # initial position of block - random x position
        basket_x = random.randint(1, self.size - 2)  # basket y is fixed at bottom, size - 2 because basket has length 3
        self.state = [block_y, block_x, basket_x]

    def observe(self):  # take current state of game, make 10x10 grid of zeros and put 1s where block and basket are
        canvas = [0] * self.size**2
        canvas[self.state[0] * self.size + self.state[1]] = 1        #
        canvas[(self.size - 1) * self.size + self.state[2] - 1] = 1  # basket exists, and find position 1 of basket
        canvas[(self.size - 1) * self.size + self.state[2] + 0] = 1  # basket exists, and find position 2 of basket
        canvas[(self.size - 1) * self.size + self.state[2] + 1] = 1  # basket exists, and find position 3 of basket

        return np.array(canvas).reshape(1, -1)  # turn canvas into vector, reshape into form keras can understand

    def act(self, action):  # take action, move block down by 1 and move basket
        block_y, block_x, basket_x = self.state

        # action takes value of 0, 1, or 2, based on whether we move left, stay or move right
        basket_x += int(action) - 1
        basket_x = max(1, basket_x)
        basket_x = min(self.size - 2, basket_x)  # ensure basket doesn't go off end of grid, wrap around

        block_y += 1    # drop block by 1 space

        self.state = [block_y, block_x, basket_x]

        reward = 0
        if block_y == self.size - 1:
            if abs(block_x - basket_x) <= 1:
                reward = 1  # we catch the block!
            else:
                reward = - 1   # we missed the block :(

        game_over = block_y == self.size - 1  # check if game over

        return self.observe(), reward, game_over


    def reset(self):   # if game finishes, reset
        self.__init__(self.size)

if __name__ == '__main__':
    # define some important constants (hyperparameters)

    GRID_DIM = 10

    EPSILON = 0.1   # explore or exploit (explore is new, exploit is based on previous knowledge)
    # epsilon is  P(network will randomly explore)

    LEARNING_RATE = 0.2     # rate at which neural network will take information as granted
    # increasing learning rate increases risk of overfitting

    # sum((actual - expected)**2)
    LOSS_FUNCTION = 'mse'   # loss function - mean squared error

    HIDDEN_LAYER1_SIZE = 100
    HIDDEN_LAYER1_ACTIVATION = "relu"
    # sum of all previous inputs multiplied by weights, then feed it forward. relu is activation function

    HIDDEN_LAYER2_SIZE = 100
    HIDDEN_LAYER2_ACTIVATION = "relu"

    BATCH_SIZE = 50     # 50 examples at a time
    EPOCHS = 1000       # extent of training, number of iterations

    model = Sequential()    # feedforward

    # layer 1
    model.add(
        Dense(
            HIDDEN_LAYER1_SIZE,
            input_shape=(GRID_DIM**2,),
            activation=HIDDEN_LAYER1_ACTIVATION
        )
    )

    # layer 2
    model.add(
        Dense(
            HIDDEN_LAYER2_SIZE,
            activation=HIDDEN_LAYER1_ACTIVATION
        )
    )

    # output layer
    model.add(Dense(3))

    model.compile(sgd(lr=LEARNING_RATE), LOSS_FUNCTION)

    # set up environment
    env = Gravity(GRID_DIM)

    win_cnt = 0
    for epoch in range(EPOCHS):
        env.reset()
        game_over = False
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t

            if random.random() <= EPSILON:  # randomly explore
                action = random.randint(0, 2)
            else:
                # take what model thinks
                q = model.predict(input_tm1)  # 3 probabilities for confidence for each move, choose most confident one
                action = np.argmax(q[0])

            input_t, reward, game_over = env.act(action)

            if reward == 1:
                win_cnt += 1

        print("Epoch: {:06d}/{:06d} | win count {}".format(epoch, EPOCHS, win_cnt))

        # save model weights
        model.save_weights("model.h5", overwrite=True)

        # store network structure
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)

# performance is at chance level at the moment as the network is not actually being trained! need to use reward to train it