

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


class ExperienceReplay():
    def __init__(self, max_memory, discount):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):   # remember what we've done
        self.memory.append([states, game_over])

        if len(self.memory) > self.max_memory:  # don't store too much
            del self.memory[0]  # delete oldest one

    def get_batch(self, model, batch_size):  # return a batch of inputs to NN and target outputs, and train NN using sgd
        len_memory = len(self.memory)

        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))  # inputs to NN
        targets = np.zeros((inputs.shape[0], num_actions))

        # pick random number of memories
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            targets[i] = model.predict(state_t)[0] # set to prediction
            Q_sa = np.max(model.predict(state_tp1)[0])   # highest value that NN predicts for next

            if game_over:
                targets[i, action_t] = reward_t   # if game_over, we already know what reward is, set target result for move we took exactly
            else:
                # have to consider max reward in future, with temporal discounting
                targets[i, action_t] = reward_t + self.discount * Q_sa   # reward

        return inputs, targets


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
    MAX_MEMORY = 500
    DISCOUNT = 0.9

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

    replay = ExperienceReplay(MAX_MEMORY, DISCOUNT)

    win_cnt = 0
    for epoch in range(EPOCHS):
        loss = 0
        env.reset()
        game_over = False
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t

            if random.random() <= EPSILON:  # randomly explore
                action = random.randint(0, 2)
            else:
                q = model.predict(input_tm1)  # feed in input, get network's prediction, and take that action
                action = np.argmax(q[0])

            input_t, reward, game_over = env.act(action)

            if reward == 1:
                win_cnt += 1

            replay.remember([input_tm1, action, reward, input_t], game_over)
            inputs, targets = replay.get_batch(model, BATCH_SIZE)    # get mini batch
            # now we get our loss from our model
            loss += model.train_on_batch(inputs, targets)

        print("Epoch: {:06d}/{:06d} | Loss {:.4f} | win count {}".format(epoch, EPOCHS, loss,  win_cnt))

        # save model weights
        model.save_weights("trained.h5", overwrite=True)

        # store network structure
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)