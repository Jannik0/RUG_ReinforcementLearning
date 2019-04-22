import random
import copy
import os
import csv
import time
from collections import deque

import numpy as np
import skimage

import gym
import keras

ENVIRONMENT_ID = 'Breakout-v0'

class Network:
    def __init__(self, output_size, learning_rate, gradient_momentum, gradient_min):
        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Conv2D(16, (8, 8), strides=(4, 4), activation="relu", input_shape=(84, 84, 4)))
        self.model.add(keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation="relu"))

        self.model.add(keras.layers.Flatten())

        self.model.add(keras.layers.Dense(256, activation="relu"))
        self.model.add(keras.layers.Dense(output_size))

        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr=learning_rate, rho=gradient_momentum, epsilon=gradient_min))

class Agent:
    def __init__(self, environment, q_net, target_net, memory, batch_size, discount_factor, actionspace_size, epsilon, epsilon_decay, epsilon_min):
        self.environment = environment
        self.q_net = q_net
        self.target_net = target_net
        self.memory = memory
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.actionspace_size = actionspace_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def train(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        prev_states = np.empty((self.batch_size, 84, 84, 4))
        next_states = np.empty((self.batch_size, 84, 84, 4))

        for i in range(self.batch_size):
            prev_states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]

        q_values = self.q_net.predict(prev_states)
        q_targets = self.target_net.predict(next_states)

        for i in range(self.batch_size):
            if mini_batch[i][4]:
                q_values[i][mini_batch[i][1]] = mini_batch[i][2]
            else:
                q_values[i][mini_batch[i][1]] = mini_batch[i][2] + self.discount_factor * np.max(q_targets[:][mini_batch[i][1]])
                #q_values[i][mini_batch[i][1]] = mini_batch[i][2] + self.discount_factor * np.max(q_targets[i])
                #q_values[i][mini_batch[i][1]] = mini_batch[i][2] + self.discount_factor * q_targets[i][mini_batch[i][1]]

        self.q_net.fit(prev_states, q_values, epochs=1, verbose=0)

        self.updateEpsilon()

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.actionspace_size)
        else:
            q_values = self.q_net.predict(state, batch_size=self.batch_size)
            action = np.argmax(q_values[0])

        return action

    def updateEpsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def storeExperience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def updateTargetNet(self):
        self.target_net.set_weights(self.q_net.get_weights())

def getPreprocessedFrame(observation):
    observation = skimage.color.rgb2gray(observation)
    observation = skimage.transform.resize(observation, (84, 84))
    return observation

def writeLog(path, epoch, accumulated_epoch_reward, epsilon):
    if not os.path.exists('./Data'):
        os.makedirs('./Data')
    with open(path, 'a') as f:
        csv_writer = csv.writer(f, delimiter=';')
        csv_writer.writerow([epoch, accumulated_epoch_reward, epsilon])

def main():
    print("Hello World")

    environment = gym.make(ENVIRONMENT_ID)

    epochs = 100000
    update_target_step = 10000

    actionspace_size = environment.action_space.n
    batch_size = 32
    discount_factor = 0.99
    learning_rate = 0.00025
    gradient_momentum = 0.95
    gradient_min = 0.01
    epsilon = 1
    epsilon_decay = 1e-06
    epsilon_min = 0.1

    memory = deque(maxlen=1000000)

    q_net = Network(actionspace_size, learning_rate, gradient_momentum, gradient_min).model
    target_net = copy.deepcopy(q_net)

    agent = Agent(environment, q_net, target_net, memory, batch_size, discount_factor, actionspace_size, epsilon, epsilon_decay, epsilon_min)

    step_number = 0
    start_time_str = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
    end_time = time.time() + 250000

    for epoch in range(epochs):
        environment.reset()

        observation, reward, done, info = environment.step(1)

        # Init state based on first frame
        frame = getPreprocessedFrame(observation)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1, 84, 84, 4))

        accumulated_epoch_reward = 0

        while not done:
            step_number += 1
            terminal = False
            lives = info['ale.lives']

            # Choose and perform action and check if life lost
            action = agent.chooseAction(state)
            observation, reward, done, info = environment.step(action)
            if lives > info['ale.lives'] or done:
                terminal = True

            # Update state based on new frame
            prev_state = state
            frame = getPreprocessedFrame(observation)
            frame = np.reshape([frame], (1, 84, 84, 1))
            state = np.append(frame, state[:, :, :, :3], axis=3)

            # Store state in memory
            agent.storeExperience(prev_state, action, reward, state, terminal)

            # Train agent
            if step_number > 5000:
                agent.train()

            # Potentially update target net
            if step_number % update_target_step == 0:
                agent.updateTargetNet()
            
            accumulated_epoch_reward += reward
        
        # Produce output
        print(epoch, ';', accumulated_epoch_reward, ';', agent.epsilon)
        writeLog('./Data/' + start_time_str + '_log.csv', epoch, accumulated_epoch_reward, agent.epsilon)

        if time.time() > end_time:
            print('timeout')
            break

if __name__ == "__main__":
    main()
