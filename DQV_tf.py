import random
import copy
import os
import sys
import csv
import time
import pickle
from collections import deque

import numpy as np
import skimage

import gym
import keras

ENVIRONMENT_ID = 'Breakout-v0'

class Network:
    def __init__(self, nettype, actionspace_size, learning_rate, gradient_momentum, gradient_min):
        frames_input = keras.layers.Input((84, 84, 4))
        actions_input = keras.layers.Input((actionspace_size,))

        conv1 = keras.layers.Conv2D(16, (8, 8), strides=(4, 4), activation="relu")(frames_input)
        conv2 = keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation="relu")(conv1)

        flattened = keras.layers.Flatten()(conv2)

        hidden = keras.layers.Dense(256, activation="relu")(flattened)

        if nettype == 'q':
            output = keras.layers.Dense(actionspace_size)(hidden)
            filtered_output = keras.layers.merge.Multiply()([output, actions_input])
            self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        if nettype == 'v':
            output = keras.layers.Dense(1)(hidden)
            self.model = keras.models.Model(inputs=frames_input, outputs=output)

        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr=learning_rate, rho=gradient_momentum, epsilon=gradient_min))

class Agent:
    def __init__(self, environment, q_net, v_net, target_net, memory, batch_size, discount_factor, actionspace_size, epsilon, epsilon_decay, epsilon_min):
        self.environment = environment
        self.q_net = q_net
        self.v_net = v_net
        self.target_net = target_net
        self.memory = memory
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.actionspace_size = actionspace_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.weight_updates = 0

    def train(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        prev_states = np.empty((self.batch_size, 84, 84, 4), dtype=np.float32)
        next_states = np.empty((self.batch_size, 84, 84, 4), dtype=np.float32)
        actions = np.empty(self.batch_size, dtype=np.int32)
        rewards = np.empty(self.batch_size, dtype=np.float32)
        terminals = np.empty(self.batch_size, dtype=np.bool)

        for i in range(self.batch_size):
            prev_states[i] = np.float32(mini_batch[i][0] / 255.0)
            next_states[i] = np.float32(mini_batch[i][3] / 255.0)
            actions[i] = mini_batch[i][1]
            rewards[i] = mini_batch[i][2]
            terminals[i] = mini_batch[i][4]

        q_values = np.empty(self.batch_size)
        v_values = np.zeros((self.batch_size,))
        v_targets = self.target_net.predict(next_states)

        for i in range(self.batch_size):
            if terminals[i]:
                q_values[i] = rewards[i]
                v_values[i] = rewards[i]
            else:
                q_values[i] = rewards[i] + self.discount_factor * v_targets[i]
                v_values[i] = rewards[i] + self.discount_factor * v_targets[i]

        one_hot_actions = np.eye(self.actionspace_size)[np.array(actions).reshape(-1)]
        self.q_net.fit([prev_states, one_hot_actions], one_hot_actions * q_values[:, None], batch_size=self.batch_size, epochs=1, verbose=0)
        self.v_net.fit(prev_states, v_values, batch_size=self.batch_size, epochs=1, verbose=0)
        self.weight_updates += 1

        self.updateEpsilon()

    def chooseAction(self, state):
        state = np.float32(state / 255.0)
        if random.random() < self.epsilon:
            action = random.randrange(self.actionspace_size)
        else:
            actions_mask = np.ones(self.actionspace_size).reshape(1, self.actionspace_size)
            q_values = self.q_net.predict([state, actions_mask])
            action = np.argmax(q_values)

        return action

    def updateEpsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def storeExperience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def updateTargetNet(self):
        self.target_net.set_weights(self.v_net.get_weights())

def getPreprocessedFrame(observation):
    observation = skimage.color.rgb2gray(observation)
    observation = skimage.transform.resize(observation, (84, 84))
    observation = np.uint8(observation * 255)
    return observation

def writeLog(path, epoch, accumulated_epoch_reward, epsilon, weight_updates):
    if not os.path.exists('./Data'):
        os.makedirs('./Data')
    with open(path, 'a') as f:
        csv_writer = csv.writer(f, delimiter=';')
        csv_writer.writerow([epoch, accumulated_epoch_reward, epsilon, weight_updates])

def saveModel(path, model):
    if not os.path.exists('./Data'):
        os.makedirs('./Data')
    model.save(path)

def saveAgent(path, agent):
    if not os.path.exists('./Data'):
        os.makedirs('./Data/')
    with open(path, 'wb') as saved_object:
        pickle.dump(agent, saved_object, pickle.HIGHEST_PROTOCOL)

def loadAgent(path):
    with open(path, 'rb') as saved_object:
        return pickle.load(saved_object)

def main():
    print("Hello World\nThis is DQV " + ENVIRONMENT_ID)
    path = './Data/' + time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime()) + '_DQV_' + ENVIRONMENT_ID
    print("session will be stored at " + path)

    environment = gym.make(ENVIRONMENT_ID)

    continue_trainig = True if len(sys.argv) == 2 else False

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

    if (continue_trainig):
        print('loading previously saved agent')
        agent = loadAgent('./Data/' + sys.argv[1])
    else:
        print("creating new agent")
        q_net = Network('q', actionspace_size, learning_rate, gradient_momentum, gradient_min).model
        v_net = Network('v', actionspace_size, learning_rate, gradient_momentum, gradient_min).model
        target_net = copy.deepcopy(v_net)
        memory = deque(maxlen=1000000)
        agent = Agent(environment, q_net, v_net, target_net, memory, batch_size, discount_factor, actionspace_size, epsilon, epsilon_decay, epsilon_min)

    step_number = 0
    end_time = time.time() + 250000

    for epoch in range(epochs):
        environment.reset()

        for _ in range(random.randint(1, 25)):
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
            accumulated_epoch_reward += reward
            reward = np.clip(reward, -1., 1.)
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
            if step_number > 50000:
                agent.train()

            # Potentially update target net
            if step_number % update_target_step == 0:
                agent.updateTargetNet()

        # Produce output
        print(epoch, ';', accumulated_epoch_reward, ';', agent.epsilon, ';', agent.weight_updates)
        writeLog(path + '_log.csv', epoch, accumulated_epoch_reward, agent.epsilon, agent.weight_updates)

        if time.time() > end_time:
            print('timeout')
            break

    # Save model and agent
    saveModel(path + '_q-model.h5', agent.q_net)
    saveModel(path + '_v-model.h5', agent.v_net)
    saveAgent(path + '_agent.pkl', agent)

if __name__ == "__main__":
    main()
