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
        
        actions_mask = np.ones((self.batch_size, self.actionspace_size))

        q_values = np.empty(self.batch_size)
        v_target = self.target_net.predict(next_states)
        v_targets = np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            if terminals[i]:
                q_values[i] = rewards[i]
                v_targets[i] = rewards[i]
            else:
                q_values[i] = rewards[i] + self.discount_factor * v_target[i]
                v_targets[i] = rewards[i] + self.discount_factor * v_target[i]

        one_hot_actions = np.eye(self.actionspace_size)[np.array(actions).reshape(-1)]
        self.q_net.fit([prev_states, one_hot_actions], one_hot_actions * q_values[:, None], batch_size=self.batch_size, epochs=1, verbose=0)
        self.v_net.fit(prev_states, v_targets, batch_size=self.batch_size, epochs=1, verbose=0)

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

def writeLog(path, epoch, accumulated_epoch_reward, epsilon, updates):
    if not os.path.exists('./Data'):
        os.makedirs('./Data')
    with open(path, 'a') as f:
        csv_writer = csv.writer(f, delimiter=';')
        csv_writer.writerow([epoch, accumulated_epoch_reward, epsilon, updates])

def main():
    print("Hello World")

    environment = gym.make(ENVIRONMENT_ID)

    epoch = 0
    total_updates = 100000
    update_target_step = 10000
    update_counter = 0
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

    q_net = Network('q', actionspace_size, learning_rate, gradient_momentum, gradient_min).model
    v_net = Network('v', actionspace_size, learning_rate, gradient_momentum, gradient_min).model
    target_net = copy.deepcopy(v_net)

    agent = Agent(environment, q_net, v_net, target_net, memory, batch_size, discount_factor, actionspace_size, epsilon, epsilon_decay, epsilon_min)

    start_time_str = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
    end_time = time.time() + 250000

    while update_counter < total_updates:
        environment.reset()

        observation, reward, done, info = environment.step(1)

        # Init state based on first frame
        frame = getPreprocessedFrame(observation)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1, 84, 84, 4))

        accumulated_epoch_reward = 0

        while not done and update_counter < total_updates and time.time() < end_time:
            update_counter += 1
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
            if update_counter > 5000:
                agent.train()

            # Potentially update target net
            if update_counter % update_target_step == 0:
                agent.updateTargetNet()
            
            accumulated_epoch_reward += reward
        
        # Produce output
        print(epoch, ';', accumulated_epoch_reward, ';', agent.epsilon, update_counter)
        writeLog('./Data/' + start_time_str + '_log.csv', epoch, accumulated_epoch_reward, agent.epsilon, update_counter)

        if time.time() > end_time:
            print('timeout')
            break

if __name__ == "__main__":
    main()
