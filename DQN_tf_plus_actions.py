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
    def __init__(self, actionspace_size, learning_rate, gradient_momentum, gradient_min):
        frames_input = keras.layers.Input((84, 84, 4))
        actions_input = keras.layers.Input((actionspace_size,))
        last_actions_input = keras.layers.Input((1,3))
        
        conv1 = keras.layers.Conv2D(16, (8, 8), strides=(4, 4), activation="relu")(frames_input)
        conv2 = keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation="relu")(conv1)

        flattened = keras.layers.Flatten()(conv2)
        
        # Concatenate flattened conv-output and actions
        flattened_actions = keras.layers.Flatten()(last_actions_input)
        merged = keras.layers.Concatenate(axis=1)([flattened,flattened_actions])

        hidden = keras.layers.Dense(256+3, activation="relu")(merged)
        output = keras.layers.Dense(actionspace_size)(hidden)

        filtered_output = keras.layers.merge.Multiply()([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input, last_actions_input], outputs=filtered_output)

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
        prev_states = np.empty((self.batch_size, 84, 84, 4), dtype=np.float32)
        next_states = np.empty((self.batch_size, 84, 84, 4), dtype=np.float32)
        actions = np.empty(self.batch_size, dtype=np.int32)
        rewards = np.empty(self.batch_size, dtype=np.float32)
        terminals = np.empty(self.batch_size, dtype=np.bool)
        prev_state_actions = np.empty((self.batch_size, 1, 3), dtype=np.float32)
        next_state_actions = np.empty((self.batch_size, 1, 3), dtype=np.float32)

        for i in range(self.batch_size):
            prev_states[i] = np.float32(mini_batch[i][0] / 255.0)
            next_states[i] = np.float32(mini_batch[i][3] / 255.0)
            actions[i] = mini_batch[i][1]
            rewards[i] = mini_batch[i][2]
            terminals[i] = mini_batch[i][4]
            prev_state_actions[i] = mini_batch[i][5]
            next_state_actions[i] = mini_batch[i][6]

        actions_mask = np.ones((self.batch_size, self.actionspace_size))

        q_values = np.empty(self.batch_size)
        q_targets = self.target_net.predict([next_states, actions_mask, next_state_actions])

        for i in range(self.batch_size):
            if terminals[i]:
                q_values[i] = rewards[i]
            else:
                q_values[i] = rewards[i] + self.discount_factor * np.max(q_targets[i])

        one_hot_actions = np.eye(self.actionspace_size)[np.array(actions).reshape(-1)]
        self.q_net.fit([prev_states, one_hot_actions, prev_state_actions], one_hot_actions * q_values[:, None], batch_size=self.batch_size, epochs=1, verbose=0)

        self.updateEpsilon()

    def saveModel(self, file_name):
        if not os.path.exists('./Data'):
            os.makedirs('./Data')
        
        # Save entire model to a HDF5 file
        path = './Data/' + file_name + '.h5'
        self.q_net.save(path)
        print('Saved model as: ', path)


    def chooseAction(self, state, last_actions):
        state = np.float32(state / 255.0)
        if random.random()  < self.epsilon:
            action = random.randrange(self.actionspace_size)
        else:
            actions_mask = np.ones(self.actionspace_size).reshape(1, self.actionspace_size)
            q_values = self.q_net.predict([state, actions_mask, last_actions.reshape(1,1,3)])
            action = np.argmax(q_values)

        return action

    def updateEpsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def storeExperience(self, state, action, reward, next_state, terminal, prev_state_actions, next_state_actions):
        self.memory.append((state, action, reward, next_state, terminal, prev_state_actions, next_state_actions))

    def updateTargetNet(self):
        self.target_net.set_weights(self.q_net.get_weights())

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
    total_updates = 1000000
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
    save_checkpoint = 50000

    memory = deque(maxlen=1000000)

    q_net = Network(actionspace_size, learning_rate, gradient_momentum, gradient_min).model
    target_net = copy.deepcopy(q_net)

    agent = Agent(environment, q_net, target_net, memory, batch_size, discount_factor, actionspace_size, epsilon, epsilon_decay, epsilon_min)

    step_number = 0
    start_time_str = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
    end_time = time.time() + 250000

    while update_counter < total_updates:
        environment.reset()

        observation, reward, done, info = environment.step(1)

        # Init state based on first frame
        frame = getPreprocessedFrame(observation)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1, 84, 84, 4))
        prev_state_actions = np.zeros(3)
        next_state_actions = np.zeros(3)
        
        accumulated_epoch_reward = 0

        while not done and update_counter < total_updates and time.time() < end_time:
            update_counter += 1
            terminal = False
            lives = info['ale.lives']

            # Choose and perform action and check if life lost
            action = agent.chooseAction(state, next_state_actions)
            observation, reward, done, info = environment.step(action)
            if lives > info['ale.lives'] or done:
                terminal = True

            # Update state based on new frame
            prev_state = state
            frame = getPreprocessedFrame(observation)
            frame = np.reshape([frame], (1, 84, 84, 1))
            state = np.append(frame, state[:, :, :, :3], axis=3)
            
            # Update actions
            prev_state_actions = next_state_actions
            next_state_actions = next_state_actions[:2]
            next_state_actions = np.append([action], next_state_actions, axis=0)
            
            # Store state in memory
            agent.storeExperience(prev_state, action, reward, state, terminal, prev_state_actions, next_state_actions)

            # Train agent
            if update_counter > 5000:
                agent.train()

            # Potentially update target net
            if update_counter % update_target_step == 0:
                agent.updateTargetNet()
            
            # Save checkpoints
            if update_counter % save_checkpoint == 0:
                agent.saveModel(start_time_str + '_Model_' + str(update_counter) + '_')

            accumulated_epoch_reward += reward
        
        # Produce output
        print(epoch, ';', accumulated_epoch_reward, ';', agent.epsilon, update_counter)
        writeLog('./Data/' + start_time_str + '_log.csv', epoch, accumulated_epoch_reward, agent.epsilon, update_counter)

        if time.time() > end_time:
            print('timeout')
            break
            
    agent.saveModel(start_time_str + '_Model')

if __name__ == "__main__":
    main()
