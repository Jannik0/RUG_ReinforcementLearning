import random
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
import tensorflow as tf

SOFTMAX_CURRENT = np.zeros(32, dtype=np.float32)
SOFTMAX_PAST = np.zeros(32, dtype=np.float32)
FACTOR = 1

class Network:
    def __init__(self, actionspace_size, learning_rate, gradient_momentum, gradient_min):
        frames_input = keras.layers.Input((84, 84, 4))
        actions_input = keras.layers.Input((actionspace_size,))

        conv1 = keras.layers.Conv2D(16, (8, 8), strides=(4, 4), activation="relu")(frames_input)
        conv2 = keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation="relu")(conv1)

        flattened = keras.layers.Flatten()(conv2)

        hidden = keras.layers.Dense(256, activation="relu")(flattened)
        output = keras.layers.Dense(actionspace_size)(hidden)

        filtered_output = keras.layers.merge.Multiply()([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        self.model.compile(loss=self.customLoss(), optimizer=keras.optimizers.RMSprop(lr=learning_rate, rho=gradient_momentum, epsilon=gradient_min))

    def customLoss(self):
        def mse(y_true, y_pred):
            return keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1)

        def kld():
            p = keras.backend.clip(tf.convert_to_tensor(SOFTMAX_CURRENT), keras.backend.epsilon(), 1)
            q = keras.backend.clip(tf.convert_to_tensor(SOFTMAX_PAST), keras.backend.epsilon(), 1)
            return keras.backend.sum(p * keras.backend.log(p / q), axis=-1)

        def loss(y_true, y_pred):
            divergence = kld()
            global FACTOR
            if FACTOR > 0.1:
                FACTOR -= 1e-06
            return mse(y_true, y_pred) - FACTOR * divergence

        return loss

class Agent:
    def __init__(self, q_net, target_net, memory, batch_size, discount_factor, actionspace_size):
        self.q_net = q_net
        self.target_net = target_net
        self.memory = memory
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.actionspace_size = actionspace_size
        self.weight_updates = 0

    def train(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        prev_states = np.empty((self.batch_size, 84, 84, 4), dtype=np.float32)
        next_states = np.empty((self.batch_size, 84, 84, 4), dtype=np.float32)
        actions = np.empty(self.batch_size, dtype=np.int32)
        rewards = np.empty(self.batch_size, dtype=np.float32)
        terminals = np.empty(self.batch_size, dtype=np.bool)
        past_qs = np.empty((self.batch_size, self.actionspace_size), dtype=np.float32)

        for i in range(self.batch_size):
            prev_states[i] = np.float32(mini_batch[i][0] / 255.0)
            next_states[i] = np.float32(mini_batch[i][3] / 255.0)
            actions[i] = mini_batch[i][1]
            rewards[i] = mini_batch[i][2]
            terminals[i] = mini_batch[i][4]
            past_qs[i] = mini_batch[i][5]

        actions_mask = np.ones((self.batch_size, self.actionspace_size))

        q_values = np.empty(self.batch_size)
        q_targets = self.target_net.predict([next_states, actions_mask])

        for i in range(self.batch_size):
            if terminals[i]:
                q_values[i] = rewards[i]
            else:
                q_values[i] = rewards[i] + self.discount_factor * np.max(q_targets[i])

            SOFTMAX_CURRENT[i] = np.exp(q_targets[i][actions[i]]) / np.sum(np.exp(q_targets[i]), axis=0)
            SOFTMAX_PAST[i] = np.exp(past_qs[i][actions[i]]) / np.sum(np.exp(past_qs[i]), axis=0)

        one_hot_actions = np.eye(self.actionspace_size)[np.array(actions).reshape(-1)]
        self.q_net.fit([prev_states, one_hot_actions], one_hot_actions * q_values[:, None], batch_size=self.batch_size, epochs=1, verbose=0)
        self.weight_updates += 1

    def play(self, environment, evaluation_games, output_path):
        average_reward = 0.0

        for _ in range(evaluation_games):
            environment.reset()

            observation, reward, done, _ = environment.step(1)

            frame = getPreprocessedFrame(observation)
            state = np.stack((frame, frame, frame, frame), axis=2)
            state = np.reshape([state], (1, 84, 84, 4))

            accumulated_epoch_reward = 0

            while not done:
                action = self.chooseAction(state)
                observation, reward, done, _ = environment.step(action)
                accumulated_epoch_reward += reward

                frame = getPreprocessedFrame(observation)
                frame = np.reshape([frame], (1, 84, 84, 1))
                state = np.append(frame, state[:, :, :, :3], axis=3)

            average_reward += accumulated_epoch_reward / evaluation_games

        writeLog(output_path + 'eval.csv', [self.weight_updates, average_reward])

    def chooseAction(self, state):
        state = np.float32(state / 255.0)
        actions_mask = np.ones(self.actionspace_size).reshape(1, self.actionspace_size)
        q_values = self.q_net.predict([state, actions_mask])
        action = np.argmax(q_values)
        return action
    
    def chooseRandomAction(self):
        action = random.randrange(self.actionspace_size)
        return action

    def storeExperience(self, state, action, reward, next_state, terminal, past_q):
        self.memory.append((state, action, reward, next_state, terminal, past_q))

    def updateTargetNet(self):
        self.target_net.set_weights(self.q_net.get_weights())

def getPreprocessedFrame(observation):
    observation = skimage.color.rgb2gray(observation)
    observation = skimage.transform.resize(observation, (84, 84))
    observation = np.uint8(observation * 255)
    return observation

def writeLog(path, content):
    with open(path, 'a') as log:
        csv_writer = csv.writer(log, delimiter=';')
        csv_writer.writerow(content)

def saveModel(path, model):
    model.save(path)

def loadModel(path):
    return keras.models.load_model(path)

def saveAgent(path, agent):
    with open(path, 'wb') as saved_object:
        pickle.dump(agent, saved_object, pickle.HIGHEST_PROTOCOL)

def loadAgent(path):
    with open(path, 'rb') as saved_object:
        return pickle.load(saved_object)

def main():
    if len(sys.argv) != 3:
        print("Please provide a valid environment and session ID")
        return

    environment_id = sys.argv[1]
    session_id = sys.argv[2]
    path = './Data/Div-DQN/' + environment_id + '/' + session_id + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    print("This is Div-DQN " + environment_id + " " + session_id + "\nSession will be stored at " + path)

    environment = gym.make(environment_id)
    eval_environment = gym.make(environment_id)

    training_start = 50000
    update_target_step = 10000
    evaluation_step = 10000
    evaluation_games = 20
    training_stop = 10000000

    actionspace_size = environment.action_space.n
    batch_size = 32
    discount_factor = 0.99
    learning_rate = 0.00025
    gradient_momentum = 0.95
    gradient_min = 0.01

    # Load previous session or create new one
    if os.path.isfile(path + 'agent.pkl'):
        print("agent found; loading previous agent")
        agent = loadAgent(path + 'agent.pkl')
        agent.q_net = loadModel(path + 'qmodel.h5')
        agent.target_net = loadModel(path + 'targetmodel.h5')
        step_number = agent.weight_updates + training_start
    else:
        if os.path.isfile(path + 'log.csv'):
            print("incomplete session found; aborting")
            return
        print("no agent found; creating new agent")
        q_net = Network(actionspace_size, learning_rate, gradient_momentum, gradient_min).model
        target_net = Network(actionspace_size, learning_rate, gradient_momentum, gradient_min).model
        memory = deque(maxlen=1000000)
        agent = Agent(q_net, target_net, memory, batch_size, discount_factor, actionspace_size)
        step_number = 0

    end_time = time.time() + 860000

    print("starting")

    # Main loop
    while agent.weight_updates < training_stop and time.time() < end_time:
        environment.reset()

        for _ in range(random.randint(1, 25)):
            observation, reward, done, info = environment.step(1)

        # Init state based on first frame
        frame = getPreprocessedFrame(observation)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1, 84, 84, 4))

        accumulated_epoch_reward = 0

        while not done and agent.weight_updates < training_stop and time.time() < end_time:
            step_number += 1
            terminal = False
            lives = info['ale.lives']

            # Choose and perform action and check if life lost
            if step_number > training_start:
                action = agent.chooseAction(state)
            else:
                action = agent.chooseRandomAction()
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
            actions_mask = np.ones(agent.actionspace_size).reshape(1, agent.actionspace_size)
            q_values = agent.q_net.predict([state, actions_mask])
            agent.storeExperience(prev_state, action, reward, state, terminal, q_values)

            # Train agent
            if step_number > training_start:
                agent.train()

                # Potentially update target net
                if step_number % update_target_step == 0:
                    agent.updateTargetNet()

                # Evaluation games
                if step_number % evaluation_step == 0:
                    agent.play(eval_environment, evaluation_games, path)

        # Produce output
        writeLog(path + 'log.csv', [agent.weight_updates, accumulated_epoch_reward, FACTOR])

    # Save models and agent
    saveModel(path + 'qmodel.h5', agent.q_net)
    #saveModel(path + 'targetmodel.h5', agent.target_net)
    #agent.q_net = agent.target_net = None
    #saveAgent(path + 'agent.pkl', agent)

if __name__ == "__main__":
    main()
