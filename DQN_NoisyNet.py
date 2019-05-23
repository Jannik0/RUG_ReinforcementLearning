import random
import copy
import os
import sys
import csv
import time
import pickle
from collections import deque
import tensorflow as tf
from keras.utils import CustomObjectScope

import numpy as np
import skimage

import gym
import keras

class NoisyLayer(keras.layers.Layer):

    def __init__(self, in_shape=(1,2592), out_dim=256, activation='tf.identity', name='Layer', **kwargs):
        
        # Parameter assignments
        self.in_shape = in_shape
        self.out_units = out_dim
        self.activation = eval(activation)
        self.activation_str = activation 
        self.name = name
        
        # Derived assignments
        self.p = float(self.in_shape[1])
        self.mu_interval_value = 1.0/np.sqrt(self.p)
        self.sig_0 = 0.5
        self.sig_init_constant = self.sig_0/np.sqrt(self.p)
        
        # Naming weights/biases
        self.w_mu_name = self.name+'w_mu'
        self.w_si_name = self.name+'w_si'
        self.b_mu_name = self.name+'b_mu'
        self.b_si_name = self.name+'b_si'
        
        super(NoisyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initializers
        self.mu_initializer = tf.initializers.random_uniform(minval=-self.mu_interval_value, maxval=self.mu_interval_value) # Mu-initializer
        self.si_initializer = tf.initializers.constant(self.sig_init_constant)                                              # Sigma-initializer
        
        # Weights
        # 'Normal' weights
        self.w_mu = self.add_weight(  name=self.w_mu_name, 
                                      shape=(self.in_shape[1], self.out_units),
                                      initializer=self.mu_initializer,
                                      trainable=True)
        # 'Noisy' weights                              
        self.w_si = self.add_weight(  name=self.w_si_name, 
                                      shape=(self.in_shape[1], self.out_units),
                                      initializer=self.si_initializer,
                                      trainable=True)
        
        # Biases
        # 'Normal' biases
        self.b_mu = self.add_weight(  name=self.b_mu_name, 
                                      shape=(self.in_shape[0], self.out_units),
                                      initializer=self.mu_initializer,
                                      trainable=True)
        # 'Noisy' biases
        self.b_si = self.add_weight(  name=self.b_si_name, 
                                      shape=(self.in_shape[0], self.out_units),
                                      initializer=self.si_initializer,
                                      trainable=True)
                                   
        # Make sure this function is going to be called on init                              
        super(NoisyLayer, self).build(input_shape)

    def call(self, inputs):
        
        # Resample noise - once per input-batch
        self.assign_resampling()

        # Putting it all together
        self.w = tf.math.add(self.w_mu, tf.math.multiply(self.w_si, self.w_eps))
        self.b = tf.math.add(self.b_mu, tf.math.multiply(self.b_si, self.q_eps))

        return self.activation(tf.math.add(tf.linalg.matmul(inputs, self.w), self.b))
    
    # Functionality supporting deepcopying, saving, and loading
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_units)
        
    def get_config(self):
        config = super(NoisyLayer, self).get_config()
        config['in_shape'] = self.in_shape
        config['out_dim'] = self.out_units
        config['activation'] = self.activation_str
        config['name'] = self.name

        return config
    
    # Noise sampling - Factorised Gaussian noise
    
    def assign_resampling(self):
        # p = related to (i) inputs; q = related to (j) outputs
        self.p_eps = self.f(self.resample_noise([self.in_shape[1], 1]))     #         = f(eps_i) in paper
        self.q_eps = self.f(self.resample_noise([1, self.out_units]))       # = eps_b = f(eps_j) in paper; Eqn. 11
        self.w_eps = self.p_eps * self.q_eps                                # Cartesian product of input_noise x output_noise; Eqn. 10

    def resample_noise(self, shape):
        return tf.random.normal(shape, mean=0.0, stddev=1.0, seed=None, name=None)

    def f(self, x):
        return tf.math.multiply(tf.math.sign(x), tf.math.sqrt(tf.math.abs(x)))
    


class Network:
    def __init__(self, actionspace_size, learning_rate, gradient_momentum, gradient_min, noisy_flag):
        frames_input = keras.layers.Input((84, 84, 4))
        actions_input = keras.layers.Input((actionspace_size,))

        conv1 = keras.layers.Conv2D(16, (8, 8), strides=(4, 4), activation="relu")(frames_input)
        conv2 = keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation="relu")(conv1)

        flattened = keras.layers.Flatten()(conv2)
        
        if noisy_flag:
            # NoisyNet
            hidden = NoisyLayer(in_shape=(1,2592), out_dim=256, activation="tf.nn.relu", name='Noisy1')(flattened)
            output = NoisyLayer(in_shape=(1,256), out_dim=actionspace_size, activation="tf.identity", name='Noisy2')(hidden) # add diff actv-fnctn
        else:
            # Standard DQN
            hidden = keras.layers.Dense(256, activation="relu")(flattened)
            output = keras.layers.Dense(actionspace_size)(hidden)

        filtered_output = keras.layers.merge.Multiply()([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr=learning_rate, rho=gradient_momentum, epsilon=gradient_min))

class Agent:
    def __init__(self, q_net, target_net, memory, batch_size, discount_factor, actionspace_size, epsilon, epsilon_decay, epsilon_min, noisy_flag):
        self.q_net = q_net
        self.target_net = target_net
        self.memory = memory
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.actionspace_size = actionspace_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.weight_updates = 0
        self.noisy_flag = noisy_flag

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
        q_targets = self.target_net.predict([next_states, actions_mask])

        for i in range(self.batch_size):
            if terminals[i]:
                q_values[i] = rewards[i]
            else:
                q_values[i] = rewards[i] + self.discount_factor * np.max(q_targets[i])

        one_hot_actions = np.eye(self.actionspace_size)[np.array(actions).reshape(-1)]
        self.q_net.fit([prev_states, one_hot_actions], one_hot_actions * q_values[:, None], batch_size=self.batch_size, epochs=1, verbose=0)
        self.weight_updates += 1
        self.updateEpsilon()
        
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

    def useEpsilonGreedy(self, state):
        if not self.noisy_flag and random.random() < self.epsilon:
            action = random.randrange(self.actionspace_size)
        else:
            action = self.chooseAction(state)

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
    observation = np.uint8(observation * 255)
    return observation

def writeLog(path, content):
    with open(path, 'a') as log:
        csv_writer = csv.writer(log, delimiter=';')
        csv_writer.writerow(content)

def saveModel(path, model):
    with CustomObjectScope({"NoisyLayer":NoisyLayer}):
        model.save(path)

def loadModel(path):
    with CustomObjectScope({"NoisyLayer":NoisyLayer}):
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
    path = './Data/NoisyNet/' + environment_id + '/' + session_id + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    print("This is DQN " + environment_id + " " + session_id + "\nSession will be stored at " + path)

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
    epsilon = 1
    epsilon_decay = 1e-06
    epsilon_min = 0.1
    noisy_flag = True

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
        q_net = Network(actionspace_size, learning_rate, gradient_momentum, gradient_min, noisy_flag).model
        with CustomObjectScope({"NoisyLayer":NoisyLayer}):
            target_net = copy.deepcopy(q_net)
        memory = deque(maxlen=1000000)
        agent = Agent(q_net, target_net, memory, batch_size, discount_factor, actionspace_size, epsilon, epsilon_decay, epsilon_min, noisy_flag)
        step_number = 0

    end_time = time.time() + 250000

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
            action = agent.useEpsilonGreedy(state)
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
            if step_number > training_start:
                agent.train()
                
                # Potentially update target net
                if step_number % update_target_step == 0:
                    agent.updateTargetNet()

                # Evaluation games
                if step_number % evaluation_step == 0:
                    agent.play(eval_environment, evaluation_games, path)

        # Produce output
        writeLog(path + 'log.csv', [agent.weight_updates, accumulated_epoch_reward, agent.epsilon])

    # Save models and agent
    saveModel(path + 'qmodel.h5', agent.q_net)
    saveModel(path + 'targetmodel.h5', agent.target_net)
    agent.q_net = agent.target_net = None
    saveAgent(path + 'agent.pkl', agent)

if __name__ == "__main__":
    main()
