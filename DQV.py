import os
import gym
import random
import time
import copy

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim
import torchvision as tv

from collections import deque
from skimage.color import rgb2array
from skimage.transform import resize

class Network(nn.Module):
    def __init__(self, learning_rate, action_space):
        super(Network, self).__init__()

        self.flattened_size = self.computeConvOutputDim()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)

        self.fc1 = nn.ReLU()

class Agent(object):
    def __init__(self, gamma, epsilon, epsilon_min, epsilon_decay, frame_skip_rate,\
                 action_space, memory_capacity, batch_size, training_epochs,\
                 update_target_net, q_net, v_net, target_v_net, final_v_activation):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space
        self.batch_size = batch_size
        self.frame_skip_rate = frame_skip_rate
        self.q_net = q_net
        self.v_net = v_net
        self.target_v_net = target_v_net
        self.memory = []
        self.memory_index = 0
        self.memory_capacity = memory_capacity
        self.training_epochs = training_epochs
        self.update_target_net = update_target_net
        self.final_v_activation = final_v_activation

        # Declarations
        self.current_state = None   # Tensor of current state (=4 most recent frames); to be updated by self.constructCurrentStateAndActions()
        self.last_actions = None    # Tensor of last 3 actions; to be updated by self.constructCurrentStateAndActions()
        self.action = 0             # Most recent action performed; used by self.constructCurrentSateAndActions()
