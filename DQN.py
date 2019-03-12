import numpy as NP
import torch as T
import torch.nn as NN
import torch.nn.functional as FUNCT
import torch.optim as OPTIM
import gym

from collections import namedtuple

#decide whether to run on GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#environment
environment = gym.make('IceHockey-v0')

#experiences
Experience = namedtuple('Experience', ('frame', 'action', 'reward', 'done'))

#TODO: flattened_size
class Network(nn.Module):
    def __init__(self, learning_rate, action_space):
        super(Network, self).__init__()

        self.conv1 = nnConv2d(in_channels = 4, out_channels = 16, kernel_size = 8, stride = 4)
        self.conv2 = nnConv2d(in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2)
        self.conv3 = nnConv2d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 1)

        self.fc1 = nn.Linear(in_features = flattened_size, out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = action_space)

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss()

        self.to(device)

    def forward(self, observation, actions):
        observation = np.mean(observation, axis = 2)
        observation = F.relu(conv1(observation))
        observation = F.relu(conv2(observation))
        observation = F.relu(conv3(observation))

        observation = observation.view(1, -1)
        observation = F.relu(sefl.fc1(observation))
        actions = self.fc2(observation)
        return actions

class Agent(object):
    def __init__(self, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, action_space, memory_capacity, batch_size, q_net, target_net):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon.decay = epsilon_decay
        self.action_space = action_space
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.q_net = q_net
        self.target_net = target_net
        self.memory = []
        self.memory_index = 0

    def storeExperience(self, *experience):
        if (len(self.memory) < self.capacity):
            self.memory.append(None)

        self.memory[self.memory_index] = Experience(*experience, self.memory_index)
        self.memory_index = (self.memory_index + 1) % self.memory_capacity

    def chooseMax(self):
        choose_max = False
        if np.random.random() > epsilon:
            choose_max = True
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        return choose_max

    def chooseAction(self, observation, actions):
        if (self.chooseMax()):
            q_values = self.q_net(observation, actions) #TODO preprocessing
            squeezed_q_values = torch.squeeze(q_values().clon()) #TODO vllt ohne clone
            reward, action = squeezed_q_values.max(0)
            return action.item()
        else:
            environment.action_space.sample()

    def constructSample(self, batch_size):
        return None

    def updateNetwork(self):
        mini_batch = self.constructSample(self.batch_size)

        for sample in mini_batch:
            q_values = self.q_net(sample.state, sample.actions)
            if sample.done:
                max_future_reward = 0
            else:
                discounted_future_rewards = self.gamma * self.target_net(sample.next_state, sample.next_actions)
                max_future_reward, _ = torch.squeeze(discounted_future_rewards.max(0))

            max_future_reward += sample.reward
#next_actions = previous actions plus current action
            target_values = q_values.clone()
            target_values[0, int(sample.next_actions[0, 0])] = max_future_reward

            self.optimizer.zero_grad()
            loss = self.q_net.loss(q_values, target_values)
            loss.backward()
            self.q_net.optimizer.step()

if __name__ '__main__':
    print('start')
