import gym
import PIL
import copy
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim
import torchvision as tv

from collections import namedtuple

# Decide whether to run on GPU or CPU
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Environment
environment = gym.make('IceHockey-v0')

# frame: observed frame for which action had to be chosen
# action: action chosen given frame
# reward & done: observed after performance of action at frame
Experience = namedtuple('Experience', ('frame', 'action', 'reward', 'done'))

# Used for training
TrainingExample = namedtuple('TrainingExample', ('current_state', 'current_state_actions', 'next_state', 'next_state_actions', 'reward', 'done'))

class Network(nn.Module):
    def __init__(self, learning_rate, output_size):
        super(Network, self).__init__()

        self.flattened_size = self.computeConvOutputDim()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)

        self.fc1 = nn.Linear(in_features=self.flattened_size + 3, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = funct.smooth_l1_loss

        self.to(device)
    
    def forward(self, observation, previous_actions):
        if not observation.is_cuda:
            observation.to(device)
        if not previous_actions.is_cuda:
            previous_actions.to(device)
    
    def computeConvOutputDim(self): #works!
        # width
        width = self.computeOutputDimensionConvLayer(in_dim=160, kernel_size=8, padding=0, stride=4)        #conv1
        width = self.computeOutputDimensionConvLayer(in_dim=width, kernel_size=4, padding=0, stride=2)      #conv2
        width = self.computeOutputDimensionConvLayer(in_dim=width, kernel_size=2, padding=0, stride=1)      #conv3
        
        # height
        height = self.computeOutputDimensionConvLayer(in_dim=210, kernel_size=8, padding=0, stride=4)       #conv1
        height = self.computeOutputDimensionConvLayer(in_dim=height, kernel_size=4, padding=0, stride=2)    #conv2
        height = self.computeOutputDimensionConvLayer(in_dim=height, kernel_size=2, padding=0, stride=1)    #conv3
        
        # width * height * out_channels
        flattened_size = width * height * 64
        
        return flattened_size

    def computeOutputDimensionConvLayer(self, in_dim, kernel_size, padding, stride): #works!
        return int((in_dim - kernel_size + 2 * padding) / stride + 1)

class Agent(object):
    def __init__(self, gamma, epsilon, epsilon_min, epsilon_decay, frame_skip_rate,\
                 action_space, memory_capacity, batch_size, training_epochs,\
                 update_target_net, q_net, v_net, target_v_net):
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

        # Declarations
        self.current_state = None   # Tensor of current state (=4 most recent frames); to be updated by self.constructCurrentStateAndActions()
        self.last_actions = None    # Tensor of last 3 actions; to be updated by self.constructCurrentStateAndActions()
        self.action = 0             # Most recent action performed; used by self.constructCurrentSateAndActions()
    
    # Returns tensor of current frame of environment
    def getGrayscaleFrameTensor(self):
        image = PIL.Image.fromarray(environment.render(mode='rgb_array'))           # Frame to PIL.Image
        image = tv.transforms.functional.to_grayscale(image, num_output_channels=1) # Use torchvision to convert to grayscale
        image = np.array(image)                                                     # Convert PIL image back to numpy-array
        return t.from_numpy(image).type('torch.FloatTensor')                        # Create tensor from numpy array
    
    # Here the experience only consists of the current frame and the action taken in that frame
    def storeExperience(self, *experience):
        if len(self.memory) < self.memory_capacity:
            self.memory.append(None)

        self.memory[self.memory_index] = Experience(*experience)
        self.memory_index = (self.memory_index + 1) % self.memory_capacity
    
    def performGreedyChoice(self):
        choose_max = False
        if np.random.random() > self.epsilon:
            choose_max = True
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        return choose_max
    
    def chooseAction(self, state, actions):
        if self.performGreedyChoice():
            q_values = self.q_net(state.to(device), actions.to(device))
            squeezed_q_values = t.squeeze(q_values)
            reward, action = squeezed_q_values.max(0)
            return action.item()
        else:
            return environment.action_space.sample()

## Main program
def main():
    print('Hello world!')
    environment.reset()
    
    # Variable assignments
    learning_rate = 1e-4
    gamma = 0.95 # Discount factor
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 1e-5
    frame_skip_rate = 3
    action_space = environment.action_space.n
    memory_capacity = 35000
    batch_size = 16
    trainings_epochs = 100
    update_target_net = 30
    
    q_net = Network(learning_rate, action_space)
    v_net = Network(learning_rate, 1)
    target_v_net = copy.deepcopy(v_net)
    
    agent = Agent(gamma, epsilon, epsilon_min, epsilon_decay, frame_skip_rate,\
                  action_space, memory_capacity, batch_size, trainings_epochs,\
                  update_target_net, q_net, v_net, target_v_net)
                 
    agent.train()
    
if __name__ == "__main__": # Call main function
    main()
