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
    def __init__(self, learning_rate, action_space):
        super(Network, self).__init__()
        
        self.flattened_size = self.computeConvOutputDim()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        
        self.fc1 = nn.Linear(in_features=self.flattened_size + 3, out_features=256) # +3 for 3 actions to be added
        self.fc2 = nn.Linear(in_features=256, out_features=action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = funct.smooth_l1_loss

        self.to(device)
    
    def forward(self, observation, previous_actions): #works!
        if not observation.is_cuda:
            print('Observation wasn\'t cuda!')
            observation.to(device)
        
        if not previous_actions.is_cuda:
            print('Previous_actions wasn\'t cuda!')
            previous_actions.to(device)
        
        observation = funct.relu(self.conv1(observation))
        observation = funct.relu(self.conv2(observation))
        observation = funct.relu(self.conv3(observation))
        
        observation = observation.view(1, self.flattened_size)  # view works directly on tensors
        observation = t.cat((previous_actions, observation), 1) # concatenates tensors (actions, conv-output)
        observation = funct.relu(self.fc1(observation))
        q_values = self.fc2(observation)
        
        return q_values

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
                 action_space, memory_capacity, batch_size, trainings_epochs,\
                 update_target_net, q_net, target_net):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space
        self.batch_size = batch_size
        self.frame_skip_rate = frame_skip_rate
        self.q_net = q_net
        self.target_net = target_net
        self.memory = []
        self.memory_index = 0
        self.memory_capacity = memory_capacity
        self.trainings_epochs = trainings_epochs
        self.update_target_net = update_target_net
        
        # Declarations
        self.current_state = None   # Tensor of current state(=4 most recent frames); to be updated by self.constructCurrentStateAndActions()
        self.last_actions = None    # Tensor of last 3 actions; to be updated by self.constructCurrentStateAndActions()
        self.action = 0             # Most recent action performed; used by self.constructCurrentStateAndActions()
    
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
    
    # Here the TrainingExample consists of the current frame plus the last three frames and the actions that led to them + the same for the next state
    def constructSample(self, batch_size):
        mini_batch = []
        random_indices = np.random.random_integers(low=0, high=self.action_space, size=batch_size)  # Number(batch_size) random ints from [low, high)

        for i in random_indices:
            while i == self.memory_index:
                i = np.random.random_integers(low=0, high=self.action_space, size=1)[0]             # For the current_index we don't have a 'next_state' yet; choose another action
                
            # 'TrainingExample' = ('current_state', 'current_state_actions', 'next_state', 'next_state_actions', 'reward', 'done')
            current_state = []
            current_state_actions = t.zeros([1, 3], dtype=t.float32)
            next_state = []
            next_state_actions = t.zeros([1, 3], dtype=t.float32)
            
            #current_state{_actions}
            current_state.append(self.memory[i].frame)                               # Frame for which prediction is to be made
            for offset in range(1, 4):                                               # For last 3 frames & actions which led to state for which prediction is to be made
                index = (i - offset) % self.memory_capacity
                current_state.append(self.memory[index].frame)
                current_state_actions[0, offset - 1] = float(self.memory[index].action)
            
            #next_state{_actions}
            if not self.memory[i].done:
                index = (i + 1) % self.memory_capacity
                next_state.append(self.memory[index].frame)
                for offset in range(0, 3):
                    index = (i - offset) % self.memory_capacity
                    next_state.append(self.memory[index].frame)
                    next_state_actions[0, offset] = float(self.memory[index].action)
            
            # Convert to right tensor format
            current_state = t.unsqueeze(t.stack(current_state), 0)
            if not self.memory[i].done:
                next_state = t.unsqueeze(t.stack(next_state), 0)
            
            mini_batch.append(TrainingExample(current_state, current_state_actions, next_state, next_state_actions, self.memory[i].reward, self.memory[i].done))

        return mini_batch
    
    # Target values dimensionality shall be right, since it has to match dimensionality of q-values returned from net
    def updateNetwork(self):
        
        if len(self.memory) < self.memory_capacity:
            return
        
        mini_batch = self.constructSample(self.batch_size)

        for sample in mini_batch:
            q_values = self.q_net(sample.current_state.to(device), sample.current_state_actions.to(device))
            
            if sample.done:
                max_future_reward = 0
            else:
                discounted_future_rewards = self.gamma * self.target_net(sample.next_state.to(device), sample.next_state_actions.to(device))
                max_future_reward, _ = t.squeeze(discounted_future_rewards).max(0)

            max_future_reward += sample.reward

            target_values = q_values.clone()
            target_values[0, int(sample.next_state_actions[0, 0])] = max_future_reward

            self.q_net.optimizer.zero_grad()
            loss = self.q_net.loss(q_values, target_values)
            loss.backward()
            self.q_net.optimizer.step()
    
    # Function to keep current state & current last_actions (multi)set up to date; shall return data to be inserted immediately into network
    # Function appears to work properly!
    def constructCurrentStateAndActions(self, init=False):
        if init: #seems to work
            init_frame = self.getGrayscaleFrameTensor()
            self.current_state = [init_frame.clone(), init_frame.clone(), init_frame.clone(), init_frame.clone()]
            self.current_state = t.unsqueeze(t.stack(self.current_state), 0)
            self.last_actions = t.zeros([1, 3], dtype=t.float32)
        else:
            # 4 frames --> state - works!
            self.current_state[0, 3] = self.current_state[0, 2].clone()
            self.current_state[0, 2] = self.current_state[0, 1].clone()
            self.current_state[0, 1] = self.current_state[0, 0].clone()
            self.current_state[0, 0] = self.getGrayscaleFrameTensor()
            # 3 last actions - works!
            self.last_actions[0, 2] = self.last_actions[0, 1]
            self.last_actions[0, 1] = self.last_actions[0, 0]
            self.last_actions[0, 0] = self.action
    
    def train(self):
        target_net_replacement_counter = 0
        for epoch in range(self.trainings_epochs):
            print('Epoch: ', epoch)
            environment.reset()                             # Start new game
            self.constructCurrentStateAndActions(init=True) # Initialize current state and last actions
            done = False
            accumulated_epoch_reward = 0
            
            while not done: #TODO: maybe add max number of rounds
                self.action = self.chooseAction(self.current_state, self.last_actions)
                
                #frame skipping - needs improvement, otherwise network misses rewards to be observed
                #for skip in range(self.frame_skip_rate):
                    #environment.step(self.action)
                
                _, reward, done, _ = environment.step(self.action)
                
                self.storeExperience(self.getGrayscaleFrameTensor(), self.action, reward, done)
                self.constructCurrentStateAndActions()      # Update current state and actions
                self.updateNetwork()
                
                if target_net_replacement_counter > self.update_target_net:
                    self.target_net = copy.deepcopy(self.q_net)
                    target_net_replacement_counter = target_net_replacement_counter % self.update_target_net
                target_net_replacement_counter += 1
                
                #environment.render()
                accumulated_epoch_reward += reward
                
            print('Reward last epoch: ', accumulated_epoch_reward, ' Epsilon: ', self.epsilon)

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
    trainings_epochs = 200
    update_target_net = 30
    
    q_net = Network(learning_rate, action_space)
    target_net = copy.deepcopy(q_net)
    
    agent = Agent(gamma, epsilon, epsilon_min, epsilon_decay, frame_skip_rate,\
                 action_space, memory_capacity, batch_size, trainings_epochs,\
                 update_target_net, q_net, target_net)
                 
    agent.train()
    
if __name__ == "__main__": # Call main function
    main()
