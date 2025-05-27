import math
import re
import os
from numpy.random.mtrand import seed
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal, Categorical, Beta
from collections import namedtuple, deque


"""
Explanation: 
Owing to the limitation of device, the simulation is conductes on CPU. 
"""


class REPLAY_BUFFER():
    def __init__(self, capacity):
        self.capacity = capacity    
        self.buffer = deque(maxlen = capacity)
    
    def push(self, transition): 
        self.buffer.append(transition)

    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:  # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return batch
        else:
            batch = random.sample(self.buffer, batch_size)
            return batch

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

# parameter orthogonalization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


######################## DDPG ########################

class Critic(nn.Module):
    def __init__(self, state_num, act_num, lr, orthogonal = True):
        super().__init__()
        hidden_layers = 128
        self.fc1 = nn.Linear(state_num + act_num, hidden_layers)
        self.lr = lr
        self.state_num = state_num
        self.act_num = act_num
        self.surprise_struc = nn.Sequential(
            nn.Linear(hidden_layers, hidden_layers // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers // 2, hidden_layers // 2),
            nn.Tanh(), 
            nn.Linear(hidden_layers // 2, 1)
        )
        if orthogonal:
            orthogonal_init(self.surprise_struc[0])
            orthogonal_init(self.surprise_struc[2])
            orthogonal_init(self.surprise_struc[4])

    def forward(self, s, a):
        x = self.fc1(torch.cat((s, a), 2))
        return self.surprise_struc(x)

class Actor(nn.Module):
    def __init__(self, state_num, act_num, max_action, lr, orthogonal = True):
        super().__init__()
        hidden_layer = 128
        self.lr = lr
        self.state_num = state_num
        self.act_num = act_num
        self.max_action = max_action
        self.fc1 = nn.Linear(state_num, hidden_layer)
        self.act_struc = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer//2),
            nn.Tanh(),
            nn.Linear(hidden_layer//2, hidden_layer//4),
            nn.Tanh(),
            nn.Linear(hidden_layer//4 , act_num),
            nn.Tanh()
        )
        if orthogonal:
            orthogonal_init(self.fc1)
            orthogonal_init(self.act_struc[0])
            orthogonal_init(self.act_struc[2])
            orthogonal_init(self.act_struc[4])

    def forward(self, s):
        x = self.fc1(s)
        return self.act_struc(x) * self.max_action

######################## DDPG ########################

######################## AL ########################

# AL Actor net
class AL_Net(nn.Module):
    def __init__(self, state_num, act_num, hidden_layer, lr, orthogonal = True):
        super().__init__()
        self.lr = lr
        self.state_num = state_num
        self.act_num = act_num
        self.alpha_struc = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer//2),
            nn.Tanh(),
            nn.Linear(hidden_layer//2, hidden_layer//4),
            nn.Tanh(),
            nn.Linear(hidden_layer//4 , act_num),
            nn.Softplus()
        )
        self.beta_struc = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer//2),
            nn.Tanh(),
            nn.Linear(hidden_layer//2, hidden_layer//4),
            nn.Tanh(),
            nn.Linear(hidden_layer//4 , act_num),
            nn.Softplus()
        )
        self.fc1 = nn.Linear(self.state_num, hidden_layer)
        if orthogonal:
            orthogonal_init(self.fc1)
            orthogonal_init(self.alpha_struc[0])
            orthogonal_init(self.alpha_struc[2])
            orthogonal_init(self.alpha_struc[4])
            orthogonal_init(self.beta_struc[0])
            orthogonal_init(self.beta_struc[2])
            orthogonal_init(self.beta_struc[4])

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        alpha = self.alpha_struc(x)  + 1.0
        beta  = self.beta_struc(x)   + 1.0
        return alpha, beta

######################## AL ########################

class DRL():
    def __init__(self, param_config) -> None:
        self.state_num  = param_config['StateNum']
        self.action_num = param_config['ActionNum']
        self.net_name = param_config['net_name']
        self.param_save_path = param_config['param_save_path']
        self.subnet_name_struc = { }

    def Numpy2Tensor(self, Mynp):
        return torch.tensor(Mynp, dtype=torch.float)
    
    def Tensor2Numpy(self, MyTensor):
        return MyTensor.numpy()
    
    def SaveNetParam(self):

        if len(self.subnet_name_struc) == 0:
            print('Please fill in the subnet_name_struc dict')
        else:
            i = 0
            while True:
                subnet_name_ = self.net_name + list(self.subnet_name_struc.keys())[0] + str(i)  + '.pkl' 
                if os.path.exists(self.param_save_path + subnet_name_):
                    i += 1
                    continue
                else:  
                    for key in self.subnet_name_struc:
                        subnet_name_ = self.net_name + key + str(i)  + '.pkl' 
                        subnet_save_path = self.param_save_path + subnet_name_
                        torch.save(self.subnet_name_struc[key].state_dict(), subnet_save_path)
                    break
            print('Save model successfully.')

    def LoadNetParam(self, load_net_path):
        net_not_load = []
        if len(self.subnet_name_struc) == 0:
            print('Please fill in the subnet_name_struc dict')
            return 
        
        for key in self.subnet_name_struc:
            name_re = re.compile(self.net_name + key + "\d+\.pkl")
            for i in range(len(load_net_path)):
                if name_re.search(load_net_path[i]):
                    self.subnet_name_struc[key].load_state_dict(torch.load(load_net_path[i]))
                    if torch.cuda.is_available():
                        self.subnet_name_struc[key] = self.subnet_name_struc[key].cuda()
                    break
                else:
                    if i == len(load_net_path)-1:
                        net_not_load.append(key)

        if len(net_not_load) == 0:                
            print('Load model successfully.')
        else:
            for it in net_not_load:
                print(it + 'is not loaded')


class DDPG(DRL):
    
    def __init__(self, param_config) -> None:
        # Paramters Initialization
        super().__init__(param_config)
        self.max_action = param_config['MaxAction']
        self.lr   = param_config['learning_rate']
        self.delta_T = param_config['delta_T']
        self.gamma   = param_config['gamma'] 
        self.tau     = param_config['tau']
        self.batch_size = param_config['batch_size']
        self.mini_batch_size = param_config['mini_batch_size']
        self.clip_param = param_config['clip_param']
        self.max_grad_norm = param_config['max_grad_norm']
        self.max_train_traj = param_config['max_train_traj']

        # Define the network structure
        self.actor_net  = Actor(self.state_num, self.action_num, self.max_action, self.lr)
        self.critic_net = Critic(self.state_num, self.action_num, self.lr)
        self.target_actor_net = Actor(self.state_num, self.action_num, self.max_action, self.lr)
        self.target_critic_net = Critic(self.state_num, self.action_num, self.lr)

        self.actor_optim   = optim.Adam(self.actor_net.parameters(), self.lr)
        self.critic_optim  = optim.Adam(self.critic_net.parameters(), self.lr)
        self.target_actor_optim = optim.Adam(self.critic_net.parameters(), self.lr)
        self.target_critic_optim = optim.Adam(self.critic_net.parameters(), self.lr)

        self.subnet_name_struc = {'actorNet' : self.actor_net ,
                                  'criticNet': self.critic_net, 
                                  'TargetActorNet' : self.target_actor_net, 
                                  'TargetCriticNet': self.target_critic_net}

    def DoActionTransform(self, net_action):
        return self.Tensor2Numpy(net_action)

    def DoActionDeTransform(self, TransformedAction):
        return TransformedAction

    def GetAction(self, state):
        # Initialization
        state_tensor = self.Numpy2Tensor(state)
        act_others = {}

        # Calculate the action
        with torch.no_grad():
            net_action = self.actor_net(state_tensor)
        action = self.DoActionTransform(net_action)

        # Package the remaining variables
        act_others['net_output'] = net_action
        
        return action, act_others
    

    def update(self, TransCounter, Traj, TrainingBuffer):

        if Traj <= 1: 
            return

        if TransCounter % (self.mini_batch_size // 2) != 0:
            return

        state = torch.tensor([t.state for t in TrainingBuffer], dtype=torch.float)
        action = torch.tensor([t.action for t in TrainingBuffer], dtype=torch.float).view(-1, 1, self.action_num)
        reward = torch.tensor([t.reward for t in TrainingBuffer], dtype=torch.float)
        next_state = torch.tensor([t.next_state for t in TrainingBuffer], dtype=torch.float)

        for index in BatchSampler(SubsetRandomSampler(range(len(TrainingBuffer))), self.mini_batch_size, False):
        
            # Update the critic network
            Q_real = reward[index].view(-1,1,1) + self.gamma * self.target_critic_net( next_state[index], self.target_actor_net(next_state[index]) )
            Q_est  = self.critic_net(state[index], action[index])
            critic_loss = F.mse_loss(Q_real, Q_est)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Update the actor network
            action_loss = -self.critic_net(state, self.actor_net(state) ).mean()
            self.actor_optim.zero_grad()
            action_loss.backward()
            self.actor_optim.step()

            # Update the target networks (critic target network and actor target network)
            for param, target_param in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




class AL(DRL):
    
    def __init__(self, param_config) -> None:

        # Initialization
        super().__init__(param_config)
        self.lr   = param_config['learning_rate']
        self.delta_T = param_config['delta_T']
        self.epoch = param_config['epoch']
        self.batch_size = param_config['batch_size']
        self.clip_param = param_config['clip_param']
        self.max_grad_norm = param_config['max_grad_norm']
        self.max_train_traj = param_config['max_train_traj']
        self.hidden_layer = param_config['hidden_layer']
        self.max_act = param_config['max_act']


        self.actor_net   = AL_Net(self.state_num, self.action_num, self.hidden_layer, self.lr)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.lr, maximize=True)
        self.subnet_name_struc = {'actorNet' : self.actor_net }
        self.epoch_counter = 0

    def DoActionTransform(self, net_action):
        transformed_action =  net_action * self.max_act * 2 - torch.ones_like(net_action, dtype=torch.float) * self.max_act
        return self.Tensor2Numpy(transformed_action)

    def DoActionDeTransform(self, TransformedAction):
        net_action = (TransformedAction + torch.ones_like(TransformedAction, dtype=torch.float) * self.max_act) / (self.max_act * 2)
        return net_action

    def GetAction(self, state):

        state_tensor = self.Numpy2Tensor(state)
        act_others = {}

        # Calculate the action
        with torch.no_grad():
            alpha, beta = self.actor_net(state_tensor)
        c = Beta(alpha, beta)
        net_action = c.rsample().float()
        action = self.DoActionTransform(net_action)
        
        return action, act_others