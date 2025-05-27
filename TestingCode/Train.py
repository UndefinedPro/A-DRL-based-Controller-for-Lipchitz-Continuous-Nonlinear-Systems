from collections import namedtuple
from NonlinearSys import SpringSys, Manipulator
from DRL import DDPG, REPLAY_BUFFER, AL
from ExcelHandler import ExcelWriter as ew
import numpy as np

"""
The class diagram of training 

TRAIN
    |_ TRAIN_DDPG 
    |           |_ TRAIN_SPRING             (Train Spring system with DDPG)
    |           |_ TRAIN_MANIPULATOR        (Train Manipulator system with DDPG)
    |_ TRAIN_AL     
              |_ TRAIN_AL_SPRING            (Train Spring system with AL)
              |_ TRAIN_AL_MANIPULATOR       (Train Manipulator system with AL)
"""

# Define the Transition structure
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state','act_others','others'])


# A basic class for training 
class TRAIN():

    def __init__(self, NetName, TrainingTraj = 2000) -> None:
        self.net_name = NetName
        self.training_traj = TrainingTraj   
        self.InitParamSet()

    def InitParamSet(self):
        self.InitParamDefined = False
        self.systemParam = {}
        self.netParam = {}
        self.sys = 0
        self.net = 0 


    def SelfDefinedPrint(self, PositionIndex, buffer):
        pass

    def Train(self):
        
        if (not self.InitParamDefined) : 
            print('ERROR : Please Initial the Parameters !')
            return 1

        TrainingBuffer = []
        replay_Buffer = REPLAY_BUFFER(capacity = 100000)


        for i in range(self.training_traj):
            print('===================Trajectory {}==================='.format(i))
            state = self.sys.reset()
            
            self.SelfDefinedPrint(1, TrainingBuffer)

            done = False
            trans_counter = 0
            
            while not done: 
                trans_counter += 1
                action, act_others = self.net.GetAction(state)
                # add action noise for more exploration
                noise = np.random.normal(0, 0.01, size=action.shape)
                next_state, reward, done, others = self.sys.step(action + noise)
                trans = Transition(state, action, reward, next_state, act_others, others)
                replay_Buffer.push(trans)
                TrainingBuffer.append(trans)
                self.net.update(trans_counter, i, replay_Buffer.sample(self.net.batch_size))
                state = next_state.copy()

            self.SelfDefinedPrint(2, TrainingBuffer)   
            self.SaveTrainData(TrainingBuffer)
            
            print('===================Trajectory {} Training Finish==================='.format(i))
            del TrainingBuffer[:]
        self.net.SaveNetParam()


    def Test(self, LoadNetPath):
        
        if (not self.InitParamDefined) : 
            print('ERROR : Please Initial the Parameters !')
            return 1

        #  Load network parameters
        self.net.LoadNetParam([LoadNetPath])

        TestBuffer = []
        state = self.sys.reset()
        self.SelfDefinedPrint(11, TestBuffer)

        done = False
        trans_counter = 0
            
        while not done: 
            trans_counter += 1
            action, act_others = self.net.GetAction(state)
            next_state, reward, done, others = self.sys.step(action)
            trans = Transition(state, action, reward, next_state, act_others, others)
            state = next_state.copy()
            TestBuffer.append(trans)

        self.SaveTestData(TestBuffer)
        return TestBuffer
    
    def SaveTrainData(self):
        pass 
    
    def SaveTestData(self):
        pass


# A subclass for training DDPG
class TRAIN_DDPG(TRAIN):

    def __init__(self, NetName, TrainingTraj=2000):
        super().__init__(NetName, TrainingTraj)
        self.reward_buffer = []
        self.step_buffer = []
    
    def SelfDefinedPrint(self, PositionIndex, buffer):
        # Here is an example for using function "SelfDefinedPrint" 
        if PositionIndex == 2:
            print('Trajectory finished {} steps'.format(len(buffer)))
            print('Total reward is {}'.format(sum([t.reward for t in buffer])))

        # You can print any thing you need by adding "if PositionIndex == xx:" 
        # And the data you need can be added in buffer['others']
        # The following is an instance: 
        """
        if PositionIndex == 11: 
            print('Some training process data / testing process data like {}'.format(sum[it.others['MyProperty'] for it in buffer]))
        """

    def SaveTrainData(self, TrainingBuffer):
        
        total_reward = sum([t.reward for t in TrainingBuffer])
        self.reward_buffer.append(total_reward)
        self.step_buffer.append(len(TrainingBuffer))

        if len(self.reward_buffer) == self.training_traj:
            step_writer = ew('./Data/', self.net_name + '_step.xlsx')
            reward_writer = ew('./Data/', self.net_name + '_reward.xlsx')
            reward_np = np.array(self.reward_buffer).reshape(len(self.reward_buffer), 1)
            step_np = np.array(self.step_buffer).reshape(len(self.reward_buffer), 1)
            reward_writer.AddData2Excel(reward_np)
            step_writer.AddData2Excel(step_np)

# A subclass for training DDPG with spring system
class TRAIN_SPRING(TRAIN_DDPG):

    def __init__(self, NetName, TrainingTraj=2000):
        super().__init__(NetName, TrainingTraj)
    
    def InitParamSet(self):

        self.InitParamDefined = True
        self.systemParam = {
            'm' : 0.1,
            'k' : 0.05,
            'x_threshold' : 4.8,
            'start_state_bound' : 4.0,
            'dT': 0.005,
        }

        self.netParam = {
            'StateNum'  : 2,       
            'ActionNum' : 1,        
            'MaxAction' : 5,
            'learning_rate'  : 1e-4,
            'gamma'          : 0.95,
            'tau'            : 0.005,
            'batch_size'     : 1024,
            'mini_batch_size': 64,
            'clip_param'     : 0.2,
            'max_grad_norm'  : 0.5,
            'delta_T'        : 0.005,      
            'max_train_traj' : self.training_traj,
            'param_save_path': './param/DDPG/Temp_param/', 
            'net_name' : 'DDPG', 
        }
        self.sys = SpringSys(self.systemParam)
        self.net = DDPG(self.netParam)

    def SaveTestData(self, TestBuffer):

        # Spring Data
        x = [t.state[0,0] for t in TestBuffer]
        v = [t.state[0,1] for t in TestBuffer]
        u = [t.action for t in TestBuffer]

        state_writer = ew('./Data/DDPG/Temp_data/', self.net_name + '_x.xlsx')
        v_writer = ew('./Data/DDPG/Temp_data/', self.net_name + '_v.xlsx')
        u_writer = ew('./Data/DDPG/Temp_data/', self.net_name + '_u.xlsx')

        x_np = np.array(x).reshape(len(x), 1)
        v_np = np.array(v).reshape(len(v), 1)
        u_np = np.array(u).reshape(len(u), 1)

        state_writer.AddData2Excel(x_np)
        v_writer.AddData2Excel(v_np)
        u_writer.AddData2Excel(u_np)

# A subclass for training DDPG with manipulator system
class TRAIN_MANIPULATOR(TRAIN_DDPG):
    
    def __init__(self, NetName, TrainingTraj=2000):
        super().__init__(NetName, TrainingTraj)
    
    def InitParamSet(self):

        self.InitParamDefined = True
        # Prameters for training manipulator
        self.systemParam = {
            'm1' : 1,
            'm2' : 0.5,
            'l1' : 1,
            'l2' : 0.5,
            'l_c1' : 0.5,
            'l_c2' : 0.25,
            'dT'   : 0.001,
        }
        self.netParam = {
            'StateNum'  : 4,       
            'ActionNum' : 2,        
            'MaxAction' : 5,
            'learning_rate'  : 1e-4,
            'gamma'          : 0.95,
            'tau'            : 0.005,
            'batch_size'     : 1024,
            'mini_batch_size': 64,
            'clip_param'     : 0.2,
            'max_grad_norm'  : 0.5,
            'delta_T'        : 0.001,      
            'max_train_traj' : self.training_traj,
            'param_save_path': './param/AL/Temp_param/', 
            'net_name' : 'DDPG', 
        }
        self.sys = Manipulator(self.systemParam)
        self.net = DDPG(self.netParam)

    def SaveTestData(self, TestBuffer):

        # Manpulator Data
        theta1 = [t.state[0,0] for t in TestBuffer]
        theta2 = [t.state[0,1] for t in TestBuffer]
        d_theta1 = [t.state[0,2] for t in TestBuffer]
        d_theta2 = [t.state[0,3] for t in TestBuffer]

        theta1_np = np.array(theta1).reshape(len(theta1), 1)
        theta2_np = np.array(theta2).reshape(len(theta2), 1)
        d_theta1_np = np.array(d_theta1).reshape(len(d_theta1), 1)
        d_theta2_np = np.array(d_theta2).reshape(len(d_theta2), 1)

        state_writer = ew('./Data/DDPG/Temp_data/', self.net_name + '_state.xlsx')
        u_writer = ew('./Data/DDPG/Temp_data/', self.net_name + '_u.xlsx')
        state_writer.AddData2Excel(theta1_np)
        state_writer.AddData2Excel(theta2_np)
        state_writer.AddData2Excel(d_theta1_np)
        state_writer.AddData2Excel(d_theta2_np)

        u1 = [t.action[0,0] for t in TestBuffer]
        u2 = [t.action[0,1] for t in TestBuffer]
        u1_np = np.array(u1).reshape(len(u1), 1)
        u2_np = np.array(u2).reshape(len(u2), 1)
        u_writer.AddData2Excel(u1_np)
        u_writer.AddData2Excel(u2_np)


# A subclass for testing AL
class TRAIN_AL(TRAIN):

    def __init__(self, NetName, TrainingTraj=2000):
        super().__init__(NetName, TrainingTraj)
        self.reward_buffer = []
        self.step_buffer = []

# # A subclass for training AL with spring system       
class TRAIN_AL_SPRING(TRAIN_AL):

    def InitParamSet(self):
        self.InitParamDefined = True
        self.systemParam = {
            'm' : 0.1,
            'k' : 0.05,
            'x_threshold' : 4.8,
            'start_state_bound' : 4.0,
            'dT': 0.005,
        }
        self.netParam = {
            'StateNum'  : 2,
            'ActionNum' : 1,
            'learning_rate'  : 1e-4,
            'epoch'          : 1,
            'batch_size'     : 128,
            'clip_param'     : 0.2,
            'max_grad_norm'  : 0.5,
            'delta_T'        : 0.005,
            'hidden_layer'   : 64, 
            'max_act'        : 5.0,
            'max_train_traj' : self.training_traj,
            'param_save_path': './param/AL/Temp_param/', 
            'net_name' : 'Lyp', 
        }


        self.sys = SpringSys(self.systemParam)
        self.net = AL(self.netParam)


    def SaveTestData(self, TestBuffer):

        # Spring Data
        x = [t.state[0,0] for t in TestBuffer]
        v = [t.state[0,1] for t in TestBuffer]
        u = [t.action for t in TestBuffer]

        state_writer = ew('./Data/AL/Temp_data/', self.net_name + '_x.xlsx')
        v_writer = ew('./Data/AL/Temp_data/', self.net_name + '_v.xlsx')
        u_writer = ew('./Data/AL/Temp_data/', self.net_name + '_u.xlsx')

        x_np = np.array(x).reshape(len(x), 1)
        v_np = np.array(v).reshape(len(v), 1)
        u_np = np.array(u).reshape(len(u), 1)

        state_writer.AddData2Excel(x_np)
        v_writer.AddData2Excel(v_np)
        u_writer.AddData2Excel(u_np)

# A subclass for training AL with manipulator system
class TRAIN_AL_MANIPULATOR(TRAIN_AL):
    
    def InitParamSet(self):
        self.InitParamDefined = True
        self.systemParam = {
            'm1' : 1,
            'm2' : 0.5,
            'l1' : 1,
            'l2' : 0.5,
            'l_c1' : 0.5,
            'l_c2' : 0.25,
            'dT'   : 0.001,
        }
        # The net parameters except StateNum, ActionNum, hidden_layer and delta_T, are set randomly
        self.netParam = {
            'StateNum'  : 4,
            'ActionNum' : 2,
            'learning_rate'  : 1e-4,
            'epoch'          : 1,
            'batch_size'     : 128,
            'clip_param'     : 0.2,
            'max_grad_norm'  : 0.5,
            'delta_T'        : 0.001,
            'hidden_layer'   : 128, 
            'max_act'        : 15.0,
            'max_train_traj' : self.training_traj,
            'param_save_path': './param/AL/Temp_param/', 
            'net_name' : 'Lyp', 
        }

        self.sys = Manipulator(self.systemParam)
        self.net = AL(self.netParam)

    def SaveTestData(self, TestBuffer):

        # Manpulator Data
        theta1 = [t.state[0,0] for t in TestBuffer]
        theta2 = [t.state[0,1] for t in TestBuffer]
        d_theta1 = [t.state[0,2] for t in TestBuffer]
        d_theta2 = [t.state[0,3] for t in TestBuffer]

        theta1_np = np.array(theta1).reshape(len(theta1), 1)
        theta2_np = np.array(theta2).reshape(len(theta2), 1)
        d_theta1_np = np.array(d_theta1).reshape(len(d_theta1), 1)
        d_theta2_np = np.array(d_theta2).reshape(len(d_theta2), 1)

        state_writer = ew('./Data/AL/Temp_data/', self.net_name + '_state.xlsx')
        u_writer = ew('./Data/AL/Temp_data/', self.net_name + '_u.xlsx')
        state_writer.AddData2Excel(theta1_np)
        state_writer.AddData2Excel(theta2_np)
        state_writer.AddData2Excel(d_theta1_np)
        state_writer.AddData2Excel(d_theta2_np)

        u1 = [t.action[0,0] for t in TestBuffer]
        u2 = [t.action[0,1] for t in TestBuffer]
        u1_np = np.array(u1).reshape(len(u1), 1)
        u2_np = np.array(u2).reshape(len(u2), 1)
        u_writer.AddData2Excel(u1_np)
        u_writer.AddData2Excel(u2_np)