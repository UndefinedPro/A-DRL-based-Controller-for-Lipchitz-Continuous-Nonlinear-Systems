import torch
import numpy as np
import matplotlib.pyplot as plt
from math import cos as cos
from math import sin as sin

# The data type in NonlinearSys.py must be "numpy"
# Input：numpy
# Output：numpy

PI = 3.141592654

class UnlinearSystem():
    def __init__(self, param):
        self.delta_T = param['dT']
        self.iter_counter = 0
    def reset(self):
        pass

    def step(self):
        pass

    def rk4(self, func, x0, action, h):
        """
        Runge Kutta 4 order update function
        - param func: system dynamic
        - param x0: system state
        - param action: control input
        - param h: time of sample
        return: state of next time
        """
        k1 = func(x0, action)
        k2 = func(x0 + h * k1 / 2, action)
        k3 = func(x0 + h * k2 / 2, action)
        k4 = func(x0 + h * k3, action)
        
        x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x1



class SpringSys(UnlinearSystem):
    
    def __init__(self, param):
        super().__init__(param)
        self.m = param['m']
        self.k = param['k']
        self.start_state_low_bound  = -param['start_state_bound']
        self.start_state_high_bound = param['start_state_bound']
        self.x_threshold            = param['x_threshold']
        
        self.A = np.array([[0.0,1.0],
                           [0.0,0.0]])
        
        self.A1 = np.array([[0.0],[1.0]])

        self.A2 = np.array([[      0.0          , 0.0],
                            [-self.k / self.m   , 0.0]])
        
        self.B  = np.array([[0.0],[ 1 / self.m]])
        
        
    
    def dynamic(self, state, u):
        
        state = self.state

        M1 = np.matmul(self.A, state.T)
        M2 = np.matmul(np.matmul(self.A2, state.T), np.matmul(state, self.A))
        M2 = np.matmul(M2, self.A1) * (abs(state[0][0]) / state[0][0])
        M3 = np.matmul(self.B, u)

        state_dot = M1 + M2 + M3

        return state_dot.T      # 1 * 2

    def reset(self):
        # Initialize
        self.iter_counter = 0
        self.state = np.random.uniform(low = self.start_state_low_bound, high=self.start_state_high_bound, size=(1, 2))
        self.state[0][0] = 4.0
        self.state[0][1] = 0.0

        return self.state


    def step(self, action):
        
        others = {}
        self.iter_counter += 1
        x = self.state[0][0]
        
        next_state = self.rk4(self.dynamic, self.state, action, self.delta_T)
        self.state = next_state.copy()

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or self.iter_counter > 1000
        )

        reward = 1.0 - abs(x)

        return self.state, reward, terminated, others


class Manipulator(UnlinearSystem):

    def __init__(self, param):
        super().__init__(param)
        self.m1 = param['m1']
        self.m2 = param['m2']
        self.l1 = param['l1']
        self.l2 = param['l2']
        self.l_c1 = param['l_c1']
        self.l_c2 = param['l_c2']
        self.dT = param['dT']

        self.I1 = 1.0 / 3.0 * self.m1 * self.l1**2
        self.I2 = 1.0 / 3.0 * self.m2 * self.l2**2

        self.F1 = np.array([[np.random.normal(0.5, 0.04), 0],
                            [0,  np.random.normal(0.8,0.04)]])
        self.F2 = np.array([[np.random.normal(0.8, 0.04), 0],
                            [0,  np.random.normal(0.5,0.04)]])
        self.F3 = np.array([[np.random.normal(0.5, 0.04), 0],
                            [0,  np.random.normal(0.5,0.04)]])
        
        self.reset()
        self.SystemTest = False
    
    def dynamic(self, state, u):
        # state --> 1*4   u --> 1*2
        theta1 = state[0][0]
        theta2 = state[0][1]
        d_theta1 = state[0][2]
        d_theta2 = state[0][3]

        H11 = self.m1 * self.l_c1**2 + self.I1 + self.m2 * (self.l1**2 + self.l_c2**2 + 2*self.l1*self.l_c2*cos(theta2)) + self.I2
        H12 = self.m2 * (self.l_c2**2 + self.l1 * self.l_c2 * cos(theta2)) + self.I2
        H21 = H12
        H22 = self.m2 * self.l_c2**2 + self.I2
        h = self.m2 * self.l1 * self.l_c2 * sin(theta2)
        X = np.array([[-2*d_theta2 , -d_theta2],
                      [d_theta1    ,     0    ]])
        H = np.array([[H11, H12],
                      [H21, H22]])
        A22 = -h * np.matmul(np.linalg.inv(H), X) # h*H^{-1}*X

        A11_12 = np.array([[0,0,1,0],
                           [0,0,0,1]])
        A21_22 = np.hstack((np.zeros((2,2)), A22))
        A = np.vstack((A11_12, A21_22))                             # A --> 4*4
        
        Bu = np.vstack((np.zeros((2,1)), np.matmul(np.linalg.inv(H), u.T)))     # Bu --> 4*1

        state_dot = np.matmul(A, state.T) + Bu 

        return state_dot.T      # output --> 1*4 state

    def reset(self):
        
        self.iter_counter = 0
        self.state = np.array([[PI / 2.0, PI / 2.0, 0.0, 0.0]])
        return self.state

    def step(self, u):

        others = {}
        self.iter_counter += 1
        next_state = self.rk4(self.dynamic, self.state, u, self.dT)
        self.state = next_state.copy()

        [theta1, theta2, d_theta1, d_theta2] = next_state[0]

        terminated = bool(
            theta1 <  (-PI)
            or theta1 > PI
            or theta2 < (-PI)
            or theta2 > PI
            or self.iter_counter > 2000
        )

        reward = 1.0 - np.linalg.norm(self.state[0,0:1], ord = 2)

        return self.state, reward, terminated, others
    
