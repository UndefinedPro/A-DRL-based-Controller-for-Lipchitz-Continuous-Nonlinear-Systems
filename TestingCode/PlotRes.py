from ExcelHandler import ExcelReader as er
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def TrainingDataPlot(DataPath):
    
    # reward 
    x = er()
    x.LoadFile("./Data/expV/Lyp_reward.xlsx")

    col = x.DataFrame.shape[1]
    row = x.DataFrame.shape[0]

    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.85,0.85])
    ax2 = fig.add_axes([0.6,0.6,0.3,0.3])
    ax1.set_xlabel('Trajectory')
    ax1.set_ylabel('Total reward')
    ax2.set_xlabel('Trajectory')
    ax2.set_ylabel('Total reward')

    for j in range(1,col):
        ax1.plot(x.DataFrame.iloc[:,0], x.DataFrame.iloc[:,j])
        ax2.plot(x.DataFrame.iloc[(row-100):,0], x.DataFrame.iloc[(row-100):,j], label="details")
    
    plt.show()
    
    # step
    x = er()
    x.LoadFile("./Data/expV/Lyp_step.xlsx")

    col = x.DataFrame.shape[1]
    row = x.DataFrame.shape[0]

    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.85,0.85])
    ax1.set_xlabel('Trajectory')
    ax1.set_ylabel('Total step')

    for j in range(1,col):
        ax1.plot(x.DataFrame.iloc[:,0], x.DataFrame.iloc[:,j])

    plt.show()
    

def TenNetResPlot(DataPath):
    reader = er()
    reader.LoadFile(DataPath)

    row = reader.DataFrame.shape[0]
    col = reader.DataFrame.shape[1]

    axes1 = plt.figure()
    # plt.title('10 trained network control performance')
    plt.ylabel('x(m)')
    plt.xlabel('step')
    # plt.ylim(-0.5, 4.8)
    plt.axhline(0.5, color='red', linestyle='--')
    for i in range(col):
        plt.plot(np.linspace(1,1000,1000), reader.DataFrame.iloc[:,i], lw='1')
    plt.show()
    plt.savefig('./SimRes/AllStateResults.eps')
    plt.close()


def OneNetPlot():

    net_number = 0

    x_reader = er()
    v_reader = er()
    u_reader = er()
    x_reader.LoadFile('./Data/expV/OneNet/Lyp_x.xlsx')
    v_reader.LoadFile('./Data/expV/OneNet/Lyp_v.xlsx')
    u_reader.LoadFile('./Data/expV/OneNet/Lyp_u.xlsx')

    plt.figure(figsize=(10,3))
    axes0 = plt.gca()
    plt.ylabel('x(m)')
    plt.xlabel('step')
    axes0.xaxis.set_label_coords(1.01, -0.05)
    plt.ylim(-4.8, 4.8)
    plt.plot(np.linspace(1,1000,1000), x_reader.DataFrame.iloc[:,net_number],lw='1')
    plt.axhline(0.0, color='red', linestyle='--')
    plt.savefig('./SimRes/StateResult_x.eps')
    plt.show()
    plt.close()

    plt.figure(figsize=(10,3))
    axes1 = plt.gca()
    plt.ylabel('v(m/s)')
    plt.xlabel('step')
    axes1.xaxis.set_label_coords(1.01, -0.05)
    plt.plot(np.linspace(1,1000,1000), v_reader.DataFrame.iloc[:,net_number],lw='1')
    plt.axhline(0.0, color='red', linestyle='--')
    plt.savefig('./SimRes/StateResult_v.eps')
    plt.show()
    plt.close()

    plt.figure(figsize=(10,3))
    axes2 = plt.gca()
    plt.ylabel('u(N)')
    plt.xlabel('step')
    axes2.xaxis.set_label_coords(1.01, -0.05)
    plt.plot(np.linspace(1,1000,1000), u_reader.DataFrame.iloc[:,net_number],lw='1')
    plt.axhline(0.0, color='red', linestyle='--')
    plt.savefig('./SimRes/InputResult_u.eps')
    plt.show()
    plt.close()


def ManipulatorPlot():

    net_number = 0

    reward_reader = er()
    step_reader = er()
    theta1_reader = er()
    theta2_reader = er()
    reward_reader.LoadFile('./Result_2D/Final/Manipulator_Data/reward.xlsx')
    step_reader.LoadFile('./Result_2D/Final/Manipulator_Data/step.xlsx')
    theta1_reader.LoadFile('./Result_2D/Final/Manipulator_Data/theta1.xlsx')
    theta2_reader.LoadFile('./Result_2D/Final/Manipulator_Data/theta2.xlsx')

    plt.figure()
    axes0 = plt.gca()
    plt.ylabel('Total value')
    plt.xlabel('Trajectory')
    plt.plot(np.linspace(1,1999,1999), reward_reader.DataFrame.iloc[:,net_number],lw='1')
    plt.savefig('./SimRes/Manipulator_reward.eps')
    plt.show()
    plt.close()

    plt.figure()
    axes0 = plt.gca()
    plt.ylabel('Total step')
    plt.xlabel('Trajectory')
    plt.plot(np.linspace(1,1999,1999), step_reader.DataFrame.iloc[:,net_number],lw='1')
    plt.savefig('./SimRes/Manipulator_step.eps')
    plt.show()
    plt.close()

    plt.figure()
    axes1 = plt.gca()
    plt.ylabel(r'$\eta_1(rad)$')
    plt.xlabel('step')
    plt.plot(np.linspace(1,2000,2000), theta1_reader.DataFrame.iloc[:,net_number],lw='1')
    plt.axhline(0.0, color='red', linestyle='--')
    plt.savefig('./SimRes/Manipulator_theta1.eps')
    plt.show()
    plt.close()

    plt.figure()
    axes2 = plt.gca()
    plt.ylabel(r'$\eta_2(rad)$')
    plt.xlabel('step')
    plt.plot(np.linspace(1,2000,2000), theta2_reader.DataFrame.iloc[:,net_number],lw='1')
    plt.axhline(0.0, color='red', linestyle='--')
    plt.savefig('./SimRes/Manipulator_theta2.eps')
    plt.show()
    plt.close()






if __name__ == "__main__":
    # TrainingDataPlot("./Data/expV/Lyp_reward.xlsx")
    # TrainingDataPlot("./Data/expV/Lyp_step.xlsx")
    # TenNetResPlot('./Data/expV/Lyp_x.xlsx')
    OneNetPlot()