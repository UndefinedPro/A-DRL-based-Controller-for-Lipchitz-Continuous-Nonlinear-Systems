# Introcution



## Enviroment

python 3.10.14
pytorch 2.1.0 -cpu

other necessary library:

openpyxl (which is used to store data)
numpy
pandas

## File Specification

- DRL.py:          Define the structure of network and the update strategy of DDPG
- ExcelHandler.py: A file to handle Excel, some functions are integrated in this file
- Train.py:        Train process of DDPG
- NonlinearSys.py: Dynamics of nonlinear systems 

- AL_Manipulator_Test.py:       You can test the AL method under Manipulator system
- AL_Spring _Test.py:           You can test the AL method under Spring system
- DDPG_Manipulator_Train.py:    You can train/test the DDPG method under Manipulator system    
- DDPG_Spring_Train.py:         You can train/test the DDPG method under Spring system
