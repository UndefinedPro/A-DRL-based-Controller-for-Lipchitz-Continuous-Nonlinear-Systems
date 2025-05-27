from Train import TRAIN_MANIPULATOR


netname = 'DDPG'
# ===================== Train DDPG =====================
t = TRAIN_MANIPULATOR(netname, 2000)

# Train DDPG with spring system 
t.Train()

# Test spring-system controller trained by DDPG
path = './param/DDPG/Temp_param/' + netname + 'actorNet' + str(0) + '.pkl' 
t.Test(path)

# The parameters in the manuscrpt
# path = './param/DDPG/Manipulator_param_in_paper/' + netname + 'actorNet' + str(0) + '.pkl' 
# t.Test(path)