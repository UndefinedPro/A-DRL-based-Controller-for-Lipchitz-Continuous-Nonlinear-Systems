from Train import TRAIN_AL_MANIPULATOR


netname = 'Lyp'
t = TRAIN_AL_MANIPULATOR(netname, 2000)

path = './param/AL/Manipulator_param_in_paper/' + netname + 'actorNet0.pkl' 
t.Test(path)
t.PlotTestData()    # The default path points to the data in the manuscript. 
