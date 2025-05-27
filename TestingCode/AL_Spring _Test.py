from Train import TRAIN_AL_SPRING


netname = 'Lyp'
t = TRAIN_AL_SPRING(netname, 2000)

for i in range(10):
    path = './param/AL/Spring_param_in_paper/' + netname + 'actorNet' + str(i) + '.pkl' 
    t.Test(path)
