from P4Vector import *

import ops

NCOMB = 1 #only consider 1 combination

opt = tf.keras.optimizers.Adam(
    learning_rate=0.1,
)

met = PtPhiEtaMassVector.makeTensor('met',NCOMB)
lepton = PtPhiEtaMassVector.makeTensor('lepton',NCOMB)
bjet = PtPhiEtaMassVector.makeTensor('bjet',NCOMB)
bquark = PtPhiEtaMassVector.makeTensor('bquark',NCOMB)
neutrino = PtPhiEtaMassVector.makeTensor('neutrino',NCOMB)


wboson = lepton+neutrino
wboson.setName("wboson")

top = wboson+bquark
top.setName("top")

#neutrino-met resolution
chi2 = ops.square((met.pt()-neutrino.pt())/10.)
chi2 += ops.square((met.phi()-neutrino.phi())/0.2)

#bjet-bquark resolution
chi2 += ops.square((bjet.pt()-bquark.pt())/50.)
chi2 += ops.square((bjet.phi()-bquark.phi())/0.2)
chi2 += ops.square((bjet.eta()-bquark.eta())/0.2)

#mass constraints
chi2 += ops.square((wboson.mass()-80.4)/1.)
chi2 += ops.square((top.mass()-172.5)/2.)

chi2 = ops.reduce_sum(chi2)

events = [
    {
        'met': np.array([[100.,0.3,0.0,0.0]],dtype=np.float32),
        'lepton':np.array([[50.,-0.3,0.5,0.0]],dtype=np.float32),
        'bjet':np.array([[150.,-1.3,1.5,0.0]],dtype=np.float32),
    },
    {
        'met': np.array([[50.,0.6,0.0,0.0]],dtype=np.float32),
        'lepton':np.array([[150.,-1.1,-0.2,0.0]],dtype=np.float32),
        'bjet':np.array([[80.,0.3,-0.5,0.0]],dtype=np.float32),
    },
]


for event in events:
    met.setPtPhiEtaMass(event['met'])
    neutrino.setPtPhiEtaMass(event['met']) #set to the same value as starting condition
    
    lepton.setPtPhiEtaMass(event['lepton'])
    
    bjet.setPtPhiEtaMass(event['bjet'])
    bquark.setPtPhiEtaMass(event['bjet']) #set to the same value as starting condition
    
    for i in range(100):
        opt.minimize(chi2,[
                neutrino.tensor(), #minimize wrt unknown neutrino
                bquark.tensor()  #minimize wrt unknown bquark
        ])
        print (i,'chi2',chi2().numpy())
    print ("-"*50)
    print (neutrino)
    print (wboson)
    print (bquark)
    print (top)
    print()
    

