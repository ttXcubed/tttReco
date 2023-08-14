from P4Vector import *

import ops
import uproot
import awkward as ak
import itertools
import matplotlib.pyplot as plt

NCOMB = 1 #only consider 1 combination

opt = tf.keras.optimizers.Adam(
	learning_rate=0.1,
)

met = PtPhiEtaMassVector.makeTensor('met',NCOMB)
neutrino1 = PtPhiEtaMassVector.makeTensor('neutrino1',NCOMB)
neutrino2 = PtPhiEtaMassVector.makeTensor('neutrino2',NCOMB)
lepton1 = PtPhiEtaMassVector.makeTensor('lepton1',NCOMB)
lepton2 = PtPhiEtaMassVector.makeTensor('lepton2',NCOMB)
quark1 = PtPhiEtaMassVector.makeTensor('quark1',NCOMB)
quark2 = PtPhiEtaMassVector.makeTensor('quark2',NCOMB)
bjet1 = PtPhiEtaMassVector.makeTensor('bjet1',NCOMB)
bjet2 = PtPhiEtaMassVector.makeTensor('bjet2',NCOMB)
bjet3 = PtPhiEtaMassVector.makeTensor('bjet3',NCOMB)
bquark1 = PtPhiEtaMassVector.makeTensor('bquark1',NCOMB)
bquark2 = PtPhiEtaMassVector.makeTensor('bquark2',NCOMB)
bquark3 = PtPhiEtaMassVector.makeTensor('bquark3',NCOMB)

wbosonl1 = lepton1+neutrino1
wbosonl1.setName('wbosonl1')
wbosonl2 = lepton2+neutrino2
wbosonl2.setName('wbosonl2')
wbosonh = quark1+quark2
wbosonh.setName('wbosonh')

topl1 = wbosonl1+bquark1
topl1.setName('topl1')
topl2 = wbosonl2+bquark2
topl2.setName('topl2')
toph = wbosonh+bquark3
toph.setName('toph')

zprime1 = topl1+topl2					#only one of the Z prime tensors is the real one
zprime1.setName('zprime1')
zprime2 = topl1+toph
zprime2.setName('zprime2')
zprime3 = topl2+toph
zprime3.setName('zprime3')

#transverse momentum conservation constraints
chi2 = ops.square((met.px()-neutrino1.px()-neutrino2.px())/10.)
chi2 += ops.square((met.py()-neutrino1.py()-neutrino2.py())/10.)

#bjet-bquark resolution
chi2 += ops.square((bjet1.pt()-bquark1.pt())/50.)
chi2 += ops.square((bjet1.phi()-bquark1.phi())/0.2)
chi2 += ops.square((bjet1.eta()-bquark1.eta())/0.2)
chi2 += ops.square((bjet2.pt()-bquark2.pt())/50.)
chi2 += ops.square((bjet2.phi()-bquark2.phi())/0.2)
chi2 += ops.square((bjet2.eta()-bquark2.eta())/0.2)
chi2 += ops.square((bjet3.pt()-bquark3.pt())/50.)
chi2 += ops.square((bjet3.phi()-bquark3.phi())/0.2)
chi2 += ops.square((bjet3.eta()-bquark3.eta())/0.2)

#mass constraints
chi2 += ops.square((wbosonl1.mass()-80.4)/1.)
chi2 += ops.square((wbosonl2.mass()-80.4)/1.)
chi2 += ops.square((wbosonh.mass()-80.4)/1.)
chi2 += ops.square((topl1.mass()-172.5)/2.)
chi2 += ops.square((topl2.mass()-172.5)/2.)
chi2 += ops.square((toph.mass()-172.5)/2.)

chi2 = ops.reduce_sum(chi2)

f = uproot.open('/nfs/dust/cms/user/mkomm/ttx3/KinFit/Samples_new/TopPhilic_ST_m750_relWidth4_TuneCP5_13TeV-madgraph-pythia8/ttv_nano_1_Friend.root')

#Event selection: 3 W bosons, 2 leptons, 3 matching top quarks
nttt_reco_lepton = f['Friends']['nttt_reco_lepton'].array(library='np')
matched = f['Friends']['ttt_gen_top_matched'].array(library='ak')
ttt_gen_nwbosons = f['Friends']['ttt_gen_nwbosons'].array(library='np')

list1 = np.where(nttt_reco_lepton==2)[0]
list2 = np.where(ttt_gen_nwbosons==3)[0] 
list3 = np.where(ak.sum(matched, axis=1)==3)[0]
list4 = list(set(list1).intersection(list2))
dilep = np.sort(list(set(list3).intersection(list4)))

#properties of met
met_pt = f['Friends']['ttt_reco_met_pt'].array(library='np')[dilep]
met_phi = f['Friends']['ttt_reco_met_phi'].array(library='np')[dilep]

#properties of the leptons
lep_reco_idx = f['Friends']['ttt_reco_lepton_top_idx'].array(library='ak')[dilep]
lep_reco_pt = f['Friends']['ttt_reco_lepton_pt'].array(library='ak')[dilep]
lep_reco_pt = lep_reco_pt[lep_reco_idx>=0]

lep_reco_eta = f['Friends']['ttt_reco_lepton_eta'].array(library='ak')[dilep]
lep_reco_eta = lep_reco_eta[lep_reco_idx>=0]

lep_reco_phi = f['Friends']['ttt_reco_lepton_phi'].array(library='ak')[dilep]
lep_reco_phi = lep_reco_phi[lep_reco_idx>=0]

#properties of the b jets
b_reco_idx = f['Friends']['ttt_reco_bjet_top_idx'].array(library='ak')[dilep]
b_reco_pt = f['Friends']['ttt_reco_bjet_pt'].array(library='ak')[dilep]
b_reco_pt = b_reco_pt[b_reco_idx>=0]

b_reco_eta = f['Friends']['ttt_reco_bjet_eta'].array(library='ak')[dilep]
b_reco_eta = b_reco_eta[b_reco_idx>=0]

b_reco_phi = f['Friends']['ttt_reco_bjet_phi'].array(library='ak')[dilep]
b_reco_phi = b_reco_phi[b_reco_idx>=0]

#properties of the non-b quark jets
jet_reco_idx = f['Friends']['ttt_reco_jet_top_idx'].array(library='ak')[dilep]
jet_reco_pt = f['Friends']['ttt_reco_jet_pt'].array(library='ak')[dilep]
jet_reco_pt = jet_reco_pt[jet_reco_idx>=0]

jet_reco_eta = f['Friends']['ttt_reco_jet_eta'].array(library='ak')[dilep]
jet_reco_eta = jet_reco_eta[jet_reco_idx>=0]

jet_reco_phi = f['Friends']['ttt_reco_jet_phi'].array(library='ak')[dilep]
jet_reco_phi = jet_reco_phi[jet_reco_idx>=0]

nevents = len(dilep)		#max nevents = len(dilep)
events = [0]*nevents
for i in range(nevents):
	events[i] = {
		'met': np.array([[met_pt[i], met_phi[i], 0.0, 0.0]], dtype=np.float32),
		'lepton': np.array([[lep_reco_pt[i][j], lep_reco_phi[i][j], lep_reco_eta[i][j], 0.0] for j in range(len(lep_reco_pt[i]))], dtype=np.float32),
		'bjet': np.array([[b_reco_pt[i][j], b_reco_phi[i][j], b_reco_eta[i][j], 0.0] for j in range(len(b_reco_pt[i]))], dtype=np.float32),  #Should I include a 4.7 GeV mass for b jets?
		'qjet': np.array([[jet_reco_pt[i][j], jet_reco_phi[i][j], jet_reco_eta[i][j], 0.0] for j in range(len(jet_reco_pt[i]))], dtype=np.float32)
	}

#permutations of the b jets
list012 = list(itertools.permutations([0, 1, 2]))

nmasses = 18*nevents
masses = [0]*nmasses
idx_masses = 0

for event in events:
	for perm in list012:
		met.setPtPhiEtaMass(event['met'])
		neutrino1.setPtPhiEtaMass(event['met']) #set to the same value as starting condition
		neutrino2.setPtPhiEtaMass(event['met'])
    
		lepton1.setPtPhiEtaMass(event['lepton'][0].reshape((1,4)))
		lepton2.setPtPhiEtaMass(event['lepton'][1].reshape((1,4)))
    
		quark1.setPtPhiEtaMass(event['qjet'][0].reshape((1,4)))
		quark2.setPtPhiEtaMass(event['qjet'][1].reshape((1,4)))
    
		bjet1.setPtPhiEtaMass(event['bjet'][perm[0]].reshape((1,4)))
		bquark1.setPtPhiEtaMass(event['bjet'][perm[0]].reshape((1,4))) #set to the same value as starting condition
		bjet2.setPtPhiEtaMass(event['bjet'][perm[1]].reshape((1,4)))
		bquark2.setPtPhiEtaMass(event['bjet'][perm[1]].reshape((1,4))) #set to the same value as starting condition
		bjet3.setPtPhiEtaMass(event['bjet'][perm[2]].reshape((1,4)))
		bquark3.setPtPhiEtaMass(event['bjet'][perm[2]].reshape((1,4))) #set to the same value as starting condition
    
		for i in range(50):		#this can be changed to increase the number of iterations
			opt.minimize(chi2,[
				neutrino1.tensor(), #minimize wrt unknown neutrino
				neutrino2.tensor(),
				bquark1.tensor(),  #minimize wrt unknown bquark
				bquark2.tensor(),
				bquark3.tensor(),
			])
		masses[idx_masses] = zprime1.mass()().numpy()[0]
		masses[idx_masses+1] = zprime2.mass()().numpy()[0]
		masses[idx_masses+2] = zprime3.mass()().numpy()[0]
		idx_masses = idx_masses+3

plt.figure()
plt.hist(masses, bins=20)
plt.savefig('hist.png')

