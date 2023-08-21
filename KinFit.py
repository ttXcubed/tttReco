from P4Vector import *
from scipy.optimize import fsolve
import ops
import uproot
import awkward as ak
import itertools
import matplotlib.pyplot as plt 
import tensorflow as tf
import math 
import warnings

opt = tf.keras.optimizers.Adam(
	learning_rate=0.1,
)

NCOMB = 1 #only consider 1 combination

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

zprime1 = topl1+topl2					#only one of the Z prime tensors is the real one. The double leptonic decay shouldn't come from the Z' boson
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

dir_samples = '/nfs/dust/cms/user/mkomm/ttx3/KinFit/Samples_new/TopPhilic_ST_m750_relWidth4_TuneCP5_13TeV-madgraph-pythia8/'

f = uproot.concatenate((dir_samples + 'ttv_nano_1_Friend.root', dir_samples + 'ttv_nano_2_Friend.root', dir_samples + 'ttv_nano_3_Friend.root', dir_samples + 'ttv_nano_4_Friend.root', dir_samples + 'ttv_nano_5_Friend.root'))

#Event selection: 3 W bosons, 2 leptons with the same charge, 3 matching top quarks
nttt_reco_lepton = f['nttt_reco_lepton']
matched = f['ttt_gen_top_matched']
ttt_gen_nwbosons = f['ttt_gen_nwbosons']
lepcharge = f['ttt_reco_lepton_charge']

list1 = np.where(nttt_reco_lepton==2)[0]
list2 = np.where(ttt_gen_nwbosons==3)[0] 
list3 = np.where(ak.sum(matched, axis=1)==3)[0]
list4 = np.where((ak.sum(lepcharge, axis=1)==2) | (ak.sum(lepcharge, axis=1)==-2))[0]
list5 = list(set(list1).intersection(list2))
list6 = list(set(list5).intersection(list3))
dilep = np.sort(list(set(list6).intersection(list4)))

#properties of met
met_pt = f['ttt_reco_met_pt'][dilep]
met_phi = f['ttt_reco_met_phi'][dilep]

#properties of the leptons
lep_reco_idx = f['ttt_reco_lepton_top_idx'][dilep]
lep_reco_pt = f['ttt_reco_lepton_pt'][dilep]
lep_reco_pt = lep_reco_pt[lep_reco_idx>=0]

lep_reco_eta = f['ttt_reco_lepton_eta'][dilep]
lep_reco_eta = lep_reco_eta[lep_reco_idx>=0]

lep_reco_phi = f['ttt_reco_lepton_phi'][dilep]
lep_reco_phi = lep_reco_phi[lep_reco_idx>=0]

#properties of the b jets
b_reco_idx = f['ttt_reco_bjet_top_idx'][dilep]
b_reco_pt = f['ttt_reco_bjet_pt'][dilep]
b_reco_pt = b_reco_pt[b_reco_idx>=0]

b_reco_eta = f['ttt_reco_bjet_eta'][dilep]
b_reco_eta = b_reco_eta[b_reco_idx>=0]

b_reco_phi = f['ttt_reco_bjet_phi'][dilep]
b_reco_phi = b_reco_phi[b_reco_idx>=0]

#properties of the non-b quark jets
jet_reco_idx = f['ttt_reco_jet_top_idx'][dilep]
jet_reco_pt = f['ttt_reco_jet_pt'][dilep]
jet_reco_pt = jet_reco_pt[jet_reco_idx>=0]

jet_reco_eta = f['ttt_reco_jet_eta'][dilep]
jet_reco_eta = jet_reco_eta[jet_reco_idx>=0]

jet_reco_phi = f['ttt_reco_jet_phi'][dilep]
jet_reco_phi = jet_reco_phi[jet_reco_idx>=0]

nevents = len(dilep)		#max nevents = len(dilep)
events = [0]*nevents
for i in range(nevents):
	events[i] = {
		'met': np.array([[met_pt[i], met_phi[i], 0.0, 0.0]], dtype=np.float32),
		'lepton': np.array([[lep_reco_pt[i][j], lep_reco_phi[i][j], lep_reco_eta[i][j], 0.0] for j in range(len(lep_reco_pt[i]))], dtype=np.float32),
		'bjet': np.array([[b_reco_pt[i][j], b_reco_phi[i][j], b_reco_eta[i][j], 4.7] for j in range(len(b_reco_pt[i]))], dtype=np.float32),  #Should I include a 4.7 GeV mass for b jets?
		'qjet': np.array([[jet_reco_pt[i][j], jet_reco_phi[i][j], jet_reco_eta[i][j], 0.0] for j in range(len(jet_reco_pt[i]))], dtype=np.float32)
	}

#permutations of the b jets
list012 = list(itertools.permutations([0, 1, 2]))

nmasses = 3*nevents
masses = [0]*nmasses
idx_masses = 0
filter_masses = [0]*nevents
filter_masses2 = [0]*nevents

for event in events:
	print('Event', int(idx_masses/3+1), 'out of', nevents, end='\r')
	best_sum = 1e10
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
		
		#function with the constraints to find the best values for the neutrinos
		def equations(p):
			pnu1x, pnu1y, pnu1z, pnu2x, pnu2y, pnu2z = p
			pnu1 = np.sqrt(pnu1x**2+pnu1y**2+pnu1z**2)
			pnu2 = np.sqrt(pnu2x**2+pnu2y**2+pnu2z**2)
			#Mass of the W boson
			eq1 = (lepton1.p()().numpy()[0]+pnu1)**2-(lepton1.px()().numpy()[0]+pnu1x)**2-(lepton1.py()().numpy()[0]+pnu1y)**2-(lepton1.pz()().numpy()[0]+pnu1z)**2-80.4**2
			eq2 = (lepton2.p()().numpy()[0]+pnu2)**2-(lepton2.px()().numpy()[0]+pnu2x)**2-(lepton2.py()().numpy()[0]+pnu2y)**2-(lepton2.pz()().numpy()[0]+pnu2z)**2-80.4**2
			#Mass of the top quark
			eq3 = (lepton1.p()().numpy()[0]+pnu1+bjet1.energy()().numpy()[0])**2-(lepton1.px()().numpy()[0]+pnu1x+bjet1.px()().numpy()[0])**2-(lepton1.py()().numpy()[0]+pnu1y+bjet1.py()().numpy()[0])**2-(lepton1.pz()().numpy()[0]+pnu1z+bjet1.pz()().numpy()[0])**2-172.5**2
			eq4 = (lepton2.p()().numpy()[0]+pnu2+bjet2.energy()().numpy()[0])**2-(lepton2.px()().numpy()[0]+pnu2x+bjet2.px()().numpy()[0])**2-(lepton2.py()().numpy()[0]+pnu2y+bjet2.py()().numpy()[0])**2-(lepton2.pz()().numpy()[0]+pnu2z+bjet2.pz()().numpy()[0])**2-172.5**2
			#Components of the missing energy
			eq5 = pnu1x+pnu2x-met.px()().numpy()[0]
			eq6 = pnu1y+pnu2y-met.py()().numpy()[0]
			return (eq1, eq2, eq3, eq4, eq5, eq6)
	
		warnings.filterwarnings('ignore', 'The iteration is not making good progress')
		pnu1x, pnu1y, pnu1z, pnu2x, pnu2y, pnu2z =  fsolve(equations, (329.7, 0.1, 0.6, 326.4, 0.1, 0.6))
	
		if np.sum(equations((pnu1x, pnu1y, pnu1z, pnu2x, pnu2y, pnu2z))) < best_sum:
			best_sum = np.sum(equations((pnu1x, pnu1y, pnu1z, pnu2x, pnu2y, pnu2z)))
			best_perm = perm
			best_pnu1x = pnu1x
			best_pnu1y = pnu1y
			best_pnu1z = pnu1z
			best_pnu2x = pnu2x
			best_pnu2y = pnu2y
			best_pnu2z = pnu2z

	bjet1.setPtPhiEtaMass(event['bjet'][best_perm[0]].reshape((1,4)))
	bquark1.setPtPhiEtaMass(event['bjet'][best_perm[0]].reshape((1,4))) #set to the same value as starting condition
	bjet2.setPtPhiEtaMass(event['bjet'][best_perm[1]].reshape((1,4)))
	bquark2.setPtPhiEtaMass(event['bjet'][best_perm[1]].reshape((1,4))) #set to the same value as starting condition
	bjet3.setPtPhiEtaMass(event['bjet'][best_perm[2]].reshape((1,4)))
	bquark3.setPtPhiEtaMass(event['bjet'][best_perm[2]].reshape((1,4))) #set to the same value as starting condition
		
	ptnu1 = np.sqrt(best_pnu1x**2+best_pnu1y**2)
	etanu1 = np.arcsinh(best_pnu1z/ptnu1)
	if best_pnu1x>0:
		phinu1 = np.arctan(best_pnu1y/best_pnu1x)
	elif best_pnu1y>=0:
		phinu1 = np.arctan(best_pnu1y/best_pnu1x)+np.pi
	else:
		phinu1 = np.arctan(best_pnu1y/best_pnu1x)-np.pi
		
	ptnu2 = np.sqrt(best_pnu2x**2+best_pnu2y**2)
	etanu2 = np.arcsinh(best_pnu2z/ptnu2)
	if best_pnu2x>0:
		phinu2 = np.arctan(best_pnu2y/best_pnu2x)
	elif best_pnu2y>=0:
		phinu2 = np.arctan(best_pnu2y/best_pnu2x)+np.pi
	else:
		phinu2 = np.arctan(best_pnu2y/best_pnu2x)-np.pi
			
	neutrino1.setPtPhiEtaMass(np.array([[ptnu1, phinu1, etanu1, 0.0]], dtype=np.float32)) #set to the same value as starting condition
	neutrino2.setPtPhiEtaMass(np.array([[ptnu2, phinu2, etanu2, 0.0]], dtype=np.float32))
		
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
	filter_masses[int(idx_masses/3)] = np.min([zprime1.mass()().numpy()[0], zprime2.mass()().numpy()[0], zprime3.mass()().numpy()[0]])
	filter_masses2[int(idx_masses/3)] = np.max([zprime1.mass()().numpy()[0], zprime2.mass()().numpy()[0], zprime3.mass()().numpy()[0]])
	idx_masses = idx_masses+3

print()

np.save('masses', masses)
