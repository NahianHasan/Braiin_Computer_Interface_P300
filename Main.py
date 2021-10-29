import os,glob
import principal_component_analysis as PCA
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import datetime,math
import data_read as DR
from collections import Counter
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
import argparse


def filter_design(f_high, f_low,transition,fs):
	N = int(np.ceil(1./transition * fs)) #filter length
	h = signal.firwin(N, [f_high, f_low], pass_zero=False, fs=fs)
	t_h = np.arange(0, N) / fs
	plt.subplot(121)
	plt.plot(t_h, h)
	plt.xlabel('Time (s)')
	plt.ylabel('h(t)')
	H=fft(h)
	f = np.fft.fftfreq(len(h),d=1/fs)  # Frequency axis in Hz
	plt.subplot(122)
	plt.plot(f,np.abs(H))
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('H(f)')
	plt.xlim([-15, 15])
	#plt.show()
	plt.savefig('bandpass_filter.png')
	return h,H

def filtered_events(event_data,fs,h,save_path,eeg_channels,verbose=0):
	filtered_data = np.zeros(event_data.shape)
	for i in range(event_data.shape[0]):
		eeg_filt_firwin = signal.filtfilt(h, 1, event_data[i,:],axis=0)
		if verbose:
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			figure = plt.figure()
			t = np.arange(0,event_data.shape[1]/fs,1./fs)
			ax1 = plt.subplot(211)
			ax1.plot(t[0:102400], event_data[i,0:102400])#Plot the first 50s data
			plt.ylabel(eeg_channels[i]+'(re-referenced (raw))')
			ax2 = plt.subplot(212, sharex=ax1)
			ax2.plot(t[0:102400], eeg_filt_firwin[0:102400])#Plot the first 50s data
			plt.xlabel('Time (s)')
			plt.ylabel(eeg_channels[i]+'(re-referenced(filtered))')
			#plt.show()
			plt.savefig(save_path+'/'+'filterred_signal_'+eeg_channels[i]+'.png')
			plt.close()
		filtered_data[i,:] = eeg_filt_firwin
	#print(filtered_data.shape)
	return filtered_data

def extract_epochs(events,data,epoch_length,eeg_channels,fs):
	epoch_results = np.zeros((len(events),data.shape[0],int(epoch_length)))
	events = events.astype('int')
	for ev in range(len(events)):
		for i in range(data.shape[0]):
			epoch_results[ev,i,:] = data[i,events[ev]:events[ev]+epoch_length]
	return epoch_results

def extract_target_non_target(Epoch_data,Events,eeg_channels,epoch_length,baseline_DC_range,threshold):
	target_epochs = {}
	non_target_epochs = {}
	for key in Events.keys():
		target = Events[key]['target']
		stimuli = Events[key]['stimuli']
		events = Events[key]['events']
		stimu_count = Counter(stimuli.tolist())
		target_epochs[key] = {}
		non_target_epochs[key] = {}
		for ch in range(len(eeg_channels)):
			target_epochs[key][eeg_channels[ch]] = np.zeros((stimu_count[target],epoch_length))
			non_target_epochs[key][eeg_channels[ch]] = np.zeros((len(events)-stimu_count[target],epoch_length))
			targ = 0
			ntarg = 0
			for i in range(len(events)):
				epoch = Epoch_data[key][i,ch,:]-np.average(Epoch_data[key][i,ch,0:baseline_DC_range]) # after removing baseline noise
				if stimuli[i] == target:
					if not (max(np.absolute(epoch)) > threshold):
						target_epochs[key][eeg_channels[ch]][targ,:] = epoch
					targ += 1
				else:
					if not (max(np.absolute(epoch)) > threshold):
						non_target_epochs[key][eeg_channels[ch]][ntarg,:] = epoch
					ntarg += 1
	return target_epochs,non_target_epochs

def plot_session_target_non_target(targets,non_targets,eeg_channels,save_path,subject,sess,epoch_length,fs,verbose=0):
	time_axis = np.arange(0,epoch_length/fs,1/epoch_length)
	if verbose:
		for key in targets.keys():
			for ch in targets[key].keys():
				if not os.path.exists(save_path+key+'/'):
					os.makedirs(save_path+key+'/')
				#print(targets[key][ch].shape)
				#print(non_targets[key][ch].shape)
				plt.figure(figsize=(10, 10))
				plt.subplot(121)
				plt.plot(time_axis,targets[key][ch].T)
				plt.title('Targets - '+ch)
				plt.ylabel('Response (\u03bcV)')
				plt.xlabel('Time (s)')
				plt.subplot(122)
				plt.plot(time_axis,non_targets[key][ch].T)
				plt.ylabel('Response (\u03bcV)')
				plt.xlabel('Time (s)')
				plt.title('Non Targets - '+ch)
				#plt.show()
				plt.savefig(save_path+key+'/'+ch+'.png')
				plt.close()

	if not os.path.exists(save_path+'/Average_Results'):
		os.makedirs(save_path+'/Average_Results')

	keys = list(targets.keys())
	sum_trg = {}
	sum_ntrg = {}
	num_trg = {}
	num_ntrg = {}

	for ch in eeg_channels:
		avg_targets = np.zeros((targets[keys[0]][ch].shape[1],))
		trg_nums = 0
		for key in keys:
			for num_ev in range(targets[key][ch].shape[0]):
				avg_targets += targets[key][ch][num_ev,:]
				trg_nums += 1
		sum_trg[ch] = avg_targets
		num_trg[ch] = trg_nums

		avg_non_targets = np.zeros((non_targets[keys[0]][ch].shape[1],))
		ntrg_nums = 0
		for key in keys:
			for num_ev in range(non_targets[key][ch].shape[0]):
				avg_non_targets += non_targets[key][ch][num_ev,:]
				ntrg_nums += 1
		sum_ntrg[ch] = avg_non_targets
		num_ntrg[ch] = ntrg_nums

		avg_targets = avg_targets/trg_nums
		avg_non_targets = avg_non_targets/ntrg_nums
		difference = avg_targets - avg_non_targets
		plt.figure(figsize=(10, 10))
		plt.plot(time_axis,avg_targets,label='Average_Targets ('+str(trg_nums)+' trials)')
		plt.plot(time_axis,avg_non_targets,label='Average_Non_Targets ('+str(ntrg_nums)+' trials)')
		plt.plot(time_axis,difference,label='Differencce_Response')
		plt.title(subject+', '+sess+', channel '+ch)
		plt.ylabel('Average Response (\u03bcV)')
		plt.xlabel('Time (s)')
		plt.legend()
		#plt.show()
		plt.savefig(save_path+'/Average_Results/'+ch+'.png')
		plt.close()
	return sum_trg,sum_ntrg,num_trg,num_ntrg

def plot_subject_target_non_target(per_session_data,eeg_channels,save_path,subject,epoch_length,fs):
	time_axis = np.arange(0,epoch_length/fs,1/epoch_length)
	if not os.path.exists(save_path+'/Average_Results'):
		os.makedirs(save_path+'/Average_Results')
	sessions = list(per_session_data.keys())
	average_target = {}
	average_non_target = {}
	difference = {}
	total_trg = {}
	total_ntrg = {}
	for ch in eeg_channels:
		average_target[ch] = np.zeros(per_session_data[sessions[0]][0][ch].shape)
		average_non_target[ch] = np.zeros(per_session_data[sessions[0]][1][ch].shape)
		total_trg[ch] = 0
		total_ntrg[ch] = 0
		for sess in sessions:
			average_target[ch] += per_session_data[sess][0][ch]
			average_non_target[ch] += per_session_data[sess][1][ch]
			total_trg[ch] += per_session_data[sess][2][ch]
			total_ntrg[ch] += per_session_data[sess][3][ch]

		average_target[ch] = average_target[ch] / total_trg[ch]
		average_non_target[ch] = average_non_target[ch] / total_ntrg[ch]
		difference[ch] = average_target[ch] - average_non_target[ch]
		plt.figure(figsize=(10, 10))
		plt.plot(time_axis,average_target[ch],label='Average_Targets ('+str(total_trg[ch])+' trials)')
		plt.plot(time_axis,average_non_target[ch],label='Average_Non_Targets ('+str(total_ntrg[ch])+' trials)')
		plt.plot(time_axis,difference[ch],label='Differencce_Response')
		plt.title(subject+', channel '+ch)
		plt.ylabel('Average Response (\u03bcV)')
		plt.xlabel('Time (s)')
		plt.legend()
		#plt.show()
		plt.savefig(save_path+'/Average_Results/'+ch+'.png')
		plt.close()

	return average_target,average_non_target,difference,total_trg,total_ntrg

def find_all_targets_non_targets(target_epochs,non_target_epochs,eeg_channels):
	all_targets = {}
	all_non_targets = {}
	sessions = list(target_epochs.keys())
	for ch in eeg_channels:
		all_targets[ch] = list()
		all_non_targets[ch] = list()
		for sess in sessions:
			session_targets = target_epochs[sess]
			session_non_targets = non_target_epochs[sess]
			keys = list(session_targets.keys())
			for key in keys:
				for num_ev in range(session_targets[key][ch].shape[0]):
					all_targets[ch].append(session_targets[key][ch][num_ev,:])

			keys = list(session_non_targets.keys())
			for key in keys:
				for num_ev in range(session_non_targets[key][ch].shape[0]):
					all_non_targets[ch].append(session_non_targets[key][ch][num_ev,:])

		all_targets[ch] = np.array(all_targets[ch])
		all_non_targets[ch] = np.array(all_non_targets[ch])

	return all_targets,all_non_targets
def random_null_hypothesis_test(all_targets,all_non_targets,average_epoch_data,eeg_channels,subject,epoch_length,fs,save_path,data_count_factor=1,offset=0,pool_size=1000):

	P_values = {}
	time_axis = np.arange(0,epoch_length/fs,1/epoch_length)
	if not os.path.exists(save_path+'/Hypothesis_Testing/data_count_factor_'+str(data_count_factor)+'/offset_'+str(offset)):
		os.makedirs(save_path+'/Hypothesis_Testing/data_count_factor_'+str(data_count_factor)+'/offset_'+str(offset))
	G = open(save_path+'/Hypothesis_Testing/data_count_factor_'+str(data_count_factor)+'/offset_'+str(offset)+'/P_values.csv','w')
	G.write('Channel,P_value\n')


	for ch in eeg_channels:
		N = average_epoch_data[4][ch]#number of non-targets
		M = average_epoch_data[3][ch]#number of targets
		M = math.floor(M/data_count_factor)
		N = math.floor(N/data_count_factor)
		'''
		print('M = ',M)
		print('N = ',N)
		print(all_targets[ch].shape)
		print(all_non_targets[ch].shape)
		print('Number of targets = ',all_targets[ch].shape)
		print('Number of non targets = ',all_non_targets[ch].shape)
		'''
		random_differences = list()
		Peak_values = list()

		for p in range(pool_size):
			random_trg_average = np.zeros((1,all_targets[ch].shape[1]))
			random_ntrg_average = np.zeros((1,all_non_targets[ch].shape[1]))

			#Randomly choose M data indexes. Consider the data count factor
			R = np.random.choice(M+N,M,replace=False)
			for i in range(M+N):
				if i in R:
					if i < M:
						random_trg_average += all_targets[ch][i+(M*offset),:]
					else:
						random_trg_average += all_non_targets[ch][i-M+(N*offset),:]
				else:
					if i < M:
						random_ntrg_average += all_targets[ch][i+(M*offset),:]
					else:
						random_ntrg_average += all_non_targets[ch][i-M+(N*offset),:]

			random_trg_average = random_trg_average / M
			random_ntrg_average = random_ntrg_average / N
			diff = (random_trg_average - random_ntrg_average).squeeze()
			if p < 100:
				random_differences.append(diff)

			Peak_values.append(np.max(diff))

		actual_peak = max(average_epoch_data[2][ch])
		P_values[ch] = len([val for val in Peak_values if val>actual_peak])/pool_size
		G.write(ch+','+str(P_values[ch])+'\n')
		random_differences = np.array(random_differences)
		random_differences = np.reshape(random_differences,[random_differences.shape[0],random_differences.shape[-1]])

		#Plot the random average response and actual average response
		plt.figure()
		plt.plot(time_axis,random_differences[0:100,:].T,color='black')#Just plot the first 100 null examples
		plt.plot(time_axis,average_epoch_data[2][ch],color='red',linewidth=5)
		plt.title(subject+', channel '+ch)
		plt.ylabel('Average Response (\u03bcV)')
		plt.xlabel('Time (s)')
		plt.legend(['Random differences','Actual Difference'])
		#plt.show()
		plt.savefig(save_path+'/Hypothesis_Testing/data_count_factor_'+str(data_count_factor)+'/offset_'+str(offset)+'/'+ch+'.png')
		plt.close()

	#plot the p-values
	plt.figure()
	p_vals = list()
	for ch in eeg_channels:
		p_vals.append(P_values[ch])
	plt.bar(eeg_channels,p_vals)
	plt.title(subject)
	plt.ylabel('P_values')
	plt.xlabel('Channel')
	plt.xticks(rotation='vertical')
	plt.savefig(save_path+'/Hypothesis_Testing/data_count_factor_'+str(data_count_factor)+'/offset_'+str(offset)+'/P_values.png')
	plt.close()
	return P_values

def plot_subdivision_p_values(subdivisions,eeg_channels,data_count_factor,save_path,subject):
	if not os.path.exists(save_path+'/Hypothesis_Testing/data_count_factor_'+str(data_count_factor)):
		os.makedirs(save_path+'/Hypothesis_Testing/data_count_factor_'+str(data_count_factor))
	Channel_P_values = {}
	for ch in eeg_channels:
		Channel_P_values[ch] = list()
	for key in subdivisions.keys():
		for ch in subdivisions[key].keys():
			Channel_P_values[ch].append(subdivisions[key][ch])

	#plot the channelwise p-values for each subdivision
	blocks = len(list(subdivisions.keys()))
	xticks = list()
	for i in range(blocks):
		xticks.append('subdivision-'+str(i))
	for ch in eeg_channels:
		plt.figure()
		plt.bar(xticks,Channel_P_values[ch])
		plt.title(subject+' subdivision hypothesis testing')
		plt.ylabel('P_values')
		plt.xlabel('Channel')
		plt.xticks(rotation='vertical')
		plt.savefig(save_path+'/Hypothesis_Testing/data_count_factor_'+str(data_count_factor)+'/P_values_'+ch+'.png')
		plt.close()

def random_null_hypothesis_test_subdivision(target_epochs,non_target_epochs,average_epoch_data,eeg_channels,subject,epoch_length,fs,save_path,subdivision_block,data_count_factor=None,pool_size=100):
	for data_count_factor in subdivision_block:
		subdivisions = {}
		for i in range(data_count_factor):
			subdivisions[str(i)] = random_null_hypothesis_test(target_epochs,non_target_epochs,average_epoch_data,eeg_channels,subject,epoch_length,fs,save_path,data_count_factor=data_count_factor,offset=i-1,pool_size=pool_size)
		plot_subdivision_p_values(subdivisions,eeg_channels,data_count_factor,save_path,subject)

def Main():
	parser = argparse.ArgumentParser(description='P300 Data Training')
	parser.add_argument('-s',"--subject",metavar='', help="subject number [1,2,3,etc.]")
	parser.add_argument('-pa',"--part_a",metavar='', help="Whether to run part a or not [1 or 0]",default='0')
	parser.add_argument('-pb',"--part_b",metavar='', help="Whether to run part b or not [1 or 0]",default='0')
	parser.add_argument('-pc',"--part_c",metavar='', help="Whether to run part c or not [1 or 0]",default='0')

	args = parser.parse_args()
	subject='subject'+args.subject
	part_A=int(args.part_a)
	part_B=int(args.part_b)
	part_C=int(args.part_c)

	data_directory = '../Data'
	eeg_channels = ['Fp1', 'AF3', 'F7', 'F3', 'FC1','FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'MA1', 'MA2']
	fs = 2048#in Hz
	sessions = os.listdir(data_directory+'/'+subject)
	Events = DR.data_read(data_directory,subject,sessions,fs)#data read for a specific subject and session
	[h,H] = filter_design(1,12,0.5,fs)#design filter
	epoch_length = 1000#in ms
	baseline_DC_range = 100#in ms
	movement_noise_threshold = 40#in microvolts
	hypothesis_test_pool_size = 1000
	subdivision_block = [4,24]

	pca_channels = ['Fz', 'Cz', 'Pz', 'Oz']
	pca_eigs = 2
	windowing_before_pca = True
	time_window_before_pca = [0.2,0.5]
	downsampling_before_pca = True
	downsampling_rate_before_pca = 32
	data_clustering_before_pca = [1,10,25,50]

	data_count_factor = 1#A decreasing factor for tracking how many data are to be counted for hypothesis testing. 1 means all data. 4 means 1/4 of the data. 24 means 1/24 of the data, etc.
	epoch_length = math.floor(fs*epoch_length/1000)
	baseline_DC_range = math.floor(fs*baseline_DC_range/1000)
	Epoch_data = {}
	per_session_data = {}
	target_epochs = {}
	non_target_epochs = {}

	if part_A:
		for sess in Events.keys():
			print(subject,'--',sess)
			for key in Events[sess].keys():
				event_data = Events[sess][key]['data']
				reference_signal = (event_data[-2,:]+event_data[-1,:])/2#construct the reference signal
				re_reference_signals = event_data[0:-2,:] - reference_signal#re-reference signal
				save_path = './Results/Filtered_Events/'+subject+'/'+sess+'/'+key
				Events[sess][key]['data'] = filtered_events(re_reference_signals,fs,h,save_path,eeg_channels,verbose=1)#Filter te re-referenced signal

				#Extract Epochs data
				Epoch_data[key] = extract_epochs(Events[sess][key]['events'],Events[sess][key]['data'],epoch_length,eeg_channels,fs)

			target_epochs[sess],non_target_epochs[sess] = extract_target_non_target(Epoch_data,Events[sess],eeg_channels[:-2],epoch_length,baseline_DC_range,movement_noise_threshold)
			save_path = './Results/Epoch_Data/'+subject+'/'+sess+'/'
			per_session_data[sess] = plot_session_target_non_target(target_epochs[sess],non_target_epochs[sess],eeg_channels[:-2],save_path,subject,sess,epoch_length,fs,verbose=1)

		save_path = './Results/Epoch_Data/'+subject+'/'
		average_epoch_data = plot_subject_target_non_target(per_session_data,eeg_channels[:-2],save_path,subject,epoch_length,fs)
		np.save(save_path+'average_epoch_data.npy', average_epoch_data)
		np.save(save_path+'per_session_data.npy', per_session_data)
		np.save(save_path+'target_epochs.npy', target_epochs)
		np.save(save_path+'non_target_epochs.npy', non_target_epochs)
	else:
		save_path = './Results/Epoch_Data/'+subject+'/'
		average_epoch_data = np.load(save_path+'average_epoch_data.npy',allow_pickle=True)
		per_session_data = np.load(save_path+'per_session_data.npy',allow_pickle=True)
		target_epochs = np.load(save_path+'target_epochs.npy',allow_pickle=True)
		non_target_epochs = np.load(save_path+'non_target_epochs.npy',allow_pickle=True)

	#Null hypothesis testing. returns p values for each channel
	if part_B:
		save_path = './Results/Epoch_Data/'+subject+'/'
		all_targets,all_non_targets = find_all_targets_non_targets(target_epochs,non_target_epochs,eeg_channels[:-2])
		P_values = random_null_hypothesis_test(all_targets,all_non_targets,average_epoch_data,eeg_channels[:-2],subject,epoch_length,fs,save_path,data_count_factor=1,pool_size=hypothesis_test_pool_size)
		random_null_hypothesis_test_subdivision(all_targets,all_non_targets,average_epoch_data,eeg_channels[:-2],subject,epoch_length,fs,save_path,subdivision_block,pool_size=hypothesis_test_pool_size)
		np.save(save_path+'all_targets.npy', all_targets)
		np.save(save_path+'all_non_targets.npy', all_non_targets)
		np.save(save_path+'P_values.npy', P_values)
	else:
		save_path = './Results/Epoch_Data/'+subject+'/'
		all_targets = np.load(save_path+'all_targets.npy',allow_pickle=True)
		all_non_targets = np.load(save_path+'all_non_targets.npy',allow_pickle=True)
		P_values = np.load(save_path+'P_values.npy',allow_pickle=True)

	if part_C:
		#PCA on the data_channels
		save_path = './Results/Epoch_Data/'+subject+'/PCA'
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		for K in data_clustering_before_pca:
			[pca_data,ground_truth,num_trg,num_ntrg] = PCA.data_prepare(pca_channels,all_targets,all_non_targets,eeg_channels,fs,K,windowing=windowing_before_pca,time_window=time_window_before_pca,downsampling=downsampling_before_pca,downsampling_rate=downsampling_rate_before_pca)
			[Q,v,w] = PCA.calculate_eigen_vectors(pca_data,pca_eigs,K,save_path)
			[ak,bk] = PCA.get_low_dimensional_projections(v,Q,pca_data,K,save_path)
			#cl1_indices,cl2_indices = PCA.clustering_data(ak,bk,save_path)
			PCA.ROC_plot(pca_data,ak,bk,Q,num_trg,num_ntrg,ground_truth,K,save_path)

	else:
		print('\n\nDone\n\n')
Main()
