import os,glob
import scipy.io
import numpy as np
import datetime
from collections import Counter

data_directory = '../Data'
eeg_channels = ['Fp1', 'AF3', 'F7', 'F3', 'FC1','FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'MA1', 'MA2']
fs = 2048#in Hz


def unpackStamp(x):
	y = np.int32(x[0])
	mo = np.int32(x[1])
	d = np.int32(x[2])
	h = np.int32(x[3])
	mi = np.int32(x[4])
	s = x[5]
	s_new = np.int32(np.floor(s))
	micros = np.int32((s - s_new) * 1e6)
	unpacked = datetime.datetime(y, mo, d, h, mi, s_new, micros)
	return unpacked
def events2samps(events, fs):
	firsteve_time = 0.4
	Nevents = events.shape[0]
	evesamps = np.zeros(Nevents)
	for k in range(Nevents):
		td = unpackStamp(events[k, :]) - unpackStamp(events[0, :])
		evesamps[k] = np.int32(np.round(td.total_seconds()*fs + firsteve_time*fs + 1))
	return evesamps


subjects = os.listdir(data_directory)
for S in subjects:
	sessions = os.listdir(data_directory+'/'+S)
	for ses in sessions:
		try:
			EEG_data_files = os.listdir(data_directory+'/'+S+'/'+ses)
			#print(S,'-',ses,'-',EEG_data_files)
		except:
			continue

		for eeg in EEG_data_files:
			print(S,'-',ses,'-',eeg)
			EEG_data_path = data_directory+'/'+S+'/'+ses+'/'+eeg
			EEG_data = scipy.io.loadmat(EEG_data_path)
			#print(EEG_data.keys())
			target = int(EEG_data['target'].squeeze())
			stimuli = EEG_data['stimuli'].squeeze()
			targets_counted = int(EEG_data['targets_counted'].squeeze())
			data = EEG_data['data'].squeeze()
			events = EEG_data['events'].squeeze()

			#print(data)
			#print(events)
			events2samples = events2samps(events,fs)
			#print(events.shape)
			print(events2samples.shape)
			stimu_count = Counter(stimuli.tolist())
			print(target)
			print(stimu_count[target])
			print(targets_counted)
