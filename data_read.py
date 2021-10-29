import os,glob
import scipy.io
import numpy as np
import datetime


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

def data_read(data_directory,subject,sessions,fs):
	Events = {}
	for sess in sessions:
		EEG_data_files = os.listdir(data_directory+'/'+subject+'/'+sess)
		Events[sess] = {}
		for eeg in EEG_data_files:
			EEG_data_path = data_directory+'/'+subject+'/'+sess+'/'+eeg
			EEG_data = scipy.io.loadmat(EEG_data_path)
			#print(EEG_data.keys())
			target = int(EEG_data['target'].squeeze())
			stimuli = EEG_data['stimuli'].squeeze()
			targets_counted = int(EEG_data['targets_counted'].squeeze())
			data = EEG_data['data'].squeeze()
			events = EEG_data['events'].squeeze()
			events2samples = events2samps(events,fs)
			sub_name = eeg.split('.')[0]
			Events[sess][sub_name] = {'target':target,'targets_counted':targets_counted,'stimuli':stimuli,'data':data,'events':events2samples}

	return Events
