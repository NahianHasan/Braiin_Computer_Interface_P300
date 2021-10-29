import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.io
import heapq
import math

def calculate_covariance_matrix(X):
	C = np.zeros((X.shape[1],X.shape[1]))
	for i in range(0,X.shape[1]):
		for j in range(0,X.shape[1]):
			C[i,j] = np.matmul(np.transpose(X[:,i]),X[:,j])

	C = C/X.shape[1]
	return C

def calculate_eigen_vectors(waveforms,num_eigs,K,save_path):
	waveforms = waveforms.astype('float')
	C = calculate_covariance_matrix(waveforms)
	#C = np.cov(np.transpose(waveforms))
	plt.figure()
	plt.matshow(C)
	plt.colorbar()
	#plt.show()
	plt.savefig(save_path+'/Covariance_Matrix_K_'+str(K)+'.png')
	[w,v] = np.linalg.eig(C)#w=eigenvalues, v=eigenvectors; the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
	ws = heapq.nlargest(num_eigs, w)
	ind = [list(w).index(i) for i in ws]
	Q = np.zeros((waveforms.shape[1],num_eigs))
	Q[:,:] = v[:,ind]
	plt.figure()
	plt.plot(Q)
	plt.legend(['Q'+str(i) for i in range(1,Q.shape[1]+1)])
	#plt.show()
	plt.savefig(save_path+'/Eigen_vectors_K_'+str(K)+'.png')
	return Q,v,w

def get_low_dimensional_projections(v,Q,waveforms,K,save_path):
	ak = list()
	bk = list()
	for i in range(0,waveforms.shape[0]):
		ak.append(np.dot(waveforms[i,:],Q[:,0]))
		bk.append(np.dot(waveforms[i,:],Q[:,1]))

	ak,bk = np.array(ak),np.array(bk)
	plt.figure()
	plt.scatter(ak,bk)
	plt.xlabel('ak')
	plt.ylabel('bk')
	plt.savefig(save_path+'/Projections_K_'+str(K)+'.png')
	return ak,bk

def clustering_data(ak,bk,num_trg,num_ntrg,save_path,K,boundary):
	cl1_indices = np.where(bk<=boundary)
	cl1_a = ak[cl1_indices]
	cl1_b = bk[cl1_indices]
	plt.figure()
	plt.scatter(cl1_a,cl1_b)
	cl2_indices = np.where(bk>boundary)
	cl2_a = ak[cl2_indices]
	cl2_b = bk[cl2_indices]
	plt.scatter(cl2_a,cl2_b)
	plt.xlabel('ak')
	plt.ylabel('bk')
	#plt.show()
	plt.savefig(save_path+'/Cluster_K_'+str(K)+'_b_'+str(boundary)+'.png')


	cl1_a = ak[1:num_trg]
	cl1_b = bk[1:num_trg]
	plt.figure()
	plt.scatter(cl1_a,cl1_b)
	cl2_a = ak[num_trg:num_trg+num_ntrg]
	cl2_b = bk[num_trg:num_trg+num_ntrg]
	plt.scatter(cl2_a,cl2_b)
	plt.xlabel('ak')
	plt.ylabel('bk')
	#plt.show()
	plt.savefig(save_path+'/Cluster_K_'+str(K)+'real_projections.png')


	return cl1_indices[0],cl2_indices[0]
def perf_measure(y_actual, y_pred):
	TP = 0
	FP = 0
	TN = 0
	FN = 0

	for i in range(len(y_pred)):
		if y_actual[i]==y_pred[i]==1:
			TP += 1
		if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
			FP += 1
		if y_actual[i]==y_pred[i]==0:
			TN += 1
		if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
			FN += 1

	return TP, FP, TN, FN
def ROC_plot(pca_data,ak,bk,Q,num_trg,num_ntrg,ground_truth,K,save_path):
	boundaries = np.arange(np.min(bk),np.max(bk),(np.max(bk)-np.min(bk))/20)
	hit_rate = list()
	false_alarm = list()
	for i in range(len(boundaries)):

		predictions = np.zeros((ground_truth.shape[0],1))
		cl1_indices,cl2_indices = clustering_data(ak,bk,num_trg,num_ntrg,save_path,K,boundaries[i])
		predictions[cl1_indices] = 1
		predictions[cl2_indices] = 0
		[TP,FP,TN,FN] = perf_measure(ground_truth, predictions)
		hit_rate.append(TP/(TP+FN))#Sensitivity
		false_alarm.append(FP/(FP+TN))
		#plot_any_cluster_waveform(pca_data,cl1_indices,ak,bk,Q,K,'b_'+str(i)+'_cl1',save_path)
		#plot_any_cluster_waveform(pca_data,cl2_indices,ak,bk,Q,K,'b_'+str(i)+'_cl2',save_path)

	#Plot the ROC curve
	plt.figure()
	plt.plot(false_alarm,hit_rate)
	plt.title('ROC Curve (K = '+str(K)+')')
	plt.xlabel('False Alarm Rate (1-specificity)')
	plt.ylabel('Sensitivity (Hit Rate)')
	plt.savefig(save_path+'/ROC_K_'+str(K)+'.png')
	plt.close()


def plot_any_cluster_waveform(waveforms,ind,ak,bk,Q,K,name,save_path):
	plt.figure()
	for i in ind:
		plt.plot(waveforms[i,:])
	plt.xlabel('sample number')
	plt.ylabel('Signal amplitude (uv)')
	plt.savefig(save_path+'/All_Epochs_K_'+str(K)+'.png')

	cl_a = ak[ind]
	cl_b = bk[ind]
	mean_a, mean_b = np.mean(cl_a),np.mean(cl_b)
	print('mean a = ',mean_a)
	print('mean b = ',mean_b)
	plt.plot(mean_a*Q[:,0]+mean_b*Q[:,1],'k', linewidth=3)
	plt.savefig(save_path+'/Cluster_'+name+'_K_'+str(K)+'.png')

def data_prepare(pca_channels,all_targets,all_non_targets,eeg_channels,fs,K,windowing=False,time_window=[0.2,0.5],downsampling=False,downsampling_rate=32):

	all_targets = {k: all_targets.item().get(k) for k in pca_channels}#keep the pca channels only
	all_non_targets = {k: all_non_targets.item().get(k) for k in pca_channels}

	pca_trg_data = all_targets[pca_channels[0]]
	for i in range(1,len(pca_channels)):
		pca_trg_data = np.concatenate((pca_trg_data,all_targets[pca_channels[i]]),axis=-1)

	pca_ntrg_data = all_non_targets[pca_channels[0]]
	for i in range(1,len(pca_channels)):
		pca_ntrg_data = np.concatenate((pca_ntrg_data,all_non_targets[pca_channels[i]]),axis=-1)

	ground_truth = list()#1=Target, 0=non-target
	if K==1:
		pca_data = (np.concatenate((pca_trg_data,pca_ntrg_data),axis=0)).squeeze()
		ground_truth = (np.concatenate((np.ones((pca_trg_data.shape[0],1)),np.zeros((pca_ntrg_data.shape[0],1))),axis=0)).squeeze()
	else:
		pca_data = list()
		for row in range(0,pca_trg_data.shape[0]-K,K):
			pca_data.append(np.average(pca_trg_data[row:row+K,:],axis=0))
			ground_truth.append(1)
		for row in range(0,pca_ntrg_data.shape[0]-K,K):
			pca_data.append(np.average(pca_ntrg_data[row:row+K,:],axis=0))
			ground_truth.append(0)
		pca_data = (np.array(pca_data)).squeeze()
		ground_truth = np.array(ground_truth)

	num_trg = math.floor(pca_trg_data.shape[0]/K)
	num_ntrg = math.floor(pca_ntrg_data.shape[0]/K)
	del pca_trg_data
	del pca_ntrg_data

	print('PCA Data Shape (K = {}) = {}'.format(K,pca_data.shape))
	if downsampling:
		pca_data = pca_data[:,0:-1:math.floor(fs/downsampling_rate)]
		fs = downsampling_rate
	if windowing:
		pca_data = pca_data[:,math.floor(time_window[0]*fs):math.floor(time_window[1]*fs)]

	print('PCA Data Shape after post processing (K = {}) = {}'.format(K,pca_data.shape))
	return pca_data,ground_truth,num_trg,num_ntrg
