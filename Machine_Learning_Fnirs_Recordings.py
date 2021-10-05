import pandas as pd
import numpy as np 
import mne
from sklearn import svm
from sklearn.metrics import accuracy_score

#--------------------------preparing training data--------------------------
#taking data from file, 16 sets of 250 coulmns for 19 rows
matrix_join = np.zeros((19, 96))
matrix_solo = np.zeros((19, 96))
array_data = []

k = coulmn = 1
for file_num in range (1, 7):# Import 6 of the joint filles
        in_csv = 'SP0220' + str(file_num) + '_Artifact Rejection_Joint.csv'
        data_block = pd.read_csv(in_csv)
        k = 1
        while k < 32: 
            for line in range(1, 20):
               array_data.clear()
               for i in range(250):
                   array_data.append(data_block.iloc[line - 1][10 * k + i])  
               psds_welch, freqs_welch = mne.time_frequency.psd_array_welch(np.asarray(array_data), 250, 13, 30, n_fft=10, n_overlap=0, n_per_seg=None, n_jobs=1, verbose=None)
               matrix_join[line - 1][coulmn - 1] = np.average(psds_welch)
            k = k + 2
            coulmn = coulmn + 1

k = coulmn = 1
for file_num in range (1, 7):# Import 6 of the solo filles
        in_csv = 'SP0220' + str(file_num) + '_Artifact Rejection_Solo.csv'
        data_block = pd.read_csv(in_csv)
        k = 1
        while k < 32: 
            for line in range(1, 20):
               array_data.clear()
               for i in range(250):
                   array_data.append(data_block.iloc[line - 1][10 * k + i])  
               psds_welch, freqs_welch = mne.time_frequency.psd_array_welch(np.asarray(array_data), 250, 13, 30, n_fft=10, n_overlap=0, n_per_seg=None, n_jobs=1, verbose=None)
               matrix_solo[line - 1][coulmn - 1] = np.average(psds_welch)
            k = k + 2
            coulmn = coulmn + 1

y_train = []
for i in range(192):
    if i < 96 :
        y_train.append(1)
    else:
        y_train.append(0)

matrix_train = np.hstack((matrix_join, matrix_solo))

#--------------------------svm trianing--------------------------

clf = svm.SVC(kernel = 'linear') # Linear Kernel

clf.fit(matrix_train.T, y_train)

#--------------------------preparing test data--------------------------

k = coulmn = 1
matrix_join_test = np.zeros((19, 32))
for file_num in range (10, 12):# Import the 2 last joint filles
        in_csv = 'SP022' + str(file_num) + '_Artifact Rejection_Joint.csv'
        data_block = pd.read_csv(in_csv)
        k = 1
        while k < 32: 
            for line in range(1, 20):
               array_data.clear()
               for i in range(250):
                   array_data.append(data_block.iloc[line - 1][10 * k + i])  
               psds_welch, freqs_welch = mne.time_frequency.psd_array_welch(np.asarray(array_data), 250, 13, 30, n_fft=10, n_overlap=0, n_per_seg=None, n_jobs=1, verbose=None)
               matrix_join_test[line - 1][coulmn - 1] = np.average(psds_welch)
            k = k + 2
            coulmn = coulmn + 1

print("start predict")

y_pred_join = clf.predict(matrix_join_test.T)

#--------------------------predict--------------------------

y_test = []
for i in range(32):
    y_test.append(1) 

#--------------------------results--------------------------

print(accuracy_score(y_test, y_pred_join))
