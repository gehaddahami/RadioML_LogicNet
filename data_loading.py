# Importing necessary libraries and packages
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
from tqdm.notebook import tqdm 
import os.path  
import h5py 

# Data loading (The path will included in the main.py file)
# dataset_path = "/home/student/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
# os.path.isfile(dataset_path) 

#Load data from HDF5 file into a PyTorch tensor 
class Radioml_18(Dataset):
    def __init__(self, dataset_path): 
        super(Radioml_18, self).__init__()
        h5py_file = h5py.File(dataset_path, 'r')
        self.data = h5py_file['X']
        self.modulations  = np.argmax(h5py_file['Y'], axis=1) 
        self.snr = h5py_file['Z'][:, 0]
        self.len = self.data.shape[0] # Calculate the total number of samples

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']

        self.snr_classes = np.arange(-20.,31.,2) 
        
        np.random.seed(2018) # Set the random seed for reproducibility
        train_indices = []
        validation_indices = []
        test_indices = []
        for mod in range(0, 24): # Iterate over all 24 modulation classes
            for snr_idx in range(0, 26): # Iterate over all signal-to-noise ratios from (-20, 30) dB
                # Calculate the starting index for each modulation and SNR combination
                start_index = 26*4096*mod + 4069*snr_idx 
                  # Generate the indices for each subclass (modulation and SNR combination)
                indices_subclass = list(range(start_index, start_index+4096))

                # Splitting the data into 80% training and 20% testing 
                split = int(np.ceil(0.1*4096))
                np.random.shuffle(indices_subclass) 
                train_indicies_sublcass = indices_subclass[:int(0.7*len(indices_subclass))]
                validation_indices_subclass = indices_subclass[int(0.7*len(indices_subclass)):int(0.8*len(indices_subclass))]
                test_indicies_subclass = indices_subclass[int(0.8*len(indices_subclass)):] 
                
                # to choose a specific SNR valaue or range is here 
                if snr_idx >= 25: 
                    train_indices.extend(train_indicies_sublcass)
                    validation_indices.extend(validation_indices_subclass)
                    test_indices.extend(test_indicies_subclass)

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.validation_sampler = SubsetRandomSampler(validation_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

        

        print('Dataset shape:', self.data.shape)
        print("Train indices for SNRs 28, 30 dB:", len(train_indices))
        print("Validation indices for SNRs 28, 30 dB:", len(validation_indices))
        print("Test indices for SNRs 28, 30 dB:", len(test_indices)) 

        # Print input length
        input_length = self.data.shape[1]
        print("Input length:", input_length)

    def __getitem__(self, index):
        # Transform frame into pytorch channels-first format 
        return self.data[index].transpose(), self.modulations[index], self.snr[index]
    
    def __len__(self): 
        return self.len 
    