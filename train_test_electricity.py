# command to launch this code ijn the background
# nohup python3 train_test_electricity.py > train_test_electricity.log 2>&1 &

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import csv
import matplotlib.pyplot as plt
import time

import torch
import torch.optim as optim
import numpy as np
import os
from lib.utils import fix_seed, instantiate_model, read_table, get_emb
from lib.train import loop
from data.datasets import timeseries_dataset
import pandas as pd
import warnings
import time



torch.backends.cudnn.benchmark = False
num_cores = 4
torch.set_num_threads(2)
#%% Initialize parameters for datasets
datasets = ['uci_electricity','uci_traffic','kaggle_favorita', 'kaggle_webtraffic', 'kaggle_m5']
dim_inputseqlens = [168, 168, 90, 90, 90]
dim_outputseqlens = [24, 24, 28, 30, 28]
dim_maxseqlens = [500, 500, 150, 150, 119]
#%% Initiate experiment
dataset_id = 0
cuda = 0
seed = 0

num_samples_train = 1500000 if datasets[dataset_id] == 'kaggle_m5' else 500000
num_samples_validate = 30000 if datasets[dataset_id] == 'kaggle_m5' else 10000


num_samples_test = 10000

fix_seed(seed)
early_stopping_patience = 5
scaling = True
epochs = 100
#%% Load data
dataset_name = datasets[dataset_id]
experiment_dir = 'experiments/'+dataset_name
dim_inputseqlen = dim_inputseqlens[dataset_id] # Input sequence length
dim_outputseqlen = dim_outputseqlens[dataset_id]  # Output prediction length
dim_maxseqlen = dim_maxseqlens[dataset_id]
# Import data
dset = timeseries_dataset(dataset_name, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen)
training_set = dset.load('train')
validation_set = dset.load('validate')
test_set = dset.load('test')

# Initialize sample sets
id_samples_train = torch.randperm(len(training_set))[:num_samples_train]
id_samples_validate = torch.randperm(len(validation_set))[:num_samples_validate]
id_samples_test = torch.randperm(len(test_set))[:num_samples_test]

#%% Algorithm parameters


device = torch.device(cuda)
warnings.simplefilter(action='ignore', category=FutureWarning)

file_experiments = experiment_dir + f'/experiments_{dataset_name}.csv'
hyperparams_filename = f"{experiment_dir}/"
d_emb = get_emb(dataset_name)


algorithm = 'bitcn_att_skip'

# main loop to test different version of this architecture
time_per_cong=[]
for learning_rate in [0.001]: # you can fine tune it using different values (usually 0.0001, 0.0005, 0.001 )
    for batch_size in [64]: #you can fine tune it using other values (64,128,256,512)
         for d_hidden in [25]: #you cna fine tune it usqing other values like 5,10;15;20;25;30,....
            start_time = time.time()    
             
            for seed in  [0,1,2,3,4]:
                N = 6
                NATT = 4
                fix_seed(seed)
                dropout = 0.1
                kernel_size = 9
                heads = 5
                
                params= [training_set.d_lag, training_set.d_cov, d_emb,training_set.dim_output,d_hidden, dropout, N,kernel_size,NATT,heads]
            
                ## initi the model
                filename = f"{experiment_dir}/{algorithm}/{algorithm}_seed={seed}_lr={learning_rate}_bs={batch_size}_N={N}_NATT={NATT}_d_hidden={d_hidden}_heads={heads}"
                print(filename)
                if not os.path.isdir(f"{experiment_dir}/{algorithm}"): os.makedirs(f"{experiment_dir}/{algorithm}")
                fix_seed(seed)
                n_batch_train = (len(id_samples_train) + batch_size - 1) // batch_size 
                n_batch_validate = (len(id_samples_validate) + batch_size - 1) // batch_size
                if 'model' in locals(): del model
            
                model = instantiate_model(algorithm)(*params).to(device)   
            
                ############## Train #################
                optimizer = optim.Adam(model.parameters(), lr = learning_rate)
                loss_train = np.zeros((epochs))
                loss_validate = np.zeros((epochs))
                loss_validate_best = 1e6
                early_stopping_counter = 0
                best_epoch = 0
            
                ## model train / valid code 
                ## Traibn valid 
                for epoch in range(epochs):
                    print(f'Epoch {epoch + 1}/{epochs}')
                    model, loss_train[epoch], _, _, _, _ = loop(model, training_set, optimizer, batch_size, id_samples_train, train=True, metrics=True, scaling=scaling)    
                    _, loss_validate[epoch], yhat_tot, y_tot, x_tot, df_validate = loop(model, validation_set, optimizer, batch_size, id_samples_validate, train=False, metrics=True, scaling=scaling)    
                    if loss_validate[epoch] < loss_validate_best:
                        torch.save({'epoch':epoch, 
                                   'model_state_dict':model.state_dict(),
                                   'optimizer_state_dict':optimizer.state_dict()}, filename)
                        df_validate.to_csv(filename + '_validate.csv')
                        loss_validate_best = loss_validate[epoch]
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    if (early_stopping_counter == early_stopping_patience) | (epoch == epochs - 1):
                        loss_train = loss_train / n_batch_train
                        loss_validate = loss_validate / n_batch_validate
                        df_loss = pd.DataFrame({'Validation_loss':loss_validate,'Training_loss':loss_train})
                        df_loss.to_csv(filename + '_loss.csv')
                        break
             
                params= [test_set.d_lag, test_set.d_cov, d_emb,test_set.dim_output,d_hidden, dropout, N,kernel_size,NATT,heads]
                filename = f"{experiment_dir}/{algorithm}/{algorithm}_seed={seed}_lr={learning_rate}_bs={batch_size}_N={N}_NATT={NATT}_d_hidden={d_hidden}_heads={heads}"
            
                fix_seed(seed)
                n_batch_test = (len(id_samples_test) + batch_size - 1) // batch_size
                if 'model' in locals(): del model
                model = instantiate_model(algorithm)(*params) 
            
                #print(filename)
                checkpoint = torch.load(filename)
            
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                optimizer=None
                _, loss_test, yhat_tot, y_tot, x_tot, df_test = loop(model, test_set, optimizer, batch_size, id_samples_test, train=False, metrics=True, scaling=scaling)    
                df_test.to_csv(filename + '_test.csv')
                        

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Convert to hours, minutes, and seconds
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            time_per_cong.append(elapsed_time)
            print(f"Training completed in {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds.")                                            
                                    

with open('electricity_time_per_conf.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(time_per_cong)