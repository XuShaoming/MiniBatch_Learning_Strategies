import numpy as np
import sys
import os
import subprocess as sp
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from models import EarlyStopping, GRU, MSE
from preprocess import load_data, SampleData
np.random.seed(1)

# check if cuda is available
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

if torch.cuda.is_available():
    device = torch.device("cuda")
if get_gpu_memory()[0] < 1500 :
    device = torch.device("cpu")
print(device)

notes = [
    'Goal: Simulate the target variable using 7 weather drivers including Date, PRECIPmm, TMP_MXdgC, TMP_MNdgC, SOLARMJ/m2, WINDM/s, RHmd.'
    'Target variables: SW_ENDmm, SNOmm, Q_pred_mm.'
    'Data: 1000 years simulation data.'
    'Split data: First 50% of data as the training set, middle 10% of data as the validation set,  last 40% of data as the testing set.'
    'Settings:.'
    '- training: Sequential Stateful mini-batch (SSMB) algorithm.'
    '- Model: one layer GRU.'
    ]

#--------------------------------------------- Hyperparameters--------------------------
print(sys.argv)
run_iter = sys.argv[1]
run_task = sys.argv[2]
output_vars = [sys.argv[3]]
rversion = sys.argv[4]
n_steps = int(sys.argv[5])
hidden_size = int(sys.argv[6])
epochs = int(sys.argv[7])
patience = int(sys.argv[8])
learning_rate = float(sys.argv[9])
sim_type = sys.argv[10]
batch_size = int(sys.argv[11])
input_size = int(sys.argv[12])
dropout = float(sys.argv[13])
# number of RNN layers 
nlayers=int(sys.argv[14])

assert(sim_type == 'full_wsl')

# '''
# Hard-coded hyper-parameters.
# '''
# run_iter = 0
# run_task = '1_GRU_plain_hs'
# output_vars = ['SW_ENDmm']
# rversion = 'hs32'
# n_steps = 366
# hidden_size = 32
# epochs = 500
# patience = 50
# learning_rate = 0.01
# sim_type = 'full_wsl'
# batch_size = 64
# input_size = 7
# dropout = 0
# # number of RNN layers 
# nlayers=1

#--------------------------------------------- Input variables--------------------------
n_classes = len(output_vars)
input_vars = ['Date', 'PRECIPmm', 'TMP_MXdgC', 'TMP_MNdgC', 'SOLARMJ/m2', 'WINDM/s', 'RHmd']
assert(len(input_vars) == input_size)

#----------------paths-------------------
res_dir = '../results/head_water_SWAT_1000_years/'
exp_dir = res_dir + '{}/rversion_{}/'.format(run_task, rversion)
if os.path.isdir(exp_dir)==False:
    os.makedirs(exp_dir)

#--------------------------------------------- load input data -----------------------------
new_data_dir = '../data/1000_year_simulation/'

if sim_type =='full_wsl':
    path = new_data_dir + 'head_water_SWAT_1000_years.csv'
elif sim_type =='nosnow_nofrozen_wsl':
    path = new_data_dir + 'head_water_SWAT_1000_years_no_snow_no_frozen.csv'
elif sim_type =='nosnow_wsl':
    path = new_data_dir + 'head_water_SWAT_1000_years_no_snow.csv'
else:
    raise FileNotFoundError

df = pd.read_csv(path)
feat, label = load_data(df, output_vars, input_vars, input_size)

# First 50% of data as the training set, middle 10% of data as the validation set, last 40% of data as the testing set.
train_percentage = 0.5
valid_percentage = 0.1
test_percentage = 1 - (train_percentage + valid_percentage)

# Split data
T = feat.shape[0]
train_len = int(T*train_percentage)
valid_len = int(T*valid_percentage)
test_len = T - train_len - valid_len
print(train_len,valid_len,test_len)
train_x = feat[:train_len].copy()
train_y = label[:train_len].copy()
valid_x = feat[train_len:train_len+valid_len].copy()
valid_y = label[train_len:train_len+valid_len].copy()
test_x = feat[train_len+valid_len:].copy()
test_y = label[train_len+valid_len:].copy()

# Normalize data
scaler_x = StandardScaler()
scaler_x.fit(train_x)
x_train = scaler_x.transform(train_x)
x_valid = scaler_x.transform(valid_x)
x_test = scaler_x.transform(test_x)
scaler_y = StandardScaler()
scaler_y.fit(train_y)
y_train = scaler_y.transform(train_y)
y_valid = scaler_y.transform(valid_y)
y_test = scaler_y.transform(test_y)
# Masks
m_train = y_train.copy()
m_train[:,:] = 1 # this means no masking
m_valid = y_valid.copy()
m_valid[:,:] = 1 # this means no masking
m_test = y_test.copy()
m_test[:,:] = 1 # this means no masking

# Sample data
## Get indexes
train_idx = np.arange(len(y_train))
valid_idx = np.arange(len(y_valid))
test_idx = np.arange(len(y_test))
## Set stride
num_samples_train = 0
shift_train = int(n_steps)
num_samples_valid = 0
shift_valid = int(n_steps)
num_samples_test = 0
shift_test = int(n_steps)
## Get lists of indexes to sample data. 
train_idx_arr = SampleData(train_idx[len(y_train)%n_steps:],n_steps,shift_train,num_samples_train)
train_idx_arr = np.vstack((np.arange(n_steps),train_idx_arr))
num_train_samples = train_idx_arr.shape[0]
valid_idx_arr = SampleData(valid_idx,n_steps,shift_valid,num_samples_valid)
num_valid_samples = valid_idx_arr.shape[0]
test_idx_arr = SampleData(test_idx,n_steps,shift_test,num_samples_test)
num_test_samples = test_idx_arr.shape[0]
## Sample data
x_train_sp_ = x_train[train_idx_arr,:]
y_train_sp_ = y_train[train_idx_arr,:]
m_train_sp_ = y_train_sp_.copy()
m_train_sp_[:,:,:] = 1 # this means no masking
x_valid_sp_ = x_valid[valid_idx_arr,:]
y_valid_sp_ = y_valid[valid_idx_arr,:]
m_valid_sp_ = y_valid_sp_.copy()
m_valid_sp_[:,:,:] = 1 # this means no masking
x_test_sp_ = x_test[test_idx_arr,:]
y_test_sp_ = y_test[test_idx_arr,:]
m_test_sp_ = y_test_sp_.copy()
m_test_sp_[:,:,:] = 1 # this means no masking
'''
Save the hyperameters
'''
#-------------------saving all pararms------------------------
params={'learning_rate':learning_rate,'epochs':epochs,'batch_size':batch_size,'hidden_size':hidden_size,'input_size':input_size,
        'n_steps':n_steps,'dropout':dropout,'n_classes':n_classes,'num_samples_train':num_samples_train,'shift_train':shift_train,
        'num_samples_valid':num_samples_valid,'shift_valid':shift_valid,'num_samples_test':num_samples_test,'shift_test':shift_test,
        'train_percentage':train_percentage,'valid_percentage':valid_percentage,'notes':notes}

np.save(exp_dir + 'params_ss_{}'.format(n_steps),params)
np.save(exp_dir + 'train_info_ss_{}'.format(n_steps),{'train_idx_arr':train_idx_arr,'train_idx':train_idx})
np.save(exp_dir + 'test_info_ss_{}'.format(n_steps),{'test_idx_arr':test_idx_arr,'test_idx':test_idx})
print("The network hypermeters : ")
for k,v in params.items():
    if k != "notes":
        print(k,v)

'''
Create model
'''
model1=GRU(input_size, hidden_size, nlayers, n_classes, dropout)
print(model1)
params = list(model1.parameters())
# Print out the model strcuture.
print(len(params))
print("Model's state_dict:")
for param_tensor in model1.state_dict():
    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
    
# Send data to the device
x_train_ = torch.from_numpy(np.expand_dims(x_train,0)).type(torch.float32).to(device)
y_train_ = torch.from_numpy(np.expand_dims(y_train,0)).type(torch.float32).to(device)
m_train_ = torch.from_numpy(np.expand_dims(m_train,0)).type(torch.float32).to(device)
x_valid_ = torch.from_numpy(np.expand_dims(x_valid,0)).type(torch.float32).to(device)
y_valid_ = torch.from_numpy(np.expand_dims(y_valid,0)).type(torch.float32).to(device)
m_valid_ = torch.from_numpy(np.expand_dims(m_valid,0)).type(torch.float32).to(device)
x_test_ = torch.from_numpy(np.expand_dims(x_test,0)).type(torch.float32).to(device)
y_test_ = torch.from_numpy(np.expand_dims(y_test,0)).type(torch.float32).to(device)
m_test_ = torch.from_numpy(np.expand_dims(m_test,0)).type(torch.float32).to(device)
# Send data to the device
x_train_sp = torch.from_numpy(x_train_sp_).type(torch.float32).to(device)
y_train_sp = torch.from_numpy(y_train_sp_).type(torch.float32).to(device)
m_train_sp = torch.from_numpy(m_train_sp_).type(torch.float32).to(device)
x_valid_sp = torch.from_numpy(x_valid_sp_).type(torch.float32).to(device)
y_valid_sp = torch.from_numpy(y_valid_sp_).type(torch.float32).to(device)
m_valid_sp = torch.from_numpy(m_valid_sp_).type(torch.float32).to(device)
x_test_sp = torch.from_numpy(x_test_sp_).type(torch.float32).to(device)
y_test_sp = torch.from_numpy(y_test_sp_).type(torch.float32).to(device)
m_test_sp = torch.from_numpy(m_test_sp_).type(torch.float32).to(device)
# Send the model to device.
model1.to(device)

# Set optimizaton methods
optimizer = optim.Adam(model1.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.5)
early_stopping = EarlyStopping(patience=patience)

epoch_loss_train = []
epoch_loss_valid = []
epoch_loss_test = []
loss_val_best = 500000
epoch_best = 0
train_time = 0

for epoch in range(epochs):
    starttime=time.time()
    # Set the model in training mode.
    model1.train()
    # SSMB does not shuffle the samples 
    idx = torch.arange(x_train_sp.size()[0]).to(device) 
    hidden_head = model1.init_hidden(1)
    for j in range(0,num_train_samples,batch_size):
        # Sets the gradients of all optimized torch.Tensors to zero.
        model1.zero_grad()
        sj = j
        ej = min([j+batch_size,num_train_samples])
        batch_x = x_train_sp[idx[sj:ej],:,:]
        batch_y = y_train_sp[idx[sj:ej],:,:]
        batch_m = m_train_sp[idx[sj:ej],:,:]
        batch_pred = torch.zeros(batch_y.size()).type(torch.float32).to(device)
        
        for bch in range(len(idx[sj:ej])):
            bch_x = torch.unsqueeze(batch_x[bch,:,:],0)
            bch_y = torch.unsqueeze(batch_y[bch,:,:],0)
            bch_m = torch.unsqueeze(batch_m[bch,:,:],0)
            pred,hiddens = model1(bch_x,hidden_head)
            batch_pred[bch] = pred[0]
            hiddens.detach_()
            # The first segment of the first minibatch is often not aligned with the second segment.
            if j==0 and bch == 0:
                hidden_head = model1.init_hidden(1)
            # Keep the last hidden states of the previous sequence to be the initial hidden states of the current sequence.
            hidden_head = torch.unsqueeze(hiddens[:,-1,:],0)
        ## MTL losses
        # batch_losses = torch.zeros(n_classes)
        # for k in range(n_classes):
        #     batch_losses[k] = MSE(pred[:,:,k], batch_y[:,:,k], batch_m[:,:,k])
        # batch_loss = torch.sum(batch_losses)
        batch_loss = MSE(batch_pred, batch_y, batch_m)
        batch_loss.backward()
        optimizer.step()
    scheduler.step()
    endtime=time.time()
    train_time = train_time + endtime - starttime
    
    # Set the model in Validation mode.
    model1.eval()
    with torch.no_grad():
        # Apply SSIF as the validation method. 
        hidden_head = model1.init_hidden(1)
        pred_train,hiddens = model1(x_train_,hidden_head)
        hidden_head = model1.init_hidden(1)
        pred_valid,hiddens = model1(x_valid_,hidden_head)
        hidden_head = model1.init_hidden(1)
        pred_test,hiddens = model1(x_test_,hidden_head)
        
        epoch_loss_train.append(MSE(pred_train, y_train_, m_train_).cpu().numpy())
        epoch_loss_valid.append(MSE(pred_valid, y_valid_, m_valid_).cpu().numpy())
        epoch_loss_test.append(MSE(pred_test, y_test_, m_test_).cpu().numpy())
        
        # Save the model if the best MSE is decreased on the validataion set.
        if epoch_loss_valid[epoch] < loss_val_best:
            epoch_best = epoch
            loss_val_best=epoch_loss_valid[epoch]
            path_save = exp_dir+"run_iter_{}_best_model.sav".format(run_iter)
            torch.save({'epoch': epoch,
                    'model_state_dict': model1.state_dict(),
                    'loss': epoch_loss_train[epoch],
                    'los_val': epoch_loss_test[epoch],
                    }, path_save)
        print("finished training epoch", epoch+1)
        print("Epoch {} : epoch_loss_train RMSE loss {:.4f}".format(str(epoch), np.sqrt(epoch_loss_train[epoch])))
        print("Epoch {} : epoch_loss_valid RMSE loss {:.4f}".format(str(epoch), np.sqrt(epoch_loss_valid[epoch])))
        print("Epoch {} : epoch_loss_test RMSE loss {:.4f}".format(str(epoch), np.sqrt(epoch_loss_test[epoch])))
        print("loss_val_best : {:.4f}".format(np.sqrt(loss_val_best)))
        early_stopping(epoch_loss_valid[epoch])
        assert(early_stopping.best_loss == loss_val_best)
        if early_stopping.early_stop:
            break
        print()

path_save = exp_dir+"run_iter_{}_final_model.sav".format(run_iter)
torch.save({'epoch': epoch,
            'train_losses': epoch_loss_train,
            'val_losses': epoch_loss_valid,
            'test_losses': epoch_loss_test,
            'train_time': train_time,
            'model_state_dict_fs': model1.state_dict(),
            }, path_save)
print("final train_loss:",epoch_loss_train[-1],"val_loss:",epoch_loss_test[-1],"loss validation best:",loss_val_best)
print("total Training time: {:.4f}s".format(train_time))
print("Training time: {:.4f}s/epoch".format(train_time/epoch))

# Plot the learning curve
epoch_loss_train_arr = np.squeeze(epoch_loss_train)
epoch_loss_valid_arr = np.squeeze(epoch_loss_valid)
epoch_loss_test_arr = np.squeeze(epoch_loss_test)
plt.vlines(epoch_best, 0, np.max(np.sqrt(epoch_loss_valid_arr)), colors='r')
plt.plot(np.sqrt(epoch_loss_train_arr), label='training RMSE loss')
plt.plot(np.sqrt(epoch_loss_valid_arr), label='validation RMSE loss')
plt.plot(np.sqrt(epoch_loss_test_arr), label= 'testing RMSE loss')
plt.legend()
plt.savefig(exp_dir+'run_iter_{}_learning_curve.png'.format(run_iter))
plt.close()
