import numpy as np
from datetime import datetime

def load_data(df, output_vars, input_vars, input_size):
    ''' 
    Load data and separate data into features and labels.
    Parameters:
        df: a pandas data frame
        output_vars: A list contains the names of output variables.
        input_vars: A list contains the names of input variables. 
    Return:
        feat: an input data matrix
        labels: an output data matrix
    '''
    N = df.shape[0]
    feat = np.zeros((N,input_size))
    # Convert the dates to the day of the year(DOY), then bend the DOY at day 183 to give DOY a seasonal pattern as shown in figure 
    dates = df['Date'].values
    doy = np.array([datetime.strptime(cdate, '%Y-%m-%d').date().timetuple().tm_yday for cdate in dates])
    doy = 183 - np.abs(doy - 183)
    # Load features
    feat = df[input_vars].values
    feat[:,0] = doy * 1.0

    labels = np.zeros((N,len(output_vars)))
    for i, var in enumerate(output_vars):
        if var in df.keys():
            labels[:,i] = df[var]
        elif var == 'SW_Delta':
            labels[:,i] = df['SW_ENDmm'].values - df['SW_INITmm'].values
        else:
            raise ValueError('Unexpected output variables {}'.format(var))
    
    return feat, labels

def SampleData(train_idx,n_steps,shift,num_samples):
    ''' 
    Generate arrays of indexes to sample data
    train_idx: an index array of data
    n_steps: sequence length (aka window size)
    shift: shift length.
        shift>0 : regular shift sampling
        shift=0 : random sampling
    num_samples: Is activated when shift=0. It determines the number of random samples.
    
    '''
    tN = train_idx.shape[0]
    print(tN)
    if shift>0:
        for i in np.arange(0,tN, shift):
            if i + n_steps > tN:
                break
            cur_idx = np.expand_dims(train_idx[i:i+n_steps],axis=0)
            if i == 0:
                train_idx_arr = cur_idx
            else:
                train_idx_arr = np.concatenate((train_idx_arr,cur_idx),axis=0)
        if not np.all(cur_idx == train_idx[-n_steps:]):
            cur_idx = np.expand_dims(train_idx[-n_steps:],axis=0)
            train_idx_arr = np.concatenate((train_idx_arr,cur_idx),axis=0)

    if shift==0:
        start_idx = np.arange(0,tN-n_steps)
        start_idx = np.random.permutation(start_idx)
        num_samples = min([num_samples,start_idx.shape[0]])
        train_idx_arr = np.zeros((num_samples,n_steps)).astype(int)
        for i in range(num_samples):
            train_idx_arr[i,:] = train_idx[start_idx[i]:start_idx[i]+n_steps]
            
    return train_idx_arr

def init_one_feat(feature,idx):
    '''
    This function helps to create arrays to contain copied initial values in data for CMB training algorithm
    Input:
        feature: an input data matrix
        idx: the feature index that needs to be converted to an initial values array.
    Return:
        res: the converted input data matrix.
    '''
    res = np.copy(feature)
    dup = res[:,0,idx].reshape(-1,1)
    n_steps = feature.shape[1]
    dup = np.repeat(dup, n_steps, axis=1)
    res[:,:,idx] = dup
    return res