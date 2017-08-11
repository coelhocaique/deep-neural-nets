import numpy as np

def load_data(path):
    f = open(path,'r').read()
    return f.split('\n')


def fit_to_shape(data,full_batch_size,timesteps=240,input_dim=1):
    labels = []
    sequences = []

    for i in range(full_batch_size):
        #the paper extracts the cross-section median, but it not applied for us
        #we just extracts the median
        sequence = np.array(data[i:timesteps + i])
        mean = np.median(sequence)
        if data[timesteps + i] > mean:
            label = 1
        else:
            label = 0

        labels.append(label)
        sequences.append(sequence)

    labels = np.array(labels)
    reshaped_data = np.empty((full_batch_size,timesteps,input_dim))
    for i in range(len(sequences)):
        reshaped_data[i] = sequences[i].reshape(len(sequences[i]),input_dim)

    #80% is for training, 20% validating
    train_index = int(full_batch_size * 0.80)
    train_data = reshaped_data[:train_index]
    train_labels = labels[:train_index]
    validate_data = reshaped_data[train_index:]
    validate_labels = labels[train_index:]

    return train_data,train_labels,validate_data,validate_labels


def study_period(data,timesteps = 240,m = 1,input_dim=1):
    '''
        input_shape = (batch_size, timesteps, input_dim)`
        input_shape = [N ,240,1]
    '''
    #calculate return
    profit = []
    for i in range(1,len(data)):
        try:
            current_return = (float(data[i])/float(data[i-m])) - 1
            profit.append(current_return)
        except ValueError,e:
            print "error",e,"on line",i

    #standardize return
    returns = np.array(profit)
    mean = np.mean(returns)
    standard_deviation =  np.std(returns)
    returns = [(ret - mean) / standard_deviation for ret in returns]
    full_batch_size = len(profit) - timesteps

    return fit_to_shape(returns,full_batch_size)

def load(path,study_data = True):
    data = load_data(path)
    if study_data:
        return study_period(data)
    return data
