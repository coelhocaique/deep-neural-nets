import numpy as np

def fit_to_shape(data,full_batch_size,timesteps=240,input_dim=1):
    labels = []
    sequences = []

    for i in range(full_batch_size):
        #the paper extracts the cross-section median, but it not applied for us
        #we just extracts the median
        sequence = np.array(data[i:timesteps + i])
        mean = np.mean(sequence)
        if data[timesteps + i] > mean:
            label = 1
        else:
            label = 0
        labels.append(label)
        sequences.append(sequence)

    labels = np.array(labels)
    reshaped_data = np.empty((full_batch_size,timesteps,input_dim))
    for i in range(len(sequences)):
        reshaped_data[i] = sequences[i].reshape(len(sequences[i]),1)
    #75% is for training, 25% validating
    train_index = int(full_batch_size * 0.75)
    train_data,train_labels = reshaped_data[:train_index],labels[:train_index]
    validate_data,validate_labels =reshaped_data[train_index:],labels[train_index:]

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

f = open("dataset/itau_2009-02-02_2017-03-06_closing_price.csv", 'r').read()
data = f.split('\n')
train_data,train_labels,validate_data,validate_labels = study_period(data)

# profit = []
# for i in range(1,len(data)):
#     try:
#         current_return = (float(data[i])/float(data[i-1])) - 1
#         profit.append(current_return)
#     except ValueError,e:
#         print "error",e,"on line",i

# returns = np.array(profit)
# mean = np.mean(returns)
# standard_deviation =  np.std(returns)
# returns = [(ret - mean) / standard_deviation for ret in returns]

#print 'train_data: ',train_data[0]
f = open('train_mean.txt','w')
for label in train_labels:
    f.write(str(label) + '\n')
f.close()
f = open('validate_mean.txt','w')
for label in validate_labels:
    f.write(str(label) + '\n')
f.close()
#print 'train_labels: ',train_labels
#print 'validate_data: ',validate_data[0]
#print 'validate_labels: ',validate_labels
# labels = []
# sequences = []
# batch_size = 0
# for i in range(0,len(returns) - 2):
#     try:
#         new_returns = np.array(returns[i:240 + i])
#         mean = np.mean(new_returns)
#         if returns[240 + i] > mean: label = 1
#         else: label = -1
#         labels.append(label)
#         sequences.append(new_returns)
#         t_plus_1 = returns[240 + i]
#         #print "Sequence ",i ,"\nmean :",mean,"\nlabel(t+1): ",label,t_plus_1,"\n","subtracted: ",(t_plus_1 - mean)
#         #print "real profit: ",profit[239 + i],"\n"
#     except IndexError:
#         batch_size = i
#         break
#
# print len(labels) == len(sequences)
# print batch_size
# #print sequences[0]
# input_data = np.zeros((batch_size,240,1))
# #input_data[0] = sequences[0]
# print input_data.shape
# print len(input_data[0])
# print input_data[0].shape
# for i in range(len(sequences)):
#     input_data[i] = sequences[i].reshape(len(sequences[i]),1)
# print input_data[1759:]
