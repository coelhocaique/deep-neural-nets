import numpy as np

f = open("itau_2009-02-02_2017-03-06_closing_price.csv", 'r').read()
data = f.split('\n')

profit = []
for i in range(1,len(data)):
    try:
        current_return = (float(data[i])/float(data[i-1])) - 1
        profit.append(current_return)
    except ValueError,e:
        print "error",e,"on line",i

returns = np.array(profit)
mean = np.mean(returns)
standard_deviation =  np.std(returns)
returns = [(ret - mean) / standard_deviation for ret in returns]
labels = []
sequences = []
batch_size = 0
for i in range(0,len(returns) - 2):
    try:
        new_returns = np.array(returns[i:240 + i])
        mean = np.mean(new_returns)
        if returns[240 + i] > mean: label = 1
        else: label = -1
        labels.append(label)
        sequences.append(new_returns)
        t_plus_1 = returns[240 + i]
        #print "Sequence ",i ,"\nmean :",mean,"\nlabel(t+1): ",label,t_plus_1,"\n","subtracted: ",(t_plus_1 - mean)
        #print "real profit: ",profit[239 + i],"\n"
    except IndexError:
        batch_size = i
        break

print len(labels) == len(sequences)
print batch_size
#print sequences[0]
input_data = np.zeros((batch_size,240,1))
#input_data[0] = sequences[0]
print input_data.shape
print len(input_data[0])
print input_data[0].shape
for i in range(len(sequences)):
    input_data[i] = sequences[i].reshape(len(sequences[i]),1)
print input_data[1759],labels[1759]
