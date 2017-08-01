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
for i in range(0,len(returns) - 2):
    try:
        new_returns = np.array(returns[i:239 + i])
        mean = np.mean(new_returns)
        if returns[239 + i] > mean: label = 1
        else: label = -1
        labels.append(label)
        sequences.append(new_returns)
        t_plus_1 = returns[239 + i]
        print "Sequence ",i ,"\nmean :",mean,"\nlabel(t+1): ",label,t_plus_1,"\n","subtracted: ",(t_plus_1 - mean)
        print "real profit: ",profit[239 + i],"\n"
    except IndexError:
        break

print len(labels) == len(sequences)
