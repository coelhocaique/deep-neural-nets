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
returns = np.array([(ret - mean) / standard_deviation for ret in returns])

print returns.shape
