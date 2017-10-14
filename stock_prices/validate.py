import lstm
import datetime
import numpy as np
import dataset_loader as data

print (datetime.datetime.now().time(), 'Start validating ...')

true_data = data.load_data('dataset/train.csv')
studied_data, full_batch_size = data.study_period(true_data)
reshaped_data, labels = data.fit_to_shape(studied_data, full_batch_size, 'predict', timesteps=100)

model = lstm.load_model('models/EUR_USD_STRATEGY_1_M1_1507592125.0028422_acc_83.4889700056.h5')
predictions = lstm.predict(model=model, data=reshaped_data)

prediction_length = len(predictions)
right_answer = 0
distribution = [[0], [0], [0]]
real_distribution = [[0], [0], [0]]
for i in range(len(predictions)):
    prediction = predictions[i]
    label = labels[i]
    value = np.argmax(prediction)
    true_value = np.argmax(label)

    if value == true_value:
        right_answer += 1

    distribution[value][0] += 1
    real_distribution[true_value][0] += 1

    print ('Prediction %i : %g: , True value: %g' % (i, value, true_value))

print ('Score: %.2f %s' % (float(right_answer / prediction_length) * 100, '%'))
print ('Score distribution: \n %g up , %g middle, %g down' % (distribution[0][0], distribution[1][0], distribution[2][0]))
print ('True data distribution: \n %g up , %g middle, %g down' % (real_distribution[0][0], real_distribution[1][0], real_distribution[2][0]))

print (datetime.datetime.now().time(), 'finishes validating .')