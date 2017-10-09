import numpy as np


def load_data(path):
    f = open(path, 'r').read()
    return f.split('\n')


def fit_to_shape(data, full_batch_size, type_net, labels_in=None, timesteps=240, input_dim=1):
    labels = []
    sequences = []

    if labels_in:
        print ('training from raw labels')

    for i in range(full_batch_size):
        sequence = np.array(data[i:timesteps + i])
        last_value = sequence[len(sequence) - 1]
        next_value = data[timesteps + i]

        if labels_in:
            label = labels_in[timesteps + i]
        else:
            if next_value == last_value:
                label = [0, 1, 0]
            elif next_value > last_value:
                label = [0, 0, 1]
            else:
                label = [1, 0, 0]
            # label = 1 if next_value > last_value else 0

        labels.append(label)
        sequences.append(sequence)

    labels = np.array(labels)
    reshaped_data = np.empty((full_batch_size, timesteps, input_dim))

    for i in range(len(sequences)):
        reshaped_data[i] = sequences[i].reshape(len(sequences[i]), input_dim)

    if type_net == 'train':
        # 80% is for training, 20% validating
        train_index = int(full_batch_size * 0.80)
        train_data = reshaped_data[:train_index]
        train_labels = labels[:train_index]
        validate_data = reshaped_data[train_index:]
        validate_labels = labels[train_index:]

        return train_data, train_labels, validate_data, validate_labels

    elif type_net == 'predict':
        return reshaped_data, labels


def study_period(data, timesteps=240, m=1, input_dim=1):
    """
        input_shape = (batch_size, timesteps, input_dim)`
        input_shape = [N ,240,1]
    """
    # calculate return
    profit = []
    for i in range(1, len(data) - 1):
        try:
            current_return = (float(data[i])/float(data[i-m])) - 1
            # current_return = float(data[i])
            profit.append(current_return)
        except (ValueError, e):
            print ("error on line", i)

    # standardize return
    returns = np.array(profit)
    mean = np.mean(returns)
    standard_deviation = np.std(returns)
    min_num = min(returns)
    max_num = max(returns)
    # standardize
    returns = [(ret - mean) / standard_deviation for ret in returns]
    # normalise value between 0 and 1
    returns = [(ret - min_num) / (max_num - min_num) for ret in returns]
    full_batch_size = len(profit) - timesteps

    return returns, full_batch_size


def to_binary(sequence):
    return_sequence = []
    for seq in sequence:
        if seq > 0:
            return_sequence.append(1)
        else:
            return_sequence.append(-1)

    return np.array(return_sequence)


def get_labels(data):
    labels = []
    for i in range(1, len(data) - 1):
        if float(data[i]) > float(data[i - 1]):
            labels.append(1)
        else:
            labels.append(0)
    return labels


def load(path, type_net, study_data=True, timesteps=240):
    data = load_data(path)
    if study_data:
        real_labels = get_labels(data)
        returns, full_batch_size = study_period(data, timesteps=timesteps)
        return fit_to_shape(returns,
                            full_batch_size,
                            type_net,
                            timesteps=timesteps)
    else:
        real_labels = get_labels(data)
        in_data = []
        for i in range(1, len(data) - 1):
            in_data.append(float(data[i]))
        return fit_to_shape(in_data,
                            (len(in_data) - timesteps),
                            type_net,
                            labels_in=real_labels,
                            timesteps=timesteps)
