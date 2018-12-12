def train_test_shuffled_separation(data, label, train_percent=0.8, valid_percent=0.1):
    """
    """
    import numpy as np

    # Randomize training set and corresponding labels
    rand_set = np.hstack((label, data))
    np.random.shuffle(rand_set)
    data = rand_set[:, 1:data.shape[1] + 1]
    label = rand_set[:, 0]
    print("shuffled data shape:", data.shape, "shuffled label shape:", label.shape)

    # specify train and test sizes
    train_length = int(train_percent * data.shape[0])
    valid_length = int(valid_percent * train_length)

    # index first 80% for training, last 20% for test
    # also index last 20% of trianing as validation
    data_train = data[0: train_length, :]
    label_train = label[0: train_length]

    data_test = data[train_length:, :]
    label_test = label[train_length:]

    data_valid = data_train[0: valid_length, :]
    label_valid = label_train[0: valid_length]
    data_train = data_train[valid_length:, :]
    label_train = label_train[valid_length:]

    print('# train:', data_train.shape[0])
    print('# valid:', data_valid.shape[0])
    print('# test:', data_test.shape[0])
    print('# total:', data.shape[0])

    return data_train, label_train, data_valid, label_valid, data_test, label_test


def normalize_data(data, min=0, max=1):
    """
    takes a collection of images (such as xtrain) and return a modified version in the range [-1, 1]
    of type float64.
    :param x (np.ndarray): collection of images in 2d array, each image index is x[:, n]
    :return: (np.ndarray) normalized collection of images in type float64
    """
    import numpy as np
    assert isinstance(data, np.ndarray)

    max_value = np.max(data)
    min_value = np.min(data)

    scaled = np.interp(data, [min_value, max_value], [min, max])
    # convert to float64
    scaled = scaled.astype(np.float64)

    return scaled
