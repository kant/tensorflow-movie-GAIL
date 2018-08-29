import random
import cv2
import numpy as np


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def generator(data, batch_size=32, img_size=64):
    reset_seed(seed=0)
    train_count = len(data) * 0.9
    shuffle_indices = np.random.randint(
            low=0,
            high=data.shape[1],
            size=int(train_count))
    while True:
        batch = []
        for indices in shuffle_indices:
            x1 = data[:10, indices, :, :]
            x2 = data[10:, indices, :, :]
            x1 = [cv2.resize(x, (img_size, img_size)) for x in x1]
            x2 = [cv2.resize(x, (img_size, img_size)) for x in x2]
            x1 = np.reshape(np.array(x1), (10, img_size, img_size, 1))
            x2 = np.reshape(np.array(x2), (10, img_size, img_size, 1))
            batch.append(x1)
            batch.append(x2)
            if len(batch) == batch_size:
                np_batch = np.array(batch)
                np_batch = np_batch / 255
                batch = []
                yield np_batch
