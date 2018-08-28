import numpy as np
import cv2


def generator(data, batch_size=32, img_size=64):
    shuffle_indices = np.random.randint(
            low=0,
            high=data.shape[1],
            size=data.shape[1] - 1)
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
