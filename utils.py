import numpy as np


def generator(data, batch_size=32):
    shuffle_indices = np.random.randint(
            low=0,
            high=data.shape[1],
            size=data.shape[1] - 1
            )
    while True:
        batch = []
        for indices in shuffle_indices:
            x1 = data[:10, indices, :, :]
            x2 = data[10:, indices, :, :]
            batch.append(x1)
            batch.append(x2)
            if len(batch) == batch_size:
                np_batch = np.array(batch)
                np_batch = np.reshape(batch, (batch_size, 10, 64, 64, 1))
                batch = []
                yield np_batch