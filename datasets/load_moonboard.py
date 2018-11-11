import numpy as np

def load_moonboard(filename='moonboard.npz'):
    f = np.load(filename)
    x_train, y_train = f['x_train'], f['y_train']
    x_test,  y_test  = f['x_test'],  f['y_test']
    return (x_train, y_train), (x_test, y_test)

