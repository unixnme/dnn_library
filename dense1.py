# this will implement the simplest dense layer neural network

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_valid = mnist.validation.images
y_valid = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# create model
dim_in = x_train.shape[1]
dim_out = y_train.shape[1]

EPOCHS = 5
BATCH_SIZE = 10
LR = 1e-3

def generator(x, y):
    num_samples = len(x)
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(num_samples):
            end = np.minimum(num_samples, start + BATCH_SIZE)
            elems = indices[start:end]
            yield x[elems], y[elems]

num_batches_per_epoch = int(np.ceil(len(x_train) / float(BATCH_SIZE)))
train_gen = generator(x_train, y_train)


def train_keras_model(model):
    model.compile(optimizer=SGD(lr=LR), loss='mse', metrics=['accuracy'])
    model.fit_generator(generator=train_gen, steps_per_epoch=num_batches_per_epoch, epochs=EPOCHS,
                        verbose=2, validation_data=(x_valid, y_valid))

def train_numpy_model(model):
    weights = model.layers[1].get_weights()
    w = weights[0]
    b = weights[1].reshape(1, -1)
    for epoch in range(EPOCHS):
        cost = []
        accuracy = []
        for batch in range(num_batches_per_epoch):
            x, z = next(train_gen)
            y = x.dot(w) + b
            d = y - z
            cost.append(np.square(d).mean())

            grad_d = 2*d/np.float(np.prod(d.shape))
            grad_y = grad_d
            grad_w = x.T.dot(grad_y)
            grad_b = np.sum(grad_y, axis=0, keepdims=True)

            acc = np.argmax(y, axis=-1) == np.argmax(z, axis=-1)
            accuracy.append(acc.astype(np.float32).mean())

            w -= LR * grad_w
            b -= LR * grad_b
        x, z = x_valid, y_valid
        y = x.dot(w) + b
        d = y - z
        val_cost = np.square(d).mean()
        val_acc = np.argmax(y, axis=-1) == np.argmax(z, axis=-1)
        val_acc = val_acc.astype(np.float32).mean()
        print epoch, np.mean(cost), np.mean(accuracy), val_cost, val_acc

if __name__ == '__main__':
    x_in = Input(shape=(dim_in,))
    x = Dense(dim_out)(x_in)
    model = Model(inputs=x_in, outputs=x)

    train_numpy_model(model)
    train_keras_model(model)