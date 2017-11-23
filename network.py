import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as im

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))


class perceptron(object):

    def __init__(self, sizes=list(), learning_rate=.8, mini_batch_size=16,epochs=10):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.array([0])] + [np.random.randn(y, x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(y, 1) for y in sizes]

        self._zs = [np.zeros(bias.shape) for bias in self.biases]ch 72, accuracy 53.32 %.

        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta = learning_rate
        self.error_list=[]
        self.accuracy_list=[]

    def fit(self, training_data, validation_data=None):
        for epoch in range(self.epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]
            b_error=[]
            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                
                for x, y in mini_batch:
                    self.forward_prop(x)
                    err_i=[]
                    delta_nabla_b, delta_nabla_w,err_i = self.back_prop(x, y,err_i)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                b_error.append(self.error_LMS(err_i))
                self.weights = [
                    w - (self.eta / self.mini_batch_size) * dw for w, dw in
                    zip(self.weights, nabla_w)]
                self.biases = [
                    b - (self.eta / self.mini_batch_size) * db for b, db in
                    zip(self.biases, nabla_b)]
            self.error_list.append(self.error_LMS(b_error))
            if validation_data:
                accuracy = self.validate(validation_data) / 100.0
                self.accuracy_list.append(accuracy)
                print("Epoch {0}, accuracy {1} %.".format(epoch + 1, accuracy))
            else:
                print("Processed epoch {0}.".format(epoch))

    def validate(self, validation_data):
        validation_results = [(self.predict(x) == y) for x, y in validation_data]
        return sum(result for result in validation_results)        
    def predict(self, x):
        self.forward_prop(x)
        return np.argmax(self._activations[-1])
    def predict_file(self,pathh):
        img=im.imread(pathh)
        img=img.reshape(28*28,1)
        return self.predict(img)

    def forward_prop(self, x):
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i - 1]) + self.biases[i]
            )
            self._activations[i] = sigmoid(self._zs[i])
    def plot_error(self):
        plt.plot(self.error_list)
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.accuracy_list)
        plt.show()
        
    def error_LMS(self,a):
        return 0.5*np.sum(np.power((a),2))/len(a)

    def back_prop(self, x, y,err_l):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y) * sigmoid_deriv(self._zs[-1])
        err_l.append(self.error_LMS(self._activations[-1] - y))
        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                sigmoid_deriv(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w,err_l

    def load(self, filename='model.npz'):
        npz_members = np.load(os.path.join(os.curdir, 'models', filename))

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = int(npz_members['mini_batch_size'])
        self.epochs = int(npz_members['epochs'])
        self.eta = float(npz_members['eta'])

    def save(self, filename='model.npz'):

        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            mini_batch_size=self.mini_batch_size,
            epochs=self.epochs,
            eta=self.eta
        )
