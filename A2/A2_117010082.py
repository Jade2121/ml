import numpy as np
import matplotlib.pyplot as plt

def load_data():
    train = []
    test = []
    f = open("train1.txt", "r")
    for l in f:
        if not (l.startswith('//') or l.startswith('**') or l.startswith('##') or (not l[0].isdigit())):
            lis = l[:-1].split()
            lis = [float(i) for i in lis]
            train.append(lis)
    f.close()
    f = open("test1.txt", "r")
    for l in f:
        if not (l.startswith('//') or l.startswith('**') or l.startswith('##') or (not l[0].isdigit())):
            lis = l[:-1].split()
            lis = [float(i) for i in lis]
            test.append(lis)
    f.close()
    train = np.array(train)
    test = np.array(test)
    return (train[:, :-3], train[:, -3:]), (test[:, :-3], test[:, -3:])

training_data,test_data = load_data()

def drelu(x):
    return (x > 0).astype(int)

def relu(x):
    return x * ((x > 0).astype(int))

def MSE(a, y):
    softmax = np.exp(a) / sum(np.exp(a))
    M = -np.dot(softmax.reshape(np.size(softmax), 1), softmax.reshape(1, np.size(softmax)))
    for i in range(np.size(softmax)):
        M[i, i] = softmax[i] * (1 - softmax[i])
    softmax[np.array(range(3))[y == 1]] = softmax[np.array(range(3))[y == 1]] - 1
    return np.dot(M, softmax)

class NN:
    def __init__(self, training_data, validation_data, num_layers, num_nodes, learning_rate, epoch, batch_size):
        self.training_data = training_data
        self.validation_data = validation_data
        self.num_layers = num_layers
        self.activ = (relu, drelu)
        self.num_nodes = num_nodes
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.cost = MSE
        self.W = []
        # Kaiming initialization.
        for i in range(num_layers + 1):
            if i == 0:
                self.W.append(np.random.uniform(-(6 / (256 + num_nodes)) ** 0.5, (6 / (256 + num_nodes)) ** 0.5, 257 * num_nodes).reshape(num_nodes, 257))
            elif i == num_layers:
                self.W.append(np.random.uniform(-(6 / (3 + num_nodes)) ** 0.5, (6 / (3 + num_nodes)) ** 0.5, 3 * (num_nodes + 1)).reshape(3, num_nodes + 1))
            else:
                self.W.append(np.random.uniform(-(6 / (num_nodes + num_nodes)) ** 0.5, (6 / (num_nodes + num_nodes)) ** 0.5, num_nodes ** 2 + num_nodes).reshape(num_nodes, num_nodes + 1))
    
    def forward(self, data):
        A = []
        Z = [data.T]
        for i in self.W:
            A.append(np.dot(i, np.append(Z[-1], np.ones((1, np.shape(data)[0])), axis = 0)))
            Z.append(self.activ[0](A[-1]))
        Z = Z[:-1]
        return A, Z
        
    def backward(self, response, A, Z):
        dW = []
        delta = np.array([[]] * np.shape(A[-1])[0])
        for i in range(np.shape(A[-1])[1]):
            delta = np.append(delta, self.cost(A[-1][:,i], response[i]).reshape(np.shape(A[-1])[0], 1), axis = 1)
        for i in range(len(self.W)):
            dW = [np.dot(delta, np.append(Z[-(i + 1)].T, np.ones(np.shape(Z[-(i + 1)])[1]).reshape(np.shape(Z[-(i + 1)])[1], 1), axis = 1)) / np.size(response)] + dW
            if i != len(self.W) - 1:
                delta = self.activ[1](A[-(i + 2)]) * np.dot(self.W[-(i + 1)].T[:-1], delta)
        return dW
    
    # Trian the model and use the trained self.W to do the prediction after finishing all the epochs. When initializing a NN object, if the test data is incorporated into the "validation_data" parameter, then it can be used to do the prediction.
    def predict(self):
        for i in range(self.epoch):
            batches = np.array(range(len(self.training_data[0])))
            for i in range(len(self.training_data[0]) // self.batch_size):
                batch = (self.training_data[0][batches[(i * self.batch_size):((i + 1) * self.batch_size)]], self.training_data[1][batches[(i * self.batch_size):((i + 1) * self.batch_size)]])
                A, Z = self.forward(batch[0])
                dW = self.backward(batch[1], A, Z)
                for j in range(len(self.W)):
                    self.W[j] -= self.learning_rate * dW[j]
            if len(self.training_data[0]) % self.batch_size != 0:
                A, Z = self.forward(self.training_data[0][batches[len(self.training_data[0]) // self.batch_size * self.batch_size:]])
                dW = self.backward(self.training_data[1][batches[len(self.training_data[0]) // self.batch_size * self.batch_size:]], A, Z)
                for j in range(len(self.W)):
                    self.W[j] -= self.learning_rate * dW[j]
        a6 = 0
        a8 = 0
        a9 = 0
        # The "validation_data" in the following is actually test data.
        pred = self.forward(self.validation_data[0])[0][-1].T
        for i in range(np.shape(pred)[0]):
            if np.array(range(3))[pred[i] == np.max(pred[i])] == np.array(range(3))[self.validation_data[1][i] == 1]:
                if np.array(range(3))[pred[i] == np.max(pred[i])] == 0:
                    a6 += 1
                elif np.array(range(3))[pred[i] == np.max(pred[i])] == 1:
                    a8 += 1
                else:
                    a9 += 1
        return np.array([a6 / sum(self.validation_data[1][:,0]), a8 / sum(self.validation_data[1][:,1]), a9 / sum(self.validation_data[1][:,2]), (a6 + a8 + a9) / len(self.validation_data[0])])

a6 = []
a8 = []
a9 = []
a = []
for i in [200, 100, 50, 20, 10]:
    accuracy = NN(training_data, training_data, 1, i, 0.5, 5, 20).predict()
    a6.append(accuracy[0])
    a8.append(accuracy[1])
    a9.append(accuracy[2])
    a.append(accuracy[3])
plt.plot(a6, 'k-', color = "blue")
plt.plot(a8, 'k-',color = "yellow")
plt.plot(a9, 'k-',color = "red")
plt.plot(a, 'k-',color = "black")
plt.legend(["6", "8", "9", "total"], loc = 4)
plt.show()
plt.cla()

a6 = []
a8 = []
a9 = []
a = []
for i in [200, 100, 50, 20, 10]:
    accuracy = NN(training_data, test_data, 1, i, 0.5, 10, 20).predict()
    a6.append(accuracy[0])
    a8.append(accuracy[1])
    a9.append(accuracy[2])
    a.append(accuracy[3])
plt.plot(a6, 'k-', color = "blue")
plt.plot(a8, 'k-',color = "yellow")
plt.plot(a9, 'k-',color = "red")
plt.plot(a, 'k-',color = "black")
plt.legend(["6", "8", "9", "total"], loc = 4)
plt.show()

# Use h = 200
print(a6[0], a8[0], a9[0], a[0])

