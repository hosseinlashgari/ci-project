import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


f = open("Datasets/train_set_features.pkl", "rb")
train_set_features2 = pickle.load(f)
f.close()
features_STDs = np.std(a=train_set_features2, axis=0)
train_set_features = train_set_features2[:, features_STDs > 52.3]
train_set_features = np.divide(train_set_features, train_set_features.max())
f = open("Datasets/train_set_labels.pkl", "rb")
train_set_labels = pickle.load(f)
f.close()
f = open("Datasets/test_set_features.pkl", "rb")
test_set_features2 = pickle.load(f)
f.close()
features_STDs = np.std(a=test_set_features2, axis=0)
test_set_features = test_set_features2[:, features_STDs > 48]
test_set_features = np.divide(test_set_features, test_set_features.max())
f = open("Datasets/test_set_labels.pkl", "rb")
test_set_labels = pickle.load(f)
f.close()
train_set = []
test_set = []
for i in range(len(train_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(train_set_labels[i])] = 1
    label = label.reshape(4, 1)
    train_set.append((train_set_features[i].reshape(102, 1), label))
for i in range(len(test_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(test_set_labels[i])] = 1
    label = label.reshape(4, 1)
    test_set.append((test_set_features[i].reshape(102, 1), label))
print(train_set[0])

# Number of code executions
iteration_number = 10
# Number of data for training
train_number = len(train_set)
epochs = 10
batch_size = 10
learning_rate = 1
# Learning rate method: 0 = divide by batch size , 1 = divide by gradian norm
learning_rate_method = 1
is_vectorized = True

before_training_precision = []
after_training_train_set_precision = []
after_training_test_set_precision = []
costs_avg = np.zeros(epochs)
training_time = []

for iteration in range(iteration_number):
    random.shuffle(train_set)
    random.shuffle(test_set)
    W = [np.random.randn(150, 102), np.random.randn(60, 150), np.random.randn(4, 60)]
    B = [np.zeros((150, 1)), np.zeros((60, 1)), np.zeros((4, 1))]
    A = []

    # first feed forward
    for i in range(train_number):
        A.append([train_set[i][0]])
        for j in range(3):
            A[i].append(sigmoid(W[j] @ A[i][j] + B[j]))

    # before training test
    true = 0
    for i in range(train_number):
        if np.argmax(train_set[i][1]) == np.argmax(A[i][3]):
            true += 1
    before_training_precision.append(true / train_number)

    start_time = time.time()
    for epoch in range(epochs):
        for n in range(0, train_number - batch_size + 1, batch_size):
            if is_vectorized:
                G_W = [np.zeros((150, 102)), np.zeros((60, 150)), np.zeros((4, 60))]
                G_B = [np.zeros((150, 1)), np.zeros((60, 1)), np.zeros((4, 1))]
                G_A0 = np.zeros((batch_size, 150, 1))
                G_A1 = np.zeros((batch_size, 60, 1))
                G_A2 = np.zeros((batch_size, 4, 1))
                for i in range(n, n + batch_size):
                    for j in range(3):
                        A[i][j + 1] = sigmoid(W[j] @ A[i][j] + B[j])
                # backpropagation
                for k in range(n, n + batch_size):
                    G_A2[k - n] = 2 * (A[k][3] - train_set[k][1])
                    b_temp = G_A2[k - n] * A[k][3] * (1 - A[k][3])
                    G_B[2] += b_temp
                    G_W[2] += b_temp @ np.transpose(A[k][2])
                for k in range(n, n + batch_size):
                    G_A1[k - n] = np.transpose(W[2]) @ (G_A2[k - n] * A[k][3] * (1 - A[k][3]))
                    b_temp = G_A1[k - n] * A[k][2] * (1 - A[k][2])
                    G_B[1] += b_temp
                    G_W[1] += b_temp @ np.transpose(A[k][1])
                for k in range(n, n + batch_size):
                    G_A0[k - n] = np.transpose(W[1]) @ (G_A1[k - n] * A[k][2] * (1 - A[k][2]))
                    b_temp = G_A0[k - n] * A[k][1] * (1 - A[k][1])
                    G_B[0] += b_temp
                    G_W[0] += b_temp @ np.transpose(A[k][0])
                # update W and B
                for i in range(3):
                    if learning_rate_method:
                        if np.linalg.norm(G_W[i]):
                            W[i] -= learning_rate * G_W[i] / np.linalg.norm(G_W[i])
                        if np.linalg.norm(G_B[i]):
                            B[i] -= learning_rate * G_B[i] / np.linalg.norm(G_B[i])
                    else:
                        W[i] -= learning_rate * G_W[i] / batch_size
                        B[i] -= learning_rate * G_B[i] / batch_size
            else:
                G_W = [np.zeros((150, 102)), np.zeros((60, 150)), np.zeros((4, 60))]
                G_B = [np.zeros((150, 1)), np.zeros((60, 1)), np.zeros((4, 1))]
                for i in range(n, n + batch_size):
                    for j in range(3):
                        A[i][j + 1] = sigmoid(W[j] @ A[i][j] + B[j])
                # back propagation
                G_A2 = np.zeros((batch_size, len(W[2])))
                for i in range(len(W[2])):
                    for k in range(n, n + batch_size):
                        G_A2[k-n][i] = 2 * (A[k][3][i] - train_set[k][1][i])
                        temp = G_A2[k - n][i] * A[k][3][i] * (1 - A[k][3][i])
                        G_B[2][i] += temp
                        for j in range(len(W[2][0])):
                            G_W[2][i][j] += temp * A[k][2][j]

                G_A1 = np.zeros((batch_size, len(W[1])))
                for i in range(len(W[1])):
                    for k in range(n, n + batch_size):
                        for m in range(len(W[2])):
                            G_A1[k-n][i] += G_A2[k-n][m] * A[k][3][m] * (1 - A[k][3][m]) * W[2][m][i]
                        temp = G_A1[k - n][i] * A[k][2][i] * (1 - A[k][2][i])
                        G_B[1][i] += temp
                        for j in range(len(W[1][0])):
                            G_W[1][i][j] += temp * A[k][1][j]

                G_A0 = np.zeros((batch_size, len(W[0])))
                for i in range(len(W[0])):
                    for k in range(n, n + batch_size):
                        for m in range(len(W[1])):
                            G_A0[k-n][i] += G_A1[k-n][m] * A[k][2][m] * (1 - A[k][2][m]) * W[1][m][i]
                        temp = G_A0[k - n][i] * A[k][1][i] * (1 - A[k][1][i])
                        G_B[0][i] += temp
                        for j in range(len(W[0][0])):
                            G_W[0][i][j] += temp * A[k][0][j]
                # update W and B
                for i in range(3):
                    for j in range(len(W[i])):
                        for k in range(len(W[i][0])):
                            W[i][j][k] -= learning_rate * G_W[i][j][k] / batch_size
                        B[i][j] -= learning_rate * G_B[i][j] / batch_size
        # shuffle and feed forward and calculate cost
        random.shuffle(train_set[0:train_number])
        for i in range(train_number):
            A[i][0] = train_set[i][0]
            for j in range(3):
                A[i][j + 1] = sigmoid(W[j] @ A[i][j] + B[j])
        cost = 0
        for i in range(train_number):
            cost += sum(np.power((A[i][3] - train_set[i][1]), 2))
        costs_avg[epoch] += cost / train_number

    end_time = time.time()
    training_time.append(end_time - start_time)
    # train set precision
    true = 0
    for i in range(train_number):
        if np.argmax(train_set[i][1]) == np.argmax(A[i][3]):
            true += 1
    after_training_train_set_precision.append(true / train_number)
    # test set feed forward and precision
    C = []
    for i in range(len(test_set)):
        C.append([test_set[i][0]])
        for j in range(3):
            C[i].append(sigmoid(W[j] @ C[i][j] + B[j]))
    true = 0
    for i in range(len(test_set)):
        if np.argmax(test_set[i][1]) == np.argmax(C[i][3]):
            true += 1
    after_training_test_set_precision.append(true / len(test_set))

print("Before training precision:",
      round(100*sum(before_training_precision)/iteration_number, 2), "%")
print("After training precision(train set):",
      round(100 * sum(after_training_train_set_precision) / iteration_number, 2), "%")
print("After training precision(test set):",
      round(100 * sum(after_training_test_set_precision) / iteration_number, 2), "%")
print("Training time:",
      round(sum(training_time) / iteration_number, 2), "seconds")
plt.plot(np.arange(1, iteration_number + 1), 100 * np.array(after_training_test_set_precision))
plt.xlabel("Iteration")
plt.ylabel("Test_set Precision")
plt.show()
plt.plot(np.arange(1, epochs + 1), costs_avg / iteration_number)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()
