import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# get data from local file
iris = pd.read_csv('iris.data')

# clean up data
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris['is_setosa'] = (iris['class'] == "Iris-setosa").astype(int)
iris['is_versicolor'] = (iris['class'] == "Iris-versicolor").astype(int)
iris['is_virginica'] = (iris['class'] == "Iris-virginica").astype(int)
iris = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'is_setosa', 'is_versicolor', 'is_virginica']]

# randomize rows order
data = iris.sample(frac=1)

# split to training and testing sets, make torch tensors
train = data[:110]
train_x_np = np.array(train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], dtype='float')
train_y_np = np.array(train[['is_setosa', 'is_versicolor', 'is_virginica']], dtype='float')

# create torch tensors for input
train_x = torch.tensor(train_x_np, dtype=torch.float32)
train_y = torch.tensor(train_y_np, dtype=torch.float32)

# test dataset
test = data[111:]
test_x_np = np.array(test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], dtype='float')
test_y_np = np.array(test[['is_setosa', 'is_versicolor', 'is_virginica']], dtype='float')

test_x = torch.tensor(test_x_np, dtype=torch.float32)
test_y = torch.tensor(test_y_np, dtype=torch.float32)

# define model
class NeuralNet(nn.Module):
    def __init__(self, d_in, h, d_out):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(d_in, h)
        self.linear2 = nn.Linear(h, d_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        pred = self.linear2(h_relu)
        return pred

# train model
d_in, h, d_out = 4, 10, 3
model = NeuralNet(d_in, h, d_out)

# loss function and optimizer
loss = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

correct = 0
total = 0

# feed network
for p in range(500):
    for t in range(109):
        y_pred = model(train_x[t])

        l = loss(y_pred, train_y[t])

        np_pred_arr = y_pred.detach().numpy()
        np_ans_arr = np.array(train_y[t])

        pred = np.argmax(np_pred_arr)
        ans = np.argmax(np_ans_arr)

        if pred == ans:
            correct += 1
        total += 1

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    if p % 10 == 0:
        print('epoch ', p, ' training accuracy: ', (correct/total))
print('\n')

# test model
correct_test = 0
total_test = 0

for t in range(37):
    y_pred = model(test_x[t])

    l = loss(y_pred, test_y[t])

    print('prediction: ', y_pred)
    print('actual: ', test_y[t])
    print()

    np_pred_arr = y_pred.detach().numpy()
    np_ans_arr = np.array(test_y[t])

    pred = np.argmax(np_pred_arr)
    ans = np.argmax(np_ans_arr)

    if pred == ans:
        correct_test += 1
    total_test += 1

print('\nTESTING ACCURACY: ', correct_test / total_test)