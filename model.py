import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import torch.nn.init as init


def init_weight(layer, init_type, scale):
    # print(f'before {layer=}\n{layer.weight=}')
    match init_type:
        case 'uniform':
            init.uniform_(layer.weight, a=-scale *
                          (3 ** 0.5), b=scale * (3 ** 0.5))
        case 'gaussian':
            init.normal_(layer.weight, mean=0, std=scale)
        case 'constant':
            init.constant_(layer.weight, val=0.5 * scale)

    # print(f'after {layer=}\n{layer.weight=}')


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class Dense_Model(nn.Module):
    def __init__(self, params, output_size):
        super(Dense_Model, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Dropout(p=params['input_dropout_ratio']))
        for indx_layer in range(params['n_layer']):
            temp = nn.Linear(params['n'+str(indx_layer)],
                             params['n'+str(indx_layer+1)])
            init_weight(
                temp, params['init_weight_distribution'], scale=params['init_weight_scale'])

            self.layers.append(temp)
            self.layers.append(get_activation(params['activation']))

        temp = nn.Linear(params['n'+str(params['n_layer'])], output_size)
        init_weight(
            temp, params['init_weight_distribution'], scale=params['init_weight_scale'])
        self.layers.append(temp)
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        # print(f'before {x=}')
        for layer in self.layers:
            x = layer(x)
        # print(f'after {x=}')

        return x


def get_activation(activation_type):
    match activation_type:

        case 'tanh':
            return nn.Tanh()
        case 'relu':
            return nn.ReLU()
        case 'sigmoid':
            return nn.Sigmoid()

def get_accuracy(ind):
    device=torch.device('cuda')
    params = {'init_weight_distribution': 0,
              'init_weight_scale': 0,
              'n_layer': 0,
              'n1': 0,
              'n2': 0,
              'n3': 0,
              'n4': 0,
              'n5': 0,
              'activation': 0,
              'rho': 0,
              'eps': 0,
              'input_dropout_ratio': 0,
              'l1': 0,
              'l2': 0}

    for indx, key in enumerate(params.keys()):
        params[key] = ind[indx]

    df = pd.read_csv('./data/BCWD dataset.csv')

    X = df[df.columns[2:-1]].values
    X = (X-X.min())/(X.max()-X.min())
    X = torch.tensor(X, dtype=torch.float32).to(device)

    Y = df[df.columns[1]].values
    unique_categories = np.unique(Y)
    num_categories = len(unique_categories)
    Y = np.eye(num_categories)[np.searchsorted(unique_categories, Y)]
    Y = torch.tensor(Y, dtype=torch.float16).to(device)

    input_size = X.shape[1]
    output_size = num_categories
    batch_size = 128
    num_epochs = 200
    n_folds = 3

    params['n0'] = input_size

    # percent_train = 0.7

    # x_train, x_valid, y_train, y_valid = train_test_split(
    #     X, Y, train_size=percent_train, random_state=7, shuffle=True, stratify=Y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    acc_per_fold = []
    for _, (train_indices, val_indices) in enumerate(kf.split(X)):
        # print(f'{train_indices=}')
        dataset_train = CustomDataset(X[train_indices], Y[train_indices])
        dataset_valid = CustomDataset(X[val_indices], Y[val_indices])

        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_valid = DataLoader(
            dataset_valid, batch_size=batch_size, shuffle=True)

        model = Dense_Model(params, output_size)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model.parameters(
        ), lr=1.0, rho=params['rho'], eps=params['eps'], weight_decay=params['l2'])
        # Training loop
        # acc_train = 0
        # acc_test = 0
        for epoch in range(num_epochs):

            model.train()
            # total_loss = 0.0
            all_predictions = []
            all_labels = []

            for batch_data, batch_labels in dataloader_train:
                # batch_data.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                l1_reg = torch.tensor(0., requires_grad=True)

                for param in model.parameters():
                    l1_reg = l1_reg + torch.norm(param, 1)

                # Add L1 and L2 regularization to the loss
                loss = loss + params['l1'] * l1_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # total_loss += loss.item()

                _, predictions = torch.max(outputs, 1)
                _, targets = torch.max(batch_labels, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
            # if epoch == num_epochs-1:
            # print(f'{all_labels=}\n{all_predictions=}')

            # if epoch == num_epochs-1:
            #     accuracy = accuracy_score(all_labels, all_predictions)
            #     acc_train = accuracy
            #     print(
            #         f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {acc_train=}")

            if epoch == num_epochs-1:
                all_predictions = []
                all_labels = []
                model.eval()
                with torch.no_grad():
                    for batch_data, batch_labels in dataloader_valid:
                        outputs = model(batch_data)

                        _, predictions = torch.max(outputs, 1)
                        _, targets = torch.max(batch_labels, 1)

                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(targets.cpu().numpy())

                    accuracy = accuracy_score(all_labels, all_predictions)
                    acc_per_fold.append(accuracy)
                    # acc_test=accuracy
                    # print(
                    #     f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {acc_test=}")
    return np.mean(acc_per_fold)
