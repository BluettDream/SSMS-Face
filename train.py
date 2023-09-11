# -*- coding=utf-8 -*-
# name: MengHao Tian
# date: 2023/4/18 16:33
import argparse
import os.path
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_PATH = os.path.dirname(__file__)


class MyDataset(Dataset):
    def __init__(self, data, labels):
        super(MyDataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_data(data_path: str = BASE_PATH + '/data.csv') -> (DataLoader, DataLoader, int):
    # Load data
    df = pd.read_csv(data_path)
    data = np.array(df.iloc[:, :-1])  # shape=(num_samples, 128)
    with open(BASE_PATH + '/id_to_label.pkl', 'rb') as f:
        id_to_label = pickle.load(f)
    # 将user_id转换为label
    df['label'] = df['user_id'].map(id_to_label)
    labels = np.array(df['label'])  # shape=(num_samples,)
    cls_num = len(df['label'].unique())
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels.data)

    train_dataset = MyDataset(x_train, y_train)
    test_dataset = MyDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, cls_num


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True), nn.Dropout(dropout_rate))
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size), nn.ReLU(True), nn.Dropout(dropout_rate))
        self.layer3 = nn.Sequential(nn.Linear(2 * hidden_size, 4 * hidden_size), nn.ReLU(True), nn.Dropout(dropout_rate))
        self.output = nn.Sequential(nn.Linear(4 * hidden_size, output_size), nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.output(x)
        return out


def train(model: MLP, train_loader: DataLoader, save_path: str = BASE_PATH + '/model.pth', epochs: int = 100):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=2e-3, momentum=0.9, weight_decay=1e-5)

    # Train the model
    num_epochs = epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Print statistics
        print('Epoch [{}/{}], Loss: {:.6f}'
              .format(epoch + 1, num_epochs, running_loss / len(train_loader)))

    torch.save(model, save_path)


def test(test_loader: DataLoader, model_path: str = BASE_PATH + '/model.pth') -> float:
    model = torch.load(model_path)
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print accuracy
        print('Accuracy: {:.6f}%'.format(100 * correct / total))
    return 100 * correct / total


if __name__ == '__main__':
    train_loader, test_loader, class_num = load_data()
    # Create model
    model = MLP(input_size=128, hidden_size=128, output_size=class_num, dropout_rate=0.35)
    train(model, train_loader, epochs=500)
    test(test_loader)
