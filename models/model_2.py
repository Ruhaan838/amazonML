# CRNN

import torch
from torch import nn
from torch.data.utils import Dataset, DataLoader

from tqdm import tqdm



# B:32, C:3 , H:256, W:256
# r, g, b

# 1, 512, 15, 15
class ConvNet(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        # self.b, self.c, self.h, self.w = dataset.shape
        self.c = c

        #conv_block_1
        self.conv1 = nn.Conv2d(self.c, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        #conv_block_2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        #conv_block_3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(2, 2))

        #conv_block_4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4_2 = nn.BatchNorm2d(512)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(2, 2))
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.max_pool_1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.max_pool_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.max_pool_3(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.max_pool_4(x)
        x = self.conv4_3(x)

        return x


# 96*96
# 6 * 28 *28 --> conv --> flatten (6,784) ---> linear() --> 6 * 9216 --> lstm --> 6 * 256
# (a, b. c) --> (a, b*c)
class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out, _ = self.lstm1(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class CRNN(nn.Module):
    def __init__(self, conv_net: ConvNet, lstm: BiLSTM) -> None:
        super().__init__()
        self.conv = conv_net
        self.lstm = lstm

    def forward(self, x):
        x = self.conv(x)  # (b, 512, 15, 15)
        # map to sequence
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # (b, c, h*w)
        x = self.lstm(x)
        return x




# output --> entity_value
def train(model: nn.Module, dataloader: DataLoader, criterion: nn, optimizer: torch.optim, device: str):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_description( Traning_Loss = loss.item() )
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = (correct / total) * 100

    # print(f'loss: {epoch_loss:.4f}, acc: {epoch_acc:.2f}')
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn, device: str):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    eval_loss = running_loss / len(dataloader)
    eval_acc = (correct / total) * 100

    # print(f'loss: {eval_loss:.4f}, acc: {eval_acc:.2f}')
    return eval_loss, eval_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"epoch: {epoch+1}/{num_epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}")



if __name__ == '__main__':
    inputs = torch.randn(1, 3, 256, 256)
    conv = ConvNet(3)

    lstm = BiLSTM(225, 2048, 20)
    model = CRNN(conv, lstm)

    criterion = nn.CrossEntropyLoss(ignore_index=)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # print(model(inputs).size())






