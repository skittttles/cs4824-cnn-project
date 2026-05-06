from medmnist import PneumoniaMNIST, Evaluator

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix

torch.manual_seed(42941)
torch.cuda.manual_seed_all(42941)
SIZE = 28
NUM_EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])


trainingData = PneumoniaMNIST(split = "train", download = True, size = SIZE, transform = data_transform)
validationData = PneumoniaMNIST(split = "val", download = True, size = SIZE, transform = data_transform)
testData = PneumoniaMNIST(split = "test", download = True, size = SIZE, transform = data_transform)

trainLoader = data.DataLoader(trainingData, batch_size = 128, shuffle = True)
trainLoaderEval = data.DataLoader(trainingData, batch_size = 128, shuffle = False)
valLoader = data.DataLoader(validationData, batch_size = 128, shuffle = False)
testLoader = data.DataLoader(testData, batch_size = 128, shuffle = True)

task = "binary-class"
lr = 0.001
data_flag = "pneumoniamnist" 

# CNN
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(self._get_flat_size(in_channels), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def _get_flat_size(self, in_channels):
        dummy = torch.zeros(1, in_channels, SIZE, SIZE)
        x = self.layer1(dummy)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x.view(1, -1).shape[1]        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = Net(in_channels=1, num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


# train
for epoch in range(NUM_EPOCHS):

    model.train()
    for inputs, targets in tqdm(trainLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)

        targets = targets.squeeze().long()
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()


# evaluation
def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    y_pred = torch.tensor([])
    
    data_loader = trainLoaderEval if split == 'train' else testLoader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)

            predicted = outputs.argmax(dim=1)
            y_pred = torch.cat((y_pred, predicted.float().cpu()), 0)

            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets.cpu()), 0)
            y_score = torch.cat((y_score, outputs.cpu()), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        y_pred = y_pred.numpy()

        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_score[:, 1])

        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()

        print(f'\n{split} results:')
        print(f'Accuracy: {accuracy:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'F1 Score: {f1:.3f}')
        print(f'AUC: {roc_auc:.3f}\n')

        print(f'Confusion Matrix:\n{cm}')
        print(f'True Positive: {int(TP)}')
        print(f'True Negative: {int(TN)}')
        print(f'False Positive: {int(FP)}') 
        print(f'False Negative: {int(FN)}')

print('==> Evaluating ...')
test('train')
test('test')

#Visualize some datapoints

examples = enumerate(testLoader)
batch_idx, (example_data, example_targets) = next(examples)

for i in range(10):
    plt.subplot(2,5, i+1)
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    output = model(example_data.to(device))
    plt.title("Label:{}\nPred:{}".format(example_targets.data[i].item(), output.cpu().data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])

plt.savefig("predictions.png")
plt.show()