# %% [markdown]
# # TP 1 - Classification d'Images FashionMNIST par CNN
# Framework : PyTorch | Dataset : FashionMNIST | 5 epochs

# %%
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# --- Transformations ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# --- Chargement des données ---
train_data = datasets.FashionMNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False,
                                  download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
test_loader = DataLoader(test_data, batch_size=20, shuffle=False)

# %%
# --- Architecture FashionCNN ---
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64*6*6, 600)
        self.drop = nn.Dropout(0.25) # Dropout (pas Dropout2d)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop(self.fc1(out))
        out = self.fc2(out)
        return self.fc3(out)

# %%
# --- Entraînement ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = FashionCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Évaluation sur test
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

    print(f'Epoch {epoch+1}/5 | Test Accuracy: {100*correct/total:.2f}%')