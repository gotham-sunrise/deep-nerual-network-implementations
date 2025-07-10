from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from LeNet5 import *

# Data loader
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet expects 32x32
    transforms.ToTensor()
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# Instantiate model
model = LeNet5()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
