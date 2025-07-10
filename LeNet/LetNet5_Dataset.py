from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Preprocessing: Resize and normalize to [0, 1]
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Download and load training + test sets
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split training into training + validation (e.g. 50,000 train / 10,000 val)
train_size = int(0.83 * len(mnist_train))  # ~50,000
val_size = len(mnist_train) - train_size   # ~10,000
train_dataset, val_dataset = random_split(mnist_train, [train_size, val_size])
