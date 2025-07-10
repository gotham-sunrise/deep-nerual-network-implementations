import os
import torchvision
from torchvision import transforms
from torch.utils.data import random_split
import pandas as pd
import zipfile

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

mnist_train_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_size = int(0.83 * len(mnist_train_full))
val_size = len(mnist_train_full) - train_size
train_dataset, val_dataset = random_split(mnist_train_full, [train_size, val_size])

def convert_dataset(dataset, max_items=10000):
    data = []
    for i, (img_tensor, label) in enumerate(dataset):
        if i >= max_items:
            break
        flat_img = img_tensor.view(-1).tolist()
        flat_img.append(label)
        data.append(flat_img)
    return data

columns = [f"pixel_{i}" for i in range(32*32)] + ["label"]

train_df = pd.DataFrame(convert_dataset(train_dataset), columns=columns)
val_df = pd.DataFrame(convert_dataset(val_dataset), columns=columns)
test_df = pd.DataFrame(convert_dataset(mnist_test), columns=columns)

os.makedirs("mnist_csv_lite", exist_ok=True)
train_df.to_csv("mnist_csv_lite/mnist_train_10k.csv", index=False)
val_df.to_csv("mnist_csv_lite/mnist_val_10k.csv", index=False)
test_df.to_csv("mnist_csv_lite/mnist_test_10k.csv", index=False)

with zipfile.ZipFile("mnist_csv_lite.zip", "w") as zipf:
    zipf.write("mnist_csv_lite/mnist_train_10k.csv", arcname="mnist_train_10k.csv")
    zipf.write("mnist_csv_lite/mnist_val_10k.csv", arcname="mnist_val_10k.csv")
    zipf.write("mnist_csv_lite/mnist_test_10k.csv", arcname="mnist_test_10k.csv")

print("MNIST CSV ZIP created: mnist_csv_lite.zip")
