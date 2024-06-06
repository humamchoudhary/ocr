import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import tqdm
import pandas as pd
from torch.utils.data import DataLoader, random_split

from sampler import ImbalancedDatasetSampler

# from torchsampler import ImbalancedDatasetSampler

# from resnet import ResNet18

batch_size = 128
num_epochs = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_train = pd.read_csv("./dataset_train.csv")
df_val = pd.read_csv("./dataset_val.csv")

label_counts = df_train["label"].value_counts()
label_encoder = LabelEncoder()

df_train["label"] = label_encoder.fit_transform(df_train["label"])
df_val["label"] = label_encoder.fit_transform(df_val["label"])

# token_to_char = {i: char for i, char in enumerate(label_encoder.classes_)}

num_classes = df_train["label"].nunique()

print("TRAIN")


class HandWrittenCharaterDataset(Dataset):
    def __init__(self, df, transforms=None):

        self.df = df

        self.transforms = transforms

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]
        image_path = row["image"]

        label = row["label"]
        image = Image.open(image_path).convert("L")

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label)

    def __getlabel__(self, idx):
        return self.df.iloc[idx]["label"]


class InvertImageTransform:
    def __call__(self, img):
        return 1 - img


transform = transforms.Compose(
    [
        transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # InvertImageTransform(),  # Apply the custom invert transform
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
# segmented_transform = SegmentedTransform(transform)
train_ds = HandWrittenCharaterDataset(df_train, transform)
val_ds = HandWrittenCharaterDataset(df_val, transform)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=16,
    pin_memory=True,
    # sampler=ImbalancedDatasetSampler(train_ds),
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    num_workers=16,
    pin_memory=True,
    # sampler=ImbalancedDatasetSampler(val_ds),
)


class HandwritingOCR(nn.Module):

    def __init__(self, num_classes):
        super(HandwritingOCR, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = HandwritingOCR(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


def train_fn(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=10,
    patience=5,
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epochs_no_improve = 0
    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            # Training phase
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            for inputs, labels in tqdm.tqdm(train_loader, total=len(train_loader)):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                train_acc += (outputs.max(1)[1] == labels).sum().item()

            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_acc / len(train_loader.dataset)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for inputs, labels in tqdm.tqdm(val_loader, total=len(val_loader)):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_acc += (outputs.max(1)[1] == labels).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_acc / len(val_loader.dataset)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save the model if the validation loss is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model-v5.pth")
                epochs_no_improve = (
                    0  # Reset the counter when we get a better validation loss
                )
            else:
                epochs_no_improve += 1

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Plot the training and validation loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.savefig("loss_curves-v5.png")
            # plt.show()

            # Plot the training and validation accuracy curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_accs, label="Train Accuracy")
            plt.plot(val_accs, label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.legend()
            plt.savefig("accuracy_curves-v5.png")
            # plt.show()

            # Early stopping
            # if epochs_no_improve == patience:
            #     print(f"Early stopping at epoch {epoch + 1}")
            #     break
    except KeyboardInterrupt:
        print("Stoping")

    print(f"Best Val Loss: {best_val_loss:.4f}")
    torch.save(model.state_dict(), "model-v5.pth")


if __name__ == "__main__":
    train_fn(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_epochs=num_epochs,
        patience=5,
    )
