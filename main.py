import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ContrailsDataset import ContrailsDataset
from DiceLoss import DiceLoss
from EarlyStopper import EarlyStopper
from UNet import UNet

BASE_DATA_PATH = '/kaggle/input/google-research-identify-contrails-reduce-global-warming'

TRAIN_DATA_PATH = BASE_DATA_PATH + "/train"
VALIDATION_DATA_PATH = BASE_DATA_PATH + "/validation"

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(
            mean=[0.5]*4,  # 4 bands
            std=[0.5]*4,
        ),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Normalize(
            mean=[0.5]*4,
            std=[0.5]*4,
        ),
        ToTensorV2()
    ])


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(loader, desc="Training")

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0
    total_dice = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, masks.unsqueeze(1)).item()
            total_dice += dice_score(outputs, masks.unsqueeze(1))

    return val_loss / len(loader), total_dice / len(loader)

def dice_score(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def train_model(max_epochs=40):
    best_dice = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        # Train & Validate
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_dice = validate(model, val_loader, criterion)

        # Update learning rate
        scheduler.step(val_loss)

        if val_dice > best_dice:
            best_dice = val_dice

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Early stopping check
        if early_stopper(val_dice):
            print("Early stopping triggered")
            break

    return history

if __name__ == "__main__":
    train_dataset = ContrailsDataset(
        root_dir=TRAIN_DATA_PATH,
        transform=get_train_transform()
    )

    val_dataset = ContrailsDataset(
        root_dir=VALIDATION_DATA_PATH,
        transform=get_val_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.875e-3)
    criterion = DiceLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=3,
        factor=0.5
    )

    early_stopper = EarlyStopper(
        patience=5,
        min_delta=0.005
    )

    final_model = train_model()