import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import requests
import zipfile
from PIL import Image
from tqdm import tqdm

# Константы
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
IMG_SIZE = 64
NUM_CLASSES = 10  # EuroSAT: 10 классов
DATA_DIR = "eurosat"

# 1. ЗАГРУЗКА ДАННЫХ
def download_and_extract():
    if not os.path.exists(DATA_DIR):
        print("Загрузка EuroSAT...")
        url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
        r = requests.get(url)
        with open("eurosat.zip", "wb") as f:
            f.write(r.content)
        
        with zipfile.ZipFile("eurosat.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove("eurosat.zip")
        print("Готово!")

download_and_extract()

# 2. ПСЕВДО-МАСКИ (для демо создаем синтетические маски из классов)
class EuroSATWithMasks(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.classes = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Создаем псевдо-маску: для демонстрации используем класс как регион
        # В реальном проекте здесь должны быть настоящие маски
        mask = torch.ones((IMG_SIZE, IMG_SIZE), dtype=torch.long) * label
        return img, mask

# Трансформации
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Датасет и загрузчики
full_dataset = EuroSATWithMasks(root=DATA_DIR, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. МОДЕЛЬ U-Net (упрощенная)
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=10):
        super(UNet, self).__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.down1 = double_conv(n_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv1 = double_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv2 = double_conv(128, 64)
        
        self.out = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        
        # Decoder
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv2(x)
        
        return self.out(x)

model = UNet().to(DEVICE)

# 4. МЕТРИКИ
def iou_score(pred, target, num_classes=10):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union > 0:
            ious.append(intersection / union)
    return torch.tensor(ious).mean()

def dice_coef(pred, target, num_classes=10, smooth=1e-6):
    dice = 0
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        dice += (2. * intersection + smooth) / (pred_inds.sum() + target_inds.sum() + smooth)
    return dice / num_classes

# 5. ОБУЧЕНИЕ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_ious = [], []

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_iou += iou_score(preds, masks).item()
    
    train_loss /= len(train_loader)
    val_iou /= len(val_loader)
    
    train_losses.append(train_loss)
    val_ious.append(val_iou)
    
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, IoU={val_iou:.4f}")

# 6. ВИЗУАЛИЗАЦИЯ
def visualize_predictions():
    model.eval()
    images, masks = next(iter(val_loader))
    images, masks = images.to(DEVICE), masks.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        # Оригинал
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title("Оригинал")
        axes[i, 0].axis("off")
        
        # Маска (псевдо)
        axes[i, 1].imshow(masks[i].cpu(), cmap="tab10", vmin=0, vmax=9)
        axes[i, 1].set_title("Маска (истина)")
        axes[i, 1].axis("off")
        
        # Предсказание
        axes[i, 2].imshow(preds[i].cpu(), cmap="tab10", vmin=0, vmax=9)
        axes[i, 2].set_title(f"Предсказание (IoU: {iou_score(preds[i], masks[i]):.2f})")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig("segmentation_results.png")
    plt.show()

visualize_predictions()

# Итоговые метрики
print(f"Финальный IoU: {val_ious[-1]:.4f}")
print(f"Dice на последнем батче: {dice_coef(preds, masks):.4f}")
print("Результаты сохранены в segmentation_results.png")
