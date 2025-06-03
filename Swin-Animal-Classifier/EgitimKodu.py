#Colab İçindir Colabde Deneyiniz.
from google.colab import drive
drive.mount('/content/drive')

import os
dataset_base_path = '/content/drive/MyDrive/Yazlab3/MultiZoo_Dataset'
train_path = os.path.join(dataset_base_path, 'train')
test_path = os.path.join(dataset_base_path, 'test')
print(f"Eğitim veri seti yolu: {train_path}")
if os.path.exists(train_path):
    print("Eğitim setindeki sınıflar:", os.listdir(train_path))
else:
    print(f"Eğitim yolu bulunamadı: {train_path}")

import torchvision.transforms as transforms

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

print(f"Dönüşümler tanımlandı Input size: {IMG_SIZE}x{IMG_SIZE}).")

import torch
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
import os

train_path = '/content/drive/MyDrive/Yazlab3/MultiZoo_Dataset/train'
full_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Toplam görsel sayısı: {len(full_dataset)}")

val_split = 0.20
val_size = int(val_split * len(full_dataset))
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

print(f"Eğitim seti : {len(train_dataset)}")
print(f"Doğrulama seti : {len(val_dataset)}")

BATCH_SIZE = 32
NUM_WORKERS = 2
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          pin_memory=True)

val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=True)

print(f"DataLoaderlar hazırlandı.")

try:
    images, labels = next(iter(train_loader))
    print(f"Bir batch yüklendi:")
    print(f"  Görüntü batch şekli: {images.shape}")
    print(f"  Etiket batch şekli: {labels.shape}")
    print(f"  Görüntü veri tipi: {images.dtype}")
    print(f"  Min piksel değeri: {images.min():.2f}, Max piksel değeri: {images.max():.2f}")
    print(f"  Örnek etiketler: {labels[:5]}")
except Exception as e:
    print(f"DataLoaderdan batch alınırken hata oluştu: {e}")

!pip install timm -q

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılacak cihaz: {device}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import timm
import os

MODEL_NAME = "swin_base_patch4_window7_224"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.05
EPOCHS = 15
BATCH_SIZE = 32
IMG_SIZE = 224
num_classes = 90
print(f"---- Model ve Eğitim Ayarları ----")
print(f"Model Adı: {MODEL_NAME}")
print(f"Sınıf Sayısı: {num_classes}")
print(f"Cihaz: {device}")
print(f"Epoch Sayısı: {EPOCHS}")
print(f"Batch Boyutu: {BATCH_SIZE}")
print(f"Öğrenme Oranı: {LEARNING_RATE}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"AMP: Aktif")
print(f"-----------------------------------")

print("Model yükleniyor...")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
model.to(device)
print(f"Model '{MODEL_NAME}' yüklendi, sınıf sayısı {num_classes} sınıfa göre ayarlandı ve '{device}' cihazına taşındı.")

criterion = nn.CrossEntropyLoss()
print("Kayıp Fonksiyonu: CrossEntropyLoss")
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
print(f"Optimizasyon Algoritması: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 100)
print(f"Öğrenme Oranı Zamanlayıcısı: CosineAnnealingLR (T_max={EPOCHS})")

scaler = GradScaler(enabled=torch.cuda.is_available())
print(f"Gradyan Ölçekleyici AMP için: {'Aktif' if torch.cuda.is_available() else 'Devre Dışı'}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import timm
from tqdm.notebook import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import copy
import os

MODEL_NAME = "swin_base_patch4_window7_224"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.05
EPOCHS = 50
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 90

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

GRAD_CLIP_MAX_NORM = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"---- Model ve Eğitim Ayarları ----")
print(f"Model Adı: {MODEL_NAME}")
print(f"Sınıf Sayısı: {NUM_CLASSES}")
print(f"Cihaz: {device}")
print(f"Max Epoch Sayısı: {EPOCHS}")
print(f"Batch Boyutu: {BATCH_SIZE}")
print(f"Öğrenme Oranı: {LEARNING_RATE}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"AMP: Aktif")
print(f"Gradyan Kırpma: {GRAD_CLIP_MAX_NORM}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"--------------------------------------------")

print("Model yükleniyor...")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model.to(device)
print(f"Model '{MODEL_NAME}' yüklendi ve '{device}' cihazına taşındı.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 100)
scaler = GradScaler(enabled=torch.cuda.is_available())

print("Kayıp, Optimizasyon, Zamanlayıcı ve Ölçekleyici ayarlandı.")

def train_one_epoch(model, criterion, optimizer, data_loader, device, scaler, scheduler, clip_max_norm):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    pbar = tqdm(data_loader, desc="Eğitim (Epoch)", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        pbar.set_postfix(loss=loss.item())
    if scheduler and isinstance(scheduler, CosineAnnealingLR):
         scheduler.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct_predictions / total_predictions
    return epoch_loss, epoch_acc

def evaluate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(data_loader, desc="Doğrulama", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(data_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss, accuracy, precision, recall, f1

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
}

early_stopping_counter = 0
best_val_loss = float('inf')
best_epoch = -1

best_checkpoint_path = 'best_checkpoint.pth'
latest_checkpoint_path = 'latest_checkpoint.pth'

print("\n--- Eğitim Başlatılıyor ---")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch+1}/{EPOCHS} (LR: {current_lr:.6f})")

    train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device, scaler, scheduler if isinstance(scheduler, CosineAnnealingLR) else None, GRAD_CLIP_MAX_NORM)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, criterion, val_loader, device)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)
    history['val_f1'].append(val_f1)

    epoch_time = time.time() - epoch_start_time

    print(f"Epoch {epoch+1} Tamamlandı [{epoch_time:.0f}s] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
          f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'history': history
    }
    torch.save(checkpoint, latest_checkpoint_path)

    if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
        print(f"  ** Doğrulama kaybı iyileşti ({best_val_loss:.4f} -> {val_loss:.4f}). En iyi model kaydediliyor... **")
        best_val_loss = val_loss
        best_epoch = epoch + 1
        early_stopping_counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(checkpoint, best_checkpoint_path)
    else:
        early_stopping_counter += 1
        print(f"  Doğrulama kaybı iyileşmedi. Patience: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n--- Early Stopping Tetiklendi (Epoch {epoch+1}) ---")
            print(f"Doğrulama kaybı {EARLY_STOPPING_PATIENCE} epoch boyunca iyileşmedi.")
            break

total_training_time = time.time() - start_time
print(f"\n--- Eğitim Tamamlandı ---")
print(f"Toplam Eğitim Süresi: {total_training_time // 60:.0f}dk {total_training_time % 60:.0f}s")
if best_epoch != -1:
    print(f"En İyi Epoch: {best_epoch}")
    print(f"En İyi Doğrulama Kaybı : {best_val_loss:.4f}")
    print("En iyi model ağırlıkları yükleniyor...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Hiçbir iyileşme kaydedilmedi")

print("Model eğitimi tamamlandı.")

print("Yüklenen en iyi modelin doğrulama seti üzerinde performansı değerlendiriliyor...")
final_val_loss, final_val_acc, final_val_precision, final_val_recall, final_val_f1 = evaluate(model, criterion, val_loader, device)

print("\n--- En İyi Modelin Doğrulama Performansı ---")
print(f"Doğrulama Kaybı : {final_val_loss:.4f}")
print(f"Doğrulama Doğruluğu : {final_val_acc:.4f}")
print(f"Doğrulama Precision : {final_val_precision:.4f}")
print(f"Doğrulama Recall : {final_val_recall:.4f}")
print(f"Doğrulama F1-Skoru : {final_val_f1:.4f}")

import matplotlib.pyplot as plt
epochs_range = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Eğitim Kaybı (Train Loss)')
plt.plot(epochs_range, history['val_loss'], label='Doğrulama Kaybı (Val Loss)')
plt.scatter(best_epoch, best_val_loss, marker='*', color='red', s=200, label=f'En İyi Val Loss (Epoch {best_epoch})')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], label='Eğitim Doğruluğu (Train Acc)')
plt.plot(epochs_range, history['val_acc'], label='Doğrulama Doğruluğu (Val Acc)')
best_epoch_index = best_epoch - 1
if 0 <= best_epoch_index < len(history['val_acc']):
    plt.scatter(best_epoch, history['val_acc'][best_epoch_index], marker='*', color='red', s=200, label=f'En İyi Val Loss Anındaki Acc (Epoch {best_epoch})')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk (Accuracy)')
plt.legend(loc='lower right')
plt.grid(True)

plt.suptitle('Öğrenme Eğrileri', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()