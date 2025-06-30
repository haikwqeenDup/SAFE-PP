import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 用于显示进度条
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, TensorDataset

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn as nn


def train_model(model, train_loader, device, class_weights=None, epochs=10, lr=0.001):
    model.train()
    train_losses = []
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scaler = GradScaler()

    for epoch in range(epochs):
        epoch_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        print(f"Current learning rate: {current_lr:.6f}")
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.long().to(device)
            targets = targets.reshape(-1)

            optimizer.zero_grad()

            total_loss = 0
            with autocast():  # Move autocast here to cover the forward pass
                for k in [0, 1, 2, 3]:
                    rotated_inputs = torch.rot90(inputs, k=k, dims=(2, 3))
                    outputs = model(rotated_inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss

                # outputs = model(inputs)
                # loss = criterion(outputs, targets)
                # total_loss = loss

            scaled_loss = scaler.scale(total_loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        scheduler.step()

    return train_losses


def train_model_contrast(
    model,
    train_loader,
    device,
    class_weights=None,
    epochs=10,
    lr=0.001,
    contrast_batch_size=256,
):
    model.train()
    train_losses = []
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scaler = GradScaler()
    tri_criterion = nn.TripletMarginLoss(margin=1)
    contrast_tensor = torch.load("yuexi_tensor_contrast.pt")
    anchor = contrast_tensor[:, 0, :68]
    positive = contrast_tensor[:, 1, :68]
    negative = contrast_tensor[:, 2, :68]
    total_samples = anchor.shape[0]
    for epoch in range(epochs):
        epoch_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        print(f"Current learning rate: {current_lr:.6f}")
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.long().to(device)
            targets = targets.reshape(-1)
            optimizer.zero_grad()
            total_loss = 0
            with autocast():  # Move autocast here to cover the forward pass
                # 四向旋转增强QRAS策略
                for k in [0, 1, 2, 3]:
                    rotated_inputs = torch.rot90(inputs, k=k, dims=(2, 3))
                    outputs = model(rotated_inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss
                # CL 对比学习
                indices = torch.randperm(total_samples)[:contrast_batch_size]
                # 根据索引采样 anchor, positive 和 negative
                sampled_anchor = anchor[indices].to(device)
                sampled_positive = positive[indices].to(device)
                sampled_negative = negative[indices].to(device)
                anchor_output = model.contrast_f(sampled_anchor)
                positive_output = model.contrast_f(sampled_positive)
                negative_output = model.contrast_f(sampled_negative)
                loss_contrast = tri_criterion(
                    anchor_output, positive_output, negative_output
                )
                total_loss += loss_contrast

            scaled_loss = scaler.scale(total_loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        scheduler.step()

    return train_losses


def creat_loader(batch_size=256, name=""):
    X_train = torch.load(name + "tensor_train_x.pt").float()
    y_train = torch.load(name + "tensor_train_y.pt").float().unsqueeze(1)
    print("train data all ready!")
    X_test = torch.load(name + "tensor_test_x.pt").float()
    y_test = torch.load(name + "tensor_test_y.pt").float().unsqueeze(1)
    print("test data all ready!")
    X_val = torch.load(name + "tensor_val_x.pt").float()
    y_val = torch.load(name + "tensor_val_y.pt").float().unsqueeze(1)
    print("val data all ready!")
    unique_values, counts = torch.unique(y_train, return_counts=True)
    print("不同值:", unique_values)
    print("每个值的个数:", counts)

    # 计算总样本数
    total_samples = counts.sum()

    # 计算类别权重
    class_weights = total_samples / (counts + 2)  # 避免除零
    class_weights = class_weights / class_weights.sum()

    print(f"权重设置：{class_weights}")
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader, class_weights
