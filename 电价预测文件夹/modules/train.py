import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import EfficientTransformerForecaster
from config import config

'''def train_model(X_train, y_train, X_val, y_val):
    """训练Transformer模型"""
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # 重塑数据以适应标准化
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    
    # 拟合和转换训练数据
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(X_val.shape)
    
    # 对目标值进行标准化
    y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
    y_val_reshaped = y_val.reshape(-1, y_val.shape[-1])
    
    y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
    y_val_scaled = scaler_y.transform(y_val_reshaped).reshape(y_val.shape)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(config.DEVICE)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(config.DEVICE)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(config.DEVICE)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(config.DEVICE)
    
    # 初始化模型
    input_size = X_train.shape[-1]
    output_size = y_train.shape[-1]
    model = TransformerForecaster(input_size, output_size).to(config.DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练模型
    train_losses = []
    val_losses = []
    
    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(X_train_tensor.transpose(0, 1))
        loss = criterion(outputs.transpose(0, 1), y_train_tensor)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.transpose(0, 1))
            val_loss = criterion(val_outputs.transpose(0, 1), y_val_tensor)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config.EPOCHS}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return model, scaler_X, scaler_y, train_losses, val_losses'''
# 在 train.py 中添加混合精度训练
from torch.cuda.amp import autocast, GradScaler
from model import EfficientTransformerForecaster
def train_model(X_train, y_train, X_val, y_val):
    """训练Transformer模型"""
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # 重塑数据以适应标准化
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    
    # 拟合和转换训练数据
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(X_val.shape)
    
    # 对目标值进行标准化
    y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
    y_val_reshaped = y_val.reshape(-1, y_val.shape[-1])
    
    y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
    y_val_scaled = scaler_y.transform(y_val_reshaped).reshape(y_val.shape)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(config.DEVICE)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(config.DEVICE)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(config.DEVICE)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(config.DEVICE)
    
    # 初始化模型
    input_size = X_train.shape[-1]
    output_size = y_train.shape[-1]
    model = EfficientTransformerForecaster(input_size, output_size).to(config.DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 初始化梯度缩放器
    scaler = GradScaler()
    
    # 训练模型
    train_losses = []
    val_losses = []
    
    # 使用更小的子批量
    sub_batch_size = 4
    
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0
        
        # 使用子批量处理数据
        for i in range(0, len(X_train_tensor), sub_batch_size):
            optimizer.zero_grad()
            
            # 获取子批量
            sub_batch_X = X_train_tensor[i:i+sub_batch_size]
            sub_batch_y = y_train_tensor[i:i+sub_batch_size]
            
            # 使用混合精度
            with autocast():
                # 前向传播
                outputs = model(sub_batch_X.transpose(0, 1))
                loss = criterion(outputs.transpose(0, 1), sub_batch_y)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / (len(X_train_tensor) / sub_batch_size)
        train_losses.append(avg_epoch_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.transpose(0, 1))
            val_loss = criterion(val_outputs.transpose(0, 1), y_val_tensor)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config.EPOCHS}], Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss.item():4f}')
    
    return model, scaler_X, scaler_y, train_losses, val_losses