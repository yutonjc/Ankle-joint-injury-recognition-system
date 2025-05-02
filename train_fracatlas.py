import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model import ClassificationModel3D
from datasets_fracatlas import get_datasets
import os
from datetime import datetime

def train_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型
    model = ClassificationModel3D(dropout=0.4, dropout2=0.4)
    model = model.to(device)

    # 获取数据集
    train_set, val_set = get_datasets()
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_set, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=4, 
        shuffle=False, 
        num_workers=2
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 创建保存模型的目录
    save_dir = "model_save"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs_fracatlas_{timestamp}.csv'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    # 训练循环
    num_epochs = 100
    best_val_acc = 0.0

    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels, _) in enumerate(train_loader):
                try:
                    # 调整图像维度以匹配模型输入 (B, C, D, H, W)
                    images = images.unsqueeze(2)
                    images = images.repeat(1, 1, 160, 1, 1)
                    
                    # 移动数据到设备
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # 前向传播
                    outputs = model(images)
                    # 使用第一个输出作为二分类结果
                    binary_outputs = outputs[:, 0]
                    
                    # 计算损失
                    loss = criterion(binary_outputs, labels)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 统计
                    train_loss += loss.item()
                    predictions = (binary_outputs > 0)
                    train_correct += (predictions == labels).sum().item()
                    train_total += labels.size(0)

                    # 打印进度
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Epoch [{epoch+1}/{num_epochs}] "
                              f"Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {loss.item():.4f}")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"WARNING: out of memory error in batch {batch_idx}. Skipping this batch.")
                        continue
                    else:
                        raise e

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels, _ in val_loader:
                    try:
                        # 调整图像维度以匹配模型输入
                        images = images.unsqueeze(2)
                        images = images.repeat(1, 1, 160, 1, 1)
                        
                        # 移动数据到设备
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        # 前向传播
                        outputs = model(images)
                        binary_outputs = outputs[:, 0]
                        
                        # 计算损失
                        loss = criterion(binary_outputs, labels)
                        
                        # 统计
                        val_loss += loss.item()
                        predictions = (binary_outputs > 0)
                        val_correct += (predictions == labels).sum().item()
                        val_total += labels.size(0)
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print(f"WARNING: out of memory error during validation. Skipping this batch.")
                            continue
                        else:
                            raise e

            # 计算平均损失和准确率
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # 保存日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc}\n")

            # 打印训练信息
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} (Total: {train_total})")
            print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} (Total: {val_total})")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                          os.path.join(save_dir, f'fracatlas_model_best_{timestamp}.pth'))
                print(f"Saved new best model with validation accuracy: {val_acc:.4f}")

            # 定期保存模型
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), 
                          os.path.join(save_dir, f'fracatlas_model_epoch_{epoch+1}_{timestamp}.pth'))

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    finally:
        # 保存最后的模型
        torch.save(model.state_dict(), 
                  os.path.join(save_dir, f'fracatlas_model_final_{timestamp}.pth'))
        print("Training finished")

if __name__ == "__main__":
    # 设置 CUDA 设备
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    train_model() 