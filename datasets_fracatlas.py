import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from data_augmentation import DataAugmentation  # 导入数据增强类

# 允许处理截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FracAtlasDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, is_train=True):
        """
        初始化FracAtlas数据集
        Args:
            root_dir (str): 数据集根目录
            csv_file (str): 标注文件路径
            transform (callable, optional): 数据转换操作
            is_train (bool): 是否为训练集
        """
        self.root_dir = root_dir
        self.is_train = is_train
        
        # 创建数据增强实例
        self.data_aug = DataAugmentation() if is_train else None
        
        # 设置基本的数据转换
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
            transforms.ToTensor()  # 移除归一化
        ])
        
        # 读取CSV文件
        self.df = pd.read_csv(csv_file)
        
        # 将image_id列重命名为image_name
        self.df = self.df.rename(columns={'image_id': 'image_name'})
        
        # 筛选leg标签为1的数据
        self.df = self.df[self.df['leg'] == 1]
        
        # 设置图像目录
        self.images_dir = os.path.join(root_dir, 'images')
        
        # 获取所有图像文件路径和标签
        self.image_paths = []
        self.fractured_labels = []
        self.view_labels = []
        
        # 处理所有图像
        for _, row in self.df.iterrows():
            img_name = row['image_name']
            img_path = None
            
            # 检查图像是否存在于Fractured或Non_fractured目录
            fractured_path = os.path.join(self.images_dir, 'Fractured', img_name)
            non_fractured_path = os.path.join(self.images_dir, 'Non_fractured', img_name)
            
            if os.path.exists(fractured_path):
                # 验证图像是否可以正确打开
                try:
                    with Image.open(fractured_path) as img:
                        img.verify()
                    img_path = fractured_path
                except Exception as e:
                    print(f"Warning: Skipping corrupted image {fractured_path}: {str(e)}")
                    continue
            elif os.path.exists(non_fractured_path):
                # 验证图像是否可以正确打开
                try:
                    with Image.open(non_fractured_path) as img:
                        img.verify()
                    img_path = non_fractured_path
                except Exception as e:
                    print(f"Warning: Skipping corrupted image {non_fractured_path}: {str(e)}")
                    continue
                
            if img_path is not None:
                self.image_paths.append(img_path)
                self.fractured_labels.append(row['fractured'])
                
                # 创建视角标签（one-hot编码）
                view_label = [row['frontal'], row['lateral'], row['oblique']]
                self.view_labels.append(view_label)

    def __getitem__(self, idx):
        """
        获取数据集中的某一条数据
        Args:
            idx (int): 索引
        Returns:
            tuple: (image, fractured_label, view_label) 图像、骨折标签和视角标签
        """
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 如果图像加载失败，返回一个黑色图像
            image = Image.new('RGB', (224, 224), 'black')
        
        # 应用数据增强（仅在训练集上）
        if self.is_train and self.data_aug:
            try:
                image = self.data_aug(image)
            except Exception as e:
                print(f"Error in data augmentation for image {img_path}: {str(e)}")
                # 如果数据增强失败，应用基本转换
                image = self.transform(image)
        else:
            # 验证集只应用基本转换
            image = self.transform(image)
        
        fractured_label = torch.tensor(self.fractured_labels[idx], dtype=torch.float32)
        view_label = torch.tensor(self.view_labels[idx], dtype=torch.float32)
        
        return image, fractured_label, view_label

    def __len__(self):
        """
        返回数据集大小
        Returns:
            int: 数据集中的样本数量
        """
        return len(self.image_paths)

def get_datasets(root_dir="./FRACATLAS", csv_file="./FRACATLAS/dataset.csv"):
    """
    获取训练集和验证集
    Args:
        root_dir (str): 数据集根目录
        csv_file (str): 标注文件路径
    Returns:
        tuple: (train_set, val_set) 训练集和验证集
    """
    # 创建完整的数据集
    try:
        full_dataset = FracAtlasDataset(root_dir, csv_file)
        print(f"Successfully loaded {len(full_dataset)} images")
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise
    
    # 计算训练集和验证集的大小
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)  # 80%用于训练
    val_size = total_size - train_size
    
    # 随机划分训练集和验证集
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    return train_set, val_set

if __name__ == "__main__":
    # 测试代码
    DATA_PATH = './FRACATLAS'
    CSV_FILE = './FRACATLAS/dataset.csv'
    try:
        train_set, val_set = get_datasets(DATA_PATH, CSV_FILE)
        
        print(f"训练集大小: {len(train_set)}")
        print(f"验证集大小: {len(val_set)}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
        
        # 测试加载一个批次的数据
        for images, fractured_labels, view_labels in train_loader:
            print(f"Batch shape: {images.shape}")  # 应该显示 [B, 1, H, W]
            print(f"Fractured labels: {fractured_labels}")
            print(f"View labels: {view_labels}")
            break
    except Exception as e:
        print(f"Error in main: {str(e)}") 