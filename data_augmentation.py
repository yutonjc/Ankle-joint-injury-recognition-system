import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image
import random

class DataAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),  # 随机旋转±10度
            transforms.ToTensor()  # 移除归一化
        ])
    
    def adjust_contrast(self, image):
        """增强对比度"""
        # 将图像转换为numpy数组
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 调整对比度
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        
        # 转回RGB
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image
    
    def __call__(self, image):
        """
        对图像进行数据增强
        Args:
            image: 输入图像 (numpy array 或 torch.Tensor)
        Returns:
            增强后的图像 (torch.Tensor)
        """
        # 增强对比度
        image = self.adjust_contrast(image)
        
        # 应用其他变换（旋转）
        image = self.transform(image)
        
        return image

def test_augmentation():
    """测试数据增强效果"""
    # 创建一个示例图像
    image = np.random.rand(224, 224, 3) * 255
    image = image.astype(np.uint8)
    
    # 创建数据增强实例
    aug = DataAugmentation()
    
    # 应用数据增强
    augmented_image = aug(image)
    
    # 显示原始图像和增强后的图像
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image.permute(1, 2, 0).numpy())
    plt.title('Augmented Image')
    plt.axis('off')
    
    plt.savefig('data_augmentation_example.png')
    plt.close()

if __name__ == "__main__":
    test_augmentation() 