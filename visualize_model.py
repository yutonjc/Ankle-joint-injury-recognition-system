import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Arrow

def draw_module(ax, x, y, width, height, layers, colors):
    """绘制一个模块，包含多个层"""
    # 绘制模块外框
    rect = Rectangle((x, y), width, height, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # 计算每个子层的高度
    sub_height = height / len(layers)
    
    # 绘制每个子层
    for i, (label, color) in enumerate(zip(layers, colors)):
        sub_y = y + i * sub_height
        sub_rect = Rectangle((x, sub_y), width, sub_height, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(sub_rect)
        ax.text(x + width/2, sub_y + sub_height/2, label, ha='center', va='center', fontsize=8)
    
    return y - height - 20  # 返回下一个模块的y坐标

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    """绘制连接箭头"""
    ax.arrow(x1, y1, x2-x1, y2-y1, head_width=5, head_length=10, fc=color, ec=color)

def visualize_model():
    """可视化ClassificationModel3D模型结构"""
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1200)
    ax.axis('off')
    
    # 定义模块的宽度和高度
    module_width = 200
    module_height = 120
    x = 100  # 起始x坐标
    
    # 绘制输入层
    y = draw_module(ax, x, 1000, module_width, module_height, 
                   ["Input Layer", "3D Data"], 
                   ["lightblue"])
    
    # 绘制卷积模块
    conv_modules = [
        (["Conv3d", "1→8 channels"], ["lightgreen"]),
        (["Conv3d", "8→16 channels"], ["lightgreen"]),
        (["Conv3d", "16→32 channels"], ["lightgreen"]),
        (["Conv3d", "32→64 channels"], ["lightgreen"])
    ]
    
    for layers, colors in conv_modules:
        y = draw_module(ax, x, y, module_width, module_height, 
                       layers + ["BatchNorm", "ReLU", "MaxPool3d"], 
                       colors + ["lightyellow", "lightyellow", "lightcoral"])
    
    # 绘制Flatten层
    y = draw_module(ax, x, y, module_width, module_height, 
                   ["Flatten Layer"], 
                   ["lightblue"])
    
    # 绘制全连接模块
    fc_modules = [
        (["Linear", "64→256"], ["lightpink"]),
        (["Dropout", "ReLU"], ["lightyellow", "lightyellow"]),
        (["Linear", "256→128"], ["lightpink"]),
        (["Dropout", "ReLU"], ["lightyellow", "lightyellow"]),
        (["Linear", "128→2"], ["lightpink"])
    ]
    
    for layers, colors in fc_modules:
        y = draw_module(ax, x, y, module_width, module_height, 
                       layers, colors)
    
    # 绘制输出层
    y = draw_module(ax, x, y, module_width, module_height, 
                   ["Output Layer", "2 classes"], 
                   ["lightgreen"])
    
    # 添加连接箭头，使用不同颜色表示不同类型的连接
    y_positions = np.linspace(1000, y + module_height, 8)
    arrow_colors = [
        'lightblue',  # 输入到第一个卷积层
        'lightgreen', # 卷积层之间
        'lightgreen',
        'lightgreen',
        'lightblue',  # 卷积到Flatten
        'lightpink',  # 全连接层之间
        'lightpink',
        'lightpink',
        'lightgreen'  # 到输出层
    ]
    
    for i in range(len(y_positions)-1):
        draw_arrow(ax, x + module_width/2, y_positions[i], 
                  x + module_width/2, y_positions[i+1],
                  color=arrow_colors[i])
    
    # 添加图例
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Input/Flatten'),
        Rectangle((0, 0), 1, 1, facecolor='lightgreen', label='Conv/Output'),
        Rectangle((0, 0), 1, 1, facecolor='lightyellow', label='BatchNorm/ReLU'),
        Rectangle((0, 0), 1, 1, facecolor='lightcoral', label='MaxPool'),
        Rectangle((0, 0), 1, 1, facecolor='lightpink', label='Linear')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    # 添加标题
    plt.title("ClassificationModel3D Architecture", pad=20, fontsize=16)
    
    # 保存图像
    plt.savefig('model_architecture_vertical.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_model() 