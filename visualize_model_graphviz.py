import os
from graphviz import Digraph
import torch
import torch.nn as nn


def visualize_model_graphviz():
    """使用Graphviz可视化ClassificationModel3D模型结构"""
    # 设置Graphviz的路径
    graphviz_paths = [
        'C:/Program Files/Graphviz/bin',
        'D:/Program Files/Graphviz/bin',
        'C:/Program Files (x86)/Graphviz/bin',
        'D:/Program Files (x86)/Graphviz/bin'
    ]

    # 检查Graphviz是否在PATH中
    for path in graphviz_paths:
        if os.path.exists(path):
            os.environ["PATH"] += os.pathsep + path
            print(f"找到Graphviz路径: {path}")
            break
    else:
        print("未找到Graphviz安装路径，请检查安装位置")
        return

    # 创建有向图
    dot = Digraph(comment='ClassificationModel3D Architecture')
    dot.attr(rankdir='TB')  # 从上到下的布局
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    dot.attr('edge', dir='forward')

    # 定义节点样式
    conv_style = {'fillcolor': 'lightgreen', 'shape': 'box'}
    norm_style = {'fillcolor': 'lightyellow', 'shape': 'box'}
    pool_style = {'fillcolor': 'lightcoral', 'shape': 'box'}
    fc_style = {'fillcolor': 'lightpink', 'shape': 'box'}
    input_style = {'fillcolor': 'lightblue', 'shape': 'box'}
    output_style = {'fillcolor': 'lightgreen', 'shape': 'box'}

    # 创建输入层子图
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Layer')
        c.attr(style='rounded')
        c.node('input', 'Input\n3D Data', **input_style)

    # 创建卷积模块子图
    conv_modules = [
        ('conv1', 'Conv3d\n1→8', conv_style),
        ('bn1', 'BatchNorm\nReLU', norm_style),
        ('pool1', 'MaxPool3d', pool_style),
        ('conv2', 'Conv3d\n8→16', conv_style),
        ('bn2', 'BatchNorm\nReLU', norm_style),
        ('pool2', 'MaxPool3d', pool_style),
        ('conv3', 'Conv3d\n16→32', conv_style),
        ('bn3', 'BatchNorm\nReLU', norm_style),
        ('pool3', 'MaxPool3d', pool_style),
        ('conv4', 'Conv3d\n32→64', conv_style),
        ('bn4', 'BatchNorm\nReLU', norm_style),
        ('pool4', 'MaxPool3d', pool_style)
    ]

    # 将卷积模块分成四个子图，每三个层一组
    for i in range(4):
        with dot.subgraph(name=f'cluster_{i + 1}') as c:
            c.attr(label=f'Conv Block {i + 1}')
            c.attr(style='rounded')
            start_idx = i * 3
            end_idx = start_idx + 3
            for j in range(start_idx, end_idx):
                node_id, label, style = conv_modules[j]
                c.node(node_id, label, **style)

    # 创建全连接模块子图
    fc_layers = [
        ('flatten', 'Flatten', {'fillcolor': 'lightblue', 'shape': 'box'}),
        ('fc1', 'Linear\n64→256', fc_style),
        ('drop1', 'Dropout\nReLU', norm_style),
        ('fc2', 'Linear\n256→128', fc_style),
        ('drop2', 'Dropout\nReLU', norm_style),
        ('fc3', 'Linear\n128→2', fc_style)
    ]

    # 将全连接层分成两个子图
    with dot.subgraph(name='cluster_5') as c:
        c.attr(label='FC Block 1')
        c.attr(style='rounded')
        for i in range(3):
            node_id, label, style = fc_layers[i]
            c.node(node_id, label, **style)

    with dot.subgraph(name='cluster_6') as c:
        c.attr(label='FC Block 2')
        c.attr(style='rounded')
        for i in range(3, 6):
            node_id, label, style = fc_layers[i]
            c.node(node_id, label, **style)

    # 创建输出层子图
    with dot.subgraph(name='cluster_7') as c:
        c.attr(label='Output Layer')
        c.attr(style='rounded')
        c.node('output', 'Output\n2 classes', **output_style)

    # 连接输入到第一个卷积层
    dot.edge('input', 'conv1')

    # 连接卷积模块
    for i in range(len(conv_modules) - 1):
        dot.edge(conv_modules[i][0], conv_modules[i + 1][0])

    # 连接卷积模块到Flatten
    dot.edge(conv_modules[-1][0], 'flatten')

    # 连接全连接模块
    for i in range(len(fc_layers) - 1):
        dot.edge(fc_layers[i][0], fc_layers[i + 1][0])

    # 连接最后一个全连接层到输出
    dot.edge(fc_layers[-1][0], 'output')

    # 设置图属性
    dot.attr(size='8,10')  # 调整尺寸为竖向
    dot.attr(dpi='300')
    dot.attr(ranksep='0.3')  # 减小层级间距
    dot.attr(nodesep='0.2')  # 减小节点间距

    try:
        # 保存图像
        dot.render('model_architecture_graphviz', format='png', cleanup=True)
        print("图像已成功生成：model_architecture_graphviz.png")
    except Exception as e:
        print(f"生成图像时出错：{str(e)}")
        print("\n故障排除步骤：")
        print("1. 确保已安装Graphviz软件")
        print("2. 检查Graphviz安装路径是否正确")
        print("3. 尝试手动添加Graphviz的bin目录到系统环境变量PATH中")
        print("4. 重启IDE或命令行窗口")
        print("\n常见安装路径：")
        for path in graphviz_paths:
            print(f"- {path}")


if __name__ == "__main__":
    visualize_model_graphviz()