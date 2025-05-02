import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from pywebio.input import *
from pywebio.output import *
from pywebio.session import *
from pywebio import start_server
import argparse
from model import ClassificationModel3D
import os
from datetime import datetime

# 设置样式
style = """
<style>
body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    margin: 0;
    padding: 0;
}

.header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 20px;
    text-align: center;
    border-radius: 10px;
    margin-bottom: 30px;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

.result-box {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.upload-box {
    border: 2px dashed #ccc;
    padding: 20px;
    text-align: center;
    border-radius: 10px;
    background: #f8f9fa;
    margin: 20px 0;
}

.info-box {
    background: #e9ecef;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}

.button {
    background: #007bff;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.3s;
}

.button:hover {
    background: #0056b3;
}

.loading {
    display: inline-block;
    width: 50px;
    height: 50px;
    border: 3px solid rgba(0,0,0,.3);
    border-radius: 50%;
    border-top-color: #007bff;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}
</style>
"""

def load_model():
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassificationModel3D(dropout=0.4, dropout2=0.4)
    
    # 加载模型权重
    model_path = "model_save/fracatlas_model_best.pth"  # 替换为你的模型路径
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(img):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    return transform(img)

def predict(model, device, img_tensor):
    """进行预测"""
    with torch.no_grad():
        # 添加批次维度和深度维度
        img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
        img_tensor = img_tensor.unsqueeze(2)  # 添加深度维度
        img_tensor = img_tensor.repeat(1, 1, 160, 1, 1)  # 重复到160个切片
        img_tensor = img_tensor.to(device)
        
        outputs = model(img_tensor)
        binary_output = outputs[:, 0]  # 使用第一个输出作为二分类结果
        probability = torch.sigmoid(binary_output)
        prediction = (probability > 0.5).item()
        
        return prediction, probability.item()

def get_diagnosis_text(prediction, probability):
    """根据预测结果生成诊断文本"""
    if prediction:
        severity = "严重" if probability > 0.8 else "中等"
        return {
            "result": "检测到踝关节损伤",
            "severity": severity,
            "probability": f"{probability:.2%}",
            "recommendation": "建议及时就医检查，进行专业的医学诊断和治疗。",
            "precautions": [
                "避免继续负重行走",
                "使用冰敷减轻肿胀",
                "保持踝关节抬高位置",
                "及时就医进行专业检查"
            ]
        }
    else:
        return {
            "result": "未检测到明显的踝关节损伤",
            "severity": "轻微",
            "probability": f"{probability:.2%}",
            "recommendation": "建议继续观察，如果症状持续或加重，请及时就医。",
            "precautions": [
                "适当休息，避免剧烈运动",
                "注意保暖",
                "如果出现疼痛加重，及时就医",
                "可以进行适度的踝关节活动"
            ]
        }

def display_diagnosis(diagnosis):
    """显示诊断结果"""
    with use_scope('result', clear=True):
        put_html("""
            <div class="result-box">
                <h3 style="color: #2a5298;">诊断结果</h3>
                <hr>
        """)
        
        put_html(f"""
            <div class="info-box">
                <p><strong>诊断结果：</strong>{diagnosis['result']}</p>
                <p><strong>损伤程度：</strong>{diagnosis['severity']}</p>
                <p><strong>置信度：</strong>{diagnosis['probability']}</p>
            </div>
        """)
        
        put_html("""
            <div class="info-box">
                <h4>建议措施：</h4>
                <p>{}</p>
            </div>
        """.format(diagnosis['recommendation']))
        
        put_html("""
            <div class="info-box">
                <h4>注意事项：</h4>
                <ul>
        """)
        
        for precaution in diagnosis['precautions']:
            put_html(f"<li>{precaution}</li>")
        
        put_html("""
                </ul>
            </div>
        """)
        
        put_html("</div>")  # 关闭 result-box

def main():
    """主函数"""
    # 加载模型
    model, device = load_model()
    
    # 设置页面
    set_env(title="踝关节损伤智能诊断系统")
    
    while True:
        clear()
        
        # 显示页面标题和说明
        put_html(style)
        put_html("""
            <div class="header">
                <h1>踝关节损伤智能诊断系统</h1>
                <p>基于深度学习的智能诊断平台</p>
            </div>
        """)
        
        put_html("""
            <div class="container">
                <div class="info-box">
                    <h3>使用说明</h3>
                    <p>1. 上传踝关节X光片图像（支持jpg、png格式）</p>
                    <p>2. 系统将自动分析图像并给出诊断建议</p>
                    <p>3. 本系统仅供参考，请以专业医生的诊断为准</p>
                </div>
            </div>
        """)
        
        # 图片上传
        put_html('<div class="upload-box">')
        img = file_upload("上传X光片图像", accept="image/*", required=True)
        put_html('</div>')
        
        if img is not None:
            # 显示加载动画
            put_html('<div style="text-align: center;"><div class="loading"></div></div>')
            
            # 处理图像
            img_content = img['content']
            img = Image.open(io.BytesIO(img_content))
            img_tensor = preprocess_image(img)
            
            # 进行预测
            prediction, probability = predict(model, device, img_tensor)
            
            # 生成诊断结果
            diagnosis = get_diagnosis_text(prediction, probability)
            
            # 清除加载动画
            clear('loading')
            
            # 显示诊断结果
            display_diagnosis(diagnosis)
        
        # 添加重新诊断按钮
        put_html("""
            <div style="text-align: center; margin-top: 20px;">
                <button class="button" onclick="window.location.reload()">重新诊断</button>
            </div>
        """)
        
        # 添加页脚
        put_html("""
            <div class="footer">
                <p>© 2024 踝关节损伤智能诊断系统 | 仅供研究使用</p>
                <p>如有疑问，请咨询专业医生</p>
            </div>
        """)
        
        # 等待用户选择是否继续
        choice = actions('', ['继续诊断', '退出系统'])
        if choice == '退出系统':
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    
    start_server(main, port=args.port)