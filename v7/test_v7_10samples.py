import torch
import numpy as np
import torchvision.transforms.functional as TF
from core.rgb_vla import RGBVLA
from data.formats.png_rgbd import PNGDepthDataset

# 动作映射
ACTION_NAMES = ['forward', 'left', 'right', 'descend', 'ascend', 'rotl', 'rotr', 'stop']

print("加载模型...")
model = RGBVLA(
    img_size=224,
    in_chans=4,
    num_actions=8,
    dropout=0.3,
    llm_model_name="google/bert_uncased_L-2_H-128_A-2",
    llm_dim=128,
    num_heads=4,
    freeze_llm=True
)

state_dict = torch.load("/villa/lct25-srt/.stable-wm/checkpoints/v7_hdf5/weights_epoch_500.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()
print("模型加载成功！\n")

print("加载测试数据...")
dataset = PNGDepthDataset("/DATA/DATANAS2/jzq26/DATA/collected_vla/")
print(f"数据集大小: {len(dataset)} 个样本\n")

print("=" * 60)
print("测试前10个样本")
print("=" * 60)

correct = 0
total = 10

for i in range(10):
    sample = dataset[i]
    pixels = sample['pixels']
    depth = sample['depth']
    action_true = sample['action'].item()
    
    # Resize到224
    pixels = TF.resize(pixels, [224, 224])
    depth = TF.resize(depth.unsqueeze(0), [224, 224]).squeeze(0)
    
    pixels = pixels.unsqueeze(0)
    depth = depth.unsqueeze(0)
    
    with torch.no_grad():
        batch = {'pixels': pixels, 'depth': depth}
        output = model(batch)
        pred_action = output['action_pred']
        action_pred_idx = pred_action.argmax(dim=-1).item()
    
    is_correct = (action_pred_idx == action_true)
    if is_correct:
        correct += 1
    
    print(f"样本 {i+1:2d}: 真实={ACTION_NAMES[action_true]:>8s}, 预测={ACTION_NAMES[action_pred_idx]:>8s} {'✅' if is_correct else '❌'}")

print("=" * 60)
print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")
print("=" * 60)
