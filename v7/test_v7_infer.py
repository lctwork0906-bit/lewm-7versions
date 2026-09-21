import torch
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from core.rgb_vla import RGBVLA
from data.formats.png_rgbd import PNGDepthDataset

print("加载配置...")
cfg = OmegaConf.load("configs/train/rgb_vla_hdf5.yaml")

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
print("模型加载成功！")

print("加载测试数据...")
dataset = PNGDepthDataset("/DATA/DATANAS2/jzq26/DATA/collected_vla/")
sample = dataset[0]
pixels = sample['pixels']  # (4, 3, 512, 512)
depth = sample['depth']    # (4, 512, 512)

# resize到224
pixels = TF.resize(pixels, [224, 224])
depth = TF.resize(depth.unsqueeze(0), [224, 224]).squeeze(0)

pixels = pixels.unsqueeze(0)  # (1, 4, 3, 224, 224)
depth = depth.unsqueeze(0)    # (1, 4, 224, 224)
action = sample['action']

print(f"pixels shape: {pixels.shape}")
print(f"depth shape: {depth.shape}")
print(f"真实动作: {action.item()}")

print("推理...")
with torch.no_grad():
    batch = {'pixels': pixels, 'depth': depth}
    output = model(batch)
    pred_action = output['action_pred']
    action_idx = pred_action.argmax(dim=-1).item()

print(f"预测动作索引: {action_idx}")
print("完成！")
