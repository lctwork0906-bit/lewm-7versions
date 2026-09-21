import torch
import numpy as np
import torchvision.transforms.functional as TF
from core.rgb_vla import RGBVLA
from data.formats.png_rgbd import PNGDepthDataset

ACTION_NAMES = ['forward', 'left', 'right', 'descend', 'ascend', 'rotl', 'rotr', 'stop']
model = RGBVLA(
    img_size=224, in_chans=4, num_actions=8, dropout=0.3,
    llm_model_name="google/bert_uncased_L-2_H-128_A-2",
    llm_dim=128, num_heads=4, freeze_llm=True
)
state_dict = torch.load("/villa/lct25-srt/.stable-wm/checkpoints/v7_hdf5/weights_epoch_500.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

dataset = PNGDepthDataset("/DATA/DATANAS2/jzq26/DATA/collected_vla/")
print(f"测试前100个样本...")
ascend_count = 0
for i in range(100):
    sample = dataset[i]
    pixels = TF.resize(sample['pixels'], [224, 224]).unsqueeze(0)
    depth = TF.resize(sample['depth'].unsqueeze(0), [224, 224]).squeeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model({'pixels': pixels, 'depth': depth})
        pred = output['action_pred'].argmax(dim=-1).item()
    if pred == 4:  # ascend的索引是4
        ascend_count += 1
    if i < 20:
        print(f"样本{i}: 真实={ACTION_NAMES[sample['action'].item()]}, 预测={ACTION_NAMES[pred]}")

print(f"\n100个样本中预测为ascend的数量: {ascend_count}/100 ({ascend_count}%)")
