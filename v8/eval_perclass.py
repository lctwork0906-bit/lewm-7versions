"""
逐类动作评估：对抗"全蒙C"的硬指标。

专门统计 VLA 的 action_pred 在验证集上的：
  - 总体准确率
  - 逐类召回率(Recall) + 混淆矩阵
  - 平衡准确率(Balanced Accuracy = 各类召回均值)
  - 预测分布 vs 真实分布
  - majority 基线对比（永远猜最多的类能拿多少）

为什么需要它：训练日志里的 `pred` 是 JEPA 的"未来表征预测损失"(MSE)，
不是动作分类准确率；仓库自带的 eval.py 又是空壳。所以这个脚本是判断
"模型到底有没有学到动作区分 / 还是永远猜 forward"的唯一硬证据。

用法（在 v8 目录、激活 v1 venv、且已训练出 checkpoint 后）：
  # VLA 第 1 个 epoch 的 checkpoint（默认路径自动推断）
  python -u eval_perclass.py --config-name=lewm_vla_aerialdojo \
      model.num_actions=6 eval.epoch=1
  # 或指定任意 checkpoint 路径
  python -u eval_perclass.py --config-name=lewm_vla_aerialdojo \
      model.num_actions=6 \
      eval.checkpoint=/villa/lct25-srt/.stable-wm/checkpoints/lewm_vla_aerialdojo/weights_epoch_1.pt

注意：voxel 基线(JEPA3D)不输出 action_pred（它是表征/碰撞模型，不做动作分类），
本脚本会检测到并提示；动作分类评估请用 VLA 配置。
"""
import os
import numpy as np
import torch
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm

from core.trainer import Trainer
from data import DataStrategyRegistry

# 动作名（顺序须与 preprocess_aerialdojo.py 中动作整数定义一致）
ACTION_NAMES = ["forward", "ascend", "descend", "rotl", "rotr", "stop"]


@hydra.main(version_base=None, config_path="./configs/train",
            config_name="lewm_vla_aerialdojo")
def main(cfg):
    strategy = DataStrategyRegistry.detect(cfg.data.dataset.name)
    trainer = Trainer(cfg, strategy)          # 复用：建模型 + 数据 + loader
    model = trainer.model
    device = trainer.device

    # ---- 取 checkpoint 路径 ----
    ckpt = OmegaConf.select(cfg, "eval.checkpoint", default=None)
    if ckpt is None:
        epoch = int(OmegaConf.select(cfg, "eval.epoch", default=1))
        run_name = OmegaConf.select(cfg, "output_model_name",
                                    default="lewm_vla_aerialdojo")
        ckpt = os.path.expanduser(
            f"~/.stable-wm/checkpoints/{run_name}/weights_epoch_{epoch}.pt")
    print(f"[Eval] Loading checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # ---- 跑 val 集，收集 action_pred / target ----
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(trainer.val_loader, desc="eval"):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(batch)
            if "action_pred" not in out:
                print("[Eval] 该模型不输出 action_pred（voxel 基线 JEPA3D 是表征/碰撞"
                      "模型，不做动作分类）。本脚本对 voxel 基线不适用——请用 VLA 配置"
                      "评估动作。")
                return
            logits = out["action_pred"]
            act = batch["action"]
            if act.dim() >= 2:
                target = act[:, -1].reshape(-1).long()
            else:
                target = act.reshape(-1).long()
            pred = logits.argmax(dim=-1).reshape(-1).long()
            all_pred.append(pred.cpu().numpy())
            all_true.append(target.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    num_classes = int(OmegaConf.select(cfg, "model.num_actions",
                                       default=int(all_pred.max()) + 1))
    names = (ACTION_NAMES[:num_classes]
             if len(ACTION_NAMES) >= num_classes
             else [str(i) for i in range(num_classes)])

    # ---- 指标计算 ----
    overall = float((all_pred == all_true).mean())
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1
    recalls = []
    print("\n===== 逐类动作评估 =====")
    print(f"样本数: {len(all_true)}   类别数: {num_classes}")
    print(f"{'动作':<10}{'真实数':>8}{'预测对':>8}{'召回率':>10}")
    for c in range(num_classes):
        tot = int(cm[c].sum())
        cor = int(cm[c, c])
        rec = cor / tot if tot > 0 else float('nan')
        recalls.append(rec)
        if tot > 0:
            print(f"{names[c]:<10}{tot:>8}{cor:>8}{rec:>10.3f}")
        else:
            print(f"{names[c]:<10}{tot:>8}{cor:>8}{'  N/A':>10}")
    balanced = float(np.nanmean(recalls))

    true_dist = cm.sum(axis=1)
    pred_dist = cm.sum(axis=0)
    majority = float(true_dist.max() / true_dist.sum())
    maj_name = names[int(true_dist.argmax())]

    print(f"\n总体准确率              : {overall:.4f}")
    print(f"平衡准确率(各类召回均值): {balanced:.4f}")
    print(f"随机基线(1/{num_classes})       : {1.0/num_classes:.4f}")
    print(f"Majority基线(永远猜'{maj_name}'): {majority:.4f}")
    print("\n真实分布:", dict(zip(names, [int(x) for x in true_dist])))
    print("预测分布:", dict(zip(names, [int(x) for x in pred_dist])))

    print("\n混淆矩阵 (行=真实, 列=预测):")
    print("      " + "".join(f"{n[:4]:>6}" for n in names))
    for c in range(num_classes):
        print(f"{names[c][:4]:<6}" + "".join(f"{cm[c, p]:>6}" for p in range(num_classes)))

    print("\n===== 判定 =====")
    if balanced < 1.0 / num_classes + 0.05:
        print("  ⚠️ 平衡准确率接近随机 -> 模型基本在'蒙C'，没学到动作区分。")
    else:
        print("  ✅ 平衡准确率明显高于随机 -> 模型学到了不同动作，不是蒙C。")
    pred_frac = pred_dist.max() / pred_dist.sum()
    if pred_frac > 0.8:
        print(f"  ⚠️ 预测极度集中在 '{names[int(pred_dist.argmax())]}'"
              f"({pred_frac:.2f}) -> 仍可能是退化的。")
    else:
        print(f"  ✅ 预测分布在多个动作上 -> 没有单一动作退化。")


if __name__ == "__main__":
    main()
