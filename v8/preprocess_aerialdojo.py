"""
AerialDojo (AerialBench / N_Island_0) → HDF5 转换器
===================================================
把离线录制的多视角 RGB-D 轨迹转成 LeWM voxel 训练用的 HDF5：
    voxel : (N, C, Z, Y, X)  float32   每步一帧，四视角已融合
    action: (N, 1)           int64     下一步动作索引（6 类）
    [llm_tokens]: (N, 64)    int64     --emit_language 时填（JEPA3D 不用，留给 VLA）

用法（在服务器、v7 目录下运行，需 torch + h5py + Pillow + transformers）：
    # 先小跑 2 条轨迹验证
    python preprocess_aerialdojo.py \
        --root /DATA/DATANAS1/UE_EXE/TASKS_Record_5GPU \
        --task BaseTasks --split Trainset \
        --out data/aerialdojo_base_train.h5 \
        --max_episodes 2 --max_steps 20
    # 全量
    python preprocess_aerialdojo.py \
        --root /DATA/DATANAS1/UE_EXE/TASKS_Record_5GPU \
        --task BaseTasks --split Trainset \
        --out data/aerialdojo_base_train.h5

注意：
- 体素 spec 必须与训练 config（lewm_voxel_aerialdojo.yaml 的 voxel_spec）保持一致！
  这里默认就是 A100 升级档；若想先快速冒烟，可加 --z_cells 16 --y_cells 48 --x_cells 48 --voxel_size 0.5 --samples 12。
- AerialDojo 动作只有 6 类（forward/ascend/descend/rotl/rotr/stop），这里用 6 类映射。
- 深度 0 或 ~65.535m 视为无效，统一置为 1e3（被 voxelizer 按 max_depth 截断成"无近表面"）。
- 只收录 status.reached_goal == True 的轨迹，且每步四视角齐全才入库。
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# 让 `from core.xxx` / `from data.xxx` 可导入（从 v7 目录运行）
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PIL import Image
import h5py

import torch
from core.voxel import VoxelSpec, RGBDVoxelizer
from data.transforms.voxelize import VoxelizeTransform

# AerialDojo 动作集（与现有 8 类 ACTION_MAP 不同：没有 left/right）
ACTION_MAP = {
    'forward': 0,
    'ascend': 1,
    'descend': 2,
    'rotl': 3,
    'rotr': 4,
    'stop': 5,
}
CAMERAS = ['front', 'left', 'right', 'down']


def resolve_frame_path(root, ep, saved_path, modality):
    """README §6：录制机上的绝对路径若失效，用任务划分 + basename 重建。"""
    p = Path(saved_path)
    if p.is_file():
        return str(p)
    try:
        st = ep.get('source_task', {}) or {}
        traj_dir = (Path(root) / st['task_directory'] / st['split_directory'] / ep['episode_id'])
        if modality == 'depth':
            return str(traj_dir / 'depth' / p.name)
        return str(traj_dir / p.name)
    except Exception:
        return None


def load_rgb(path):
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.uint8)  # (H, W, 3)


def load_depth(path, max_valid=65.535 - 1e-3):
    d = np.load(path, allow_pickle=False).astype(np.float32)  # (H, W) 米
    valid = np.isfinite(d) & (d > 0.0) & (d < max_valid)
    d[~valid] = 1e3  # 无效深度 → 远大于 max_depth，voxelizer 会按 max_depth 截断
    return d


def build_step_voxel(voxelizer, vtransform, step, root, ep):
    """对一步的四视角 RGB-D 各自体素化后融合，返回 (C,Z,Y,X) tensor 或 None。"""
    voxels = []
    for cam in CAMERAS:
        rgb_path = resolve_frame_path(root, ep, step['rgb'][cam], 'rgb')
        dep_path = resolve_frame_path(root, ep, step['depth'][cam], 'depth')
        if rgb_path is None or dep_path is None or not (os.path.isfile(rgb_path) and os.path.isfile(dep_path)):
            return None
        rgb = load_rgb(rgb_path)
        depth = load_depth(dep_path)
        v, _ = voxelizer.build(rgb, depth)
        voxels.append(v)
    return vtransform._combine_views(voxels)  # (C, Z, Y, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/DATA/DATANAS1/UE_EXE/TASKS_Record_5GPU')
    ap.add_argument('--task', default='BaseTasks', choices=['BaseTasks', 'StandardTasks', 'LongTasks'])
    ap.add_argument('--split', default='Trainset', choices=['Trainset', 'Testset'])
    ap.add_argument('--out', default='data/aerialdojo_base_train.h5')
    # 体素 spec（必须与训练 config 一致）
    ap.add_argument('--z_cells', type=int, default=32)
    ap.add_argument('--y_cells', type=int, default=96)
    ap.add_argument('--x_cells', type=int, default=96)
    ap.add_argument('--voxel_size', type=float, default=0.25)
    ap.add_argument('--max_depth', type=float, default=20.0)
    ap.add_argument('--z_min', type=float, default=-4.0)
    ap.add_argument('--use_color', action='store_true', default=True)
    ap.add_argument('--samples', type=int, default=48)
    # 控制规模
    ap.add_argument('--max_episodes', type=int, default=None)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--emit_language', action='store_true',
                    help='把 object_name 写成 llm_tokens（JEPA3D 不用，留给 VLA）')
    # 必须与 configs/train/model/vla_jepa.yaml 的 llm_model_name 完全一致！
    # 否则写出的 llm_tokens（token id）喂进 LLM 时词表对不上，语义全乱。
    ap.add_argument('--llm_model_name', default='prajjwal1/bert-tiny',
                    help='语言分支用的冻结 LLM（需与 vla_jepa.yaml 的 llm_model_name 一致）')
    args = ap.parse_args()

    spec = VoxelSpec(z_cells=args.z_cells, y_cells=args.y_cells, x_cells=args.x_cells,
                     voxel_size=args.voxel_size, max_depth=args.max_depth, z_min=args.z_min,
                     use_color=args.use_color, samples=args.samples)
    voxelizer = RGBDVoxelizer(spec)
    vtransform = VoxelizeTransform(spec)
    C = spec.in_channels

    # 语言分支：tokenizer 只需建一次（必须与 vla_jepa.yaml 的 llm_model_name 同词表）
    tokenizer = None
    if args.emit_language:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
        print(f"[preprocess] language tokenizer: {args.llm_model_name}")

    index_path = os.path.join(args.root, args.task, args.split, 'collected_episodes.json')
    print(f"[preprocess] reading index: {index_path}")
    with open(index_path, 'r') as f:
        episodes = json.load(f)  # JSON 数组
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    print(f"[preprocess] {len(episodes)} episodes to scan")

    t0 = time.time()
    skipped = 0
    n_ep = len(episodes)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 预创建可扩展数据集：增量 append，避免全量 ~470GB 一次性堆内存导致 OOM。
    # llm_tokens 强制 int64，绝不把模型名当 dtype 传进去（否则 h5py 会报
    # np.dtype('prajjwal1/bert-tiny') 的诡异错误）。
    with h5py.File(args.out, 'w') as f:
        vox_ds = f.create_dataset(
            'voxel',
            data=np.zeros((0, C, args.z_cells, args.y_cells, args.x_cells), dtype=np.float32),
            maxshape=(None, C, args.z_cells, args.y_cells, args.x_cells))
        act_ds = f.create_dataset('action', data=np.zeros((0, 1), dtype=np.int64),
                                  maxshape=(None, 1))
        tok_ds = (f.create_dataset('llm_tokens', data=np.zeros((0, 64), dtype=np.int64),
                                   maxshape=(None, 64)) if args.emit_language else None)
        f.attrs['task'] = args.task
        f.attrs['split'] = args.split
        f.attrs['action_map'] = json.dumps(ACTION_MAP)
        f.attrs['voxel_spec'] = json.dumps(vars(spec))

        for i, ep in enumerate(episodes):
            ep_id = ep.get('episode_id', f'#{i}')
            reached = (ep.get('status', {}) or {}).get('reached_goal', False)
            if not reached:
                skipped += 1
                print(f"[{i+1}/{n_ep}] SKIP {ep_id} (reached_goal=False) elapsed={time.time()-t0:.1f}s",
                      flush=True)
                continue
            steps = ep.get('steps', [])
            print(f"[{i+1}/{n_ep}] episode {ep_id} steps={len(steps)} elapsed={time.time()-t0:.1f}s",
                  flush=True)
            evox, eact, etok = [], [], []
            for t, step in enumerate(steps):
                if args.max_steps is not None and t >= args.max_steps:
                    print(f"    cap at max_steps={args.max_steps} kept={len(evox)} "
                          f"skipped={skipped} elapsed={time.time()-t0:.1f}s", flush=True)
                    break
                action = step.get('action')
                if action not in ACTION_MAP:
                    skipped += 1
                    continue
                vox = build_step_voxel(voxelizer, vtransform, step, args.root, ep)
                if vox is None:
                    skipped += 1
                    print(f"    step {t}: voxel=None (missing frame) kept={len(evox)} "
                          f"skipped={skipped} elapsed={time.time()-t0:.1f}s", flush=True)
                    continue
                evox.append(vox.cpu().numpy().astype(np.float32))
                eact.append([ACTION_MAP[action]])
                if args.emit_language:
                    # AerialDojo 的 description 为空，语义目标在 object_name / episode_id 里：
                    #   episode_id = "<start>_to_<goal>"  ->  goal 即 object_<goal>
                    goal = (ep.get('object_name')
                            or (ep.get('episode_id', '').split('_to_')[-1] if ep.get('episode_id') else None)
                            or 'target')
                    desc = f"fly to {goal}"
                    etok.append(tokenizer(desc, padding='max_length', truncation=True,
                                          max_length=64, return_tensors='pt')['input_ids']
                                .squeeze(0).numpy().astype(np.int64))
                print(f"    step {t}: OK kept={len(evox)} skipped={skipped} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
            # 每个 episode 处理完就落盘一块，峰值内存仅一个 episode 量级，杜绝全量 OOM
            if evox:
                vox_arr = np.stack(evox, axis=0).astype(np.float32)
                act_arr = np.array(eact, dtype=np.int64)
                n = vox_arr.shape[0]
                cur = vox_ds.shape[0]
                vox_ds.resize(cur + n, axis=0); vox_ds[cur:cur + n] = vox_arr
                act_ds.resize(cur + n, axis=0); act_ds[cur:cur + n] = act_arr
                if tok_ds is not None and etok:
                    tok_arr = np.stack(etok, axis=0).astype(np.int64)
                    tok_ds.resize(cur + n, axis=0); tok_ds[cur:cur + n] = tok_arr
                print(f"[{i+1}/{n_ep}] appended kept={len(evox)} total={cur + n} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

    # 收尾统计（重新打开读）
    with h5py.File(args.out, 'r') as f:
        n = f['voxel'].shape[0]
        act_all = f['action'][:, 0]
        has_lang = 'llm_tokens' in f
    if n == 0:
        raise RuntimeError("没有得到有效样本，请检查 root / task / split 与录制完整性")
    print(f"[preprocess] wrote {args.out}  (total {time.time()-t0:.1f}s)")
    print(f"  voxel : ({n}, {C}, {args.z_cells}, {args.y_cells}, {args.x_cells}) float32  (C={C})")
    print(f"  action: ({n}, 1) int64")
    if has_lang:
        print(f"  llm_tokens: ({n}, 64) int64  ✓")
    print(f"  skipped: {skipped}")
    print(f"  class counts: " + ", ".join(f"{k}={int((act_all == v).sum())}" for k, v in ACTION_MAP.items()))


if __name__ == '__main__':
    main()
