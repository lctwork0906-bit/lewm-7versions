"""
只重写 HDF5 里的 llm_tokens（换成 jsonl 真实物体描述），不重新体素化。
============================================================
适用前提：目标 HDF5 是用 preprocess_aerialdojo.py 默认参数（无 --max_steps /
--episode_offset / 分片）全量生成的。脚本会复现相同的「episode 顺序 + 每步收录
规则」，给每个样本分配对应描述，并**校验样本总数 N 与原 HDF5 一致**后才写入——
不一致会直接报错退出，绝不会悄悄写坏文件。

用法：
    python regen_language.py \
        --root /DATA/DATANAS1/UE_EXE/TASKS_Record_5GPU \
        --task BaseTasks --split Trainset \
        --in  data/aerialdojo_vla_train.h5 \
        --out data/aerialdojo_vla_train.langfix.h5 \
        --llm_model_name prajjwal1/bert-tiny
"""
import os
import sys
import json
import argparse

import numpy as np
import h5py
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ACTION_MAP = {'forward': 0, 'ascend': 1, 'descend': 2, 'rotl': 3, 'rotr': 4, 'stop': 5}
CAMERAS = ['front', 'left', 'right', 'down']


def resolve_frame_path(root, ep, saved_path, modality):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/DATA/DATANAS1/UE_EXE/TASKS_Record_5GPU')
    ap.add_argument('--task', default='BaseTasks')
    ap.add_argument('--split', default='Trainset')
    ap.add_argument('--in', dest='inp', required=True, help='现有 HDF5（含 voxel/action/llm_tokens）')
    ap.add_argument('--out', required=True, help='输出 HDF5（复制 voxel/action，重写 llm_tokens）')
    ap.add_argument('--llm_model_name', default='prajjwal1/bert-tiny')
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)

    # 1) 真实描述：根目录 jsonl
    desc_map = {}
    jsonl_path = os.path.join(args.root, 'collected_episodes.jsonl')
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            on = e.get('object_name')
            d = e.get('description')
            if on and d:
                desc_map[str(on)] = d
    print(f"[regen] loaded {len(desc_map)} descriptions from {jsonl_path}")

    # 2) 复现 preprocess 的样本顺序 -> 给每个样本分配文本
    index_path = os.path.join(args.root, args.task, args.split, 'collected_episodes.json')
    episodes = json.load(open(index_path))
    episodes = [e for e in episodes if (e.get('status', {}) or {}).get('reached_goal', False)]

    texts = []
    for ep in episodes:
        on = ep.get('object_name')
        real_desc = desc_map.get(on)
        if real_desc is None:
            goal = (on or (ep.get('episode_id', '').split('_to_')[-1]
                           if ep.get('episode_id') else 'target'))
            real_desc = f"fly to {goal}"
        steps = ep.get('steps', [])
        for step in steps:
            action = step.get('action')
            if action not in ACTION_MAP:
                continue
            ok = True
            for cam in CAMERAS:
                rp = resolve_frame_path(args.root, ep, step['rgb'][cam], 'rgb')
                dp = resolve_frame_path(args.root, ep, step['depth'][cam], 'depth')
                if not (rp and dp and os.path.isfile(rp) and os.path.isfile(dp)):
                    ok = False
                    break
            if not ok:
                continue
            texts.append(real_desc)

    N = len(texts)
    print(f"[regen] reconstructed {N} samples")

    # 3) 校验 N 与原 HDF5 一致（否则说明原文件不是默认全量生成，改走完整 preprocess）
    with h5py.File(args.inp, 'r') as f:
        n_existing = f['voxel'].shape[0]
        has_lang = 'llm_tokens' in f
    if not has_lang:
        raise SystemExit("[regen] 原 HDF5 没有 llm_tokens，请用 --emit_language 完整 preprocess 重跑。")
    if N != n_existing:
        raise SystemExit(
            f"[regen] 样本数不匹配！regen={N} 现有={n_existing}；\n"
            f"        说明原 HDF5 不是默认参数全量生成的（用过 --max_steps / 分片）。\n"
            f"        请改用完整 preprocess_aerialdojo.py --emit_language 重跑。")

    # 4) 重新 tokenize 并写出（复制 voxel/action，只换 llm_tokens）
    arr = np.zeros((N, 64), dtype=np.int64)
    for i, txt in enumerate(texts):
        arr[i] = (tokenizer(txt, padding='max_length', truncation=True,
                            max_length=64, return_tensors='pt')['input_ids']
                  .squeeze(0).numpy().astype(np.int64))
    print(f"[regen] tokenized {N} instructions")

    with h5py.File(args.inp, 'r') as fi, h5py.File(args.out, 'w') as fo:
        for name in ['voxel', 'action']:
            fo.create_dataset(name, data=fi[name][:])
        fo.create_dataset('llm_tokens', data=arr)
        for k in fi.attrs:
            fo.attrs[k] = fi.attrs[k]
        fo.attrs['language_source'] = 'real_description_from_jsonl'
    print(f"[regen] wrote {args.out}")

    for i in [0, N // 2, N - 1]:
        print(f"  sample {i}: {texts[i][:90]!r}")


if __name__ == '__main__':
    main()
