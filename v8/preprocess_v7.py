import os, cv2, json, torch, numpy as np, h5py
from tqdm import tqdm
from transformers import AutoTokenizer

ACTION_MAP = {'forward':0,'left':1,'right':2,'descend':3,'ascend':4,'rotl':5,'rotr':6,'stop':7}
root = "/DATA/DATANAS2/jzq26/DATA/collected_vla/"
out = "/villa/lct25-srt/v7_preprocessed.h5"

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

tasks = []
for jf in tqdm([f for f in os.listdir(root) if f.endswith('.json')], desc="Scanning"):
    try:
        with open(os.path.join(root,jf)) as f:
            for line in f:
                if not line.strip(): continue
                try:
                    d = json.loads(line.strip(), strict=False)
                    if 'steps' in d and d['steps']:
                        desc = d.get('description','')
                        for s in d['steps']:
                            if 'rgb' in s and 'front' in s['rgb']:
                                tasks.append({'rgb':s['rgb']['front'].replace('DATA/collected_vla/',''),'depth':s['depth']['front'].replace('DATA/collected_vla/',''),'action':s.get('action','stop'),'description':desc})
                except: continue
    except: continue

print(f"Found {len(tasks)} tasks")
with h5py.File(out,'w') as f:
    f.create_dataset('pixels', shape=(len(tasks),4,3,224,224), dtype=np.float32)
    f.create_dataset('depth', shape=(len(tasks),4,224,224), dtype=np.float32)
    f.create_dataset('action', shape=(len(tasks),), dtype=np.int64)
    f.create_dataset('llm_tokens', shape=(len(tasks),64), dtype=np.int64)
    for i,t in enumerate(tqdm(tasks, desc="Processing")):
        for angle, rgb_frames, depth_frames in [('front',[],[]), ('left',[],[]), ('right',[],[]), ('down',[],[])]:
            img = cv2.imread(os.path.join(root, t['rgb'].replace('front', angle)))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_frames.append(cv2.resize(img,(224,224)).transpose(2,0,1).astype(np.float32)/255.0)
            dep = cv2.imread(os.path.join(root, t['depth'].replace('front', angle)), cv2.IMREAD_UNCHANGED)
            if dep is not None:
                dep = dep.astype(np.float32)/65535.0
                depth_frames.append(cv2.resize(dep,(224,224)))
        if len(rgb_frames)==4: f['pixels'][i] = np.stack(rgb_frames)
        if len(depth_frames)==4: f['depth'][i] = np.stack(depth_frames)
        f['action'][i] = ACTION_MAP.get(t['action'], ACTION_MAP['stop'])
        tokens = tokenizer(t.get('description',''), padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        f['llm_tokens'][i] = tokens['input_ids'].squeeze(0).numpy()
print(f"✅ Saved to {out}")
