"""
Offline shape/linkage test for the rewritten VoxelJEPAEncoder + VLAJEPA.

The managed environment has no real torch, so we stub torch / torch.nn /
einops / transformers with a numpy-backed Tensor that ONLY tracks shapes
(and basic arithmetic so the module code doesn't crash). This verifies that:
  - resolution now drives token count N (NOT a fixed 18)
  - backward-compatible forward() still returns (B, latent_dim)
  - VLAJEPA cross-attention keys == N real voxel tokens
  - action_pred is (B, 8) and T>1 no longer crashes
"""
import os
import sys
import math
import types
import numpy as np

# ----------------------------------------------------------------------------
# Tensor stub (shape-tracking only)
# ----------------------------------------------------------------------------
class Tensor:
    def __init__(self, data, device='cpu'):
        if isinstance(data, Tensor):
            data = data._a
        if isinstance(data, (int, float)):
            data = np.array(data)
        self._a = np.asarray(data)
        self.device = device
        self.requires_grad = False

    @property
    def shape(self):
        return tuple(self._a.shape)

    def dim(self):
        return self._a.ndim

    def float(self):
        return Tensor(self._a.astype(np.float32), self.device)

    def long(self):
        return Tensor(self._a.astype(np.int64), self.device)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self._a, dim), self.device)

    def reshape(self, *shape):
        shape = tuple(int(s) for s in shape)
        if -1 in shape:
            total = int(np.prod(self._a.shape))
            known = -int(np.prod([s for s in shape if s != -1]))
            rem = total // known
            shape = tuple(rem if s == -1 else s for s in shape)
        return Tensor(self._a.reshape(shape), self.device)

    def permute(self, *dims):
        return Tensor(np.transpose(self._a, dims), self.device)

    def mean(self, dim=None):
        if dim is None:
            return Tensor(self._a.mean(), self.device)
        return Tensor(np.mean(self._a, axis=dim), self.device)

    def expand(self, *shape):
        shape = tuple(int(s) for s in shape)
        shape = tuple(self._a.shape[i] if s == -1 else s for i, s in enumerate(shape))
        return Tensor(np.broadcast_to(self._a, shape), self.device)

    def squeeze(self, dim=None):
        if dim is None:
            a = np.squeeze(self._a)
        else:
            a = np.squeeze(self._a, axis=dim)
        return Tensor(a, self.device)

    def __add__(self, other):
        o = other._a if isinstance(other, Tensor) else other
        return Tensor(self._a + o, self.device)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        o = other._a if isinstance(other, Tensor) else other
        return Tensor(self._a * o, self.device)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        o = other._a if isinstance(other, Tensor) else other
        return Tensor(self._a - o, self.device)

    def __truediv__(self, other):
        o = other._a if isinstance(other, Tensor) else other
        return Tensor(self._a / o, self.device)

    def __getitem__(self, key):
        return Tensor(self._a[key], self.device)

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self._a[key] = value._a
        else:
            self._a[key] = value

    def to(self, *a, **k):
        return self

    def __repr__(self):
        return f"Tensor{self.shape}"


# ----------------------------------------------------------------------------
# torch stub
# ----------------------------------------------------------------------------
torch = types.ModuleType('torch')
torch.Tensor = Tensor

def _zeros(*args, **kwargs):
    shape = tuple(int(a) for a in args if isinstance(a, (int, np.integer)))
    return Tensor(np.zeros(shape), kwargs.get('device', 'cpu'))

def _arange(*args, **kwargs):
    device = kwargs.get('device', 'cpu')
    if len(args) == 1:
        s, e, st = 0, args[0], 1
    elif len(args) == 2:
        s, e, st = args[0], args[1], 1
    else:
        s, e, st = args[0], args[1], args[2]
    return Tensor(np.arange(s, e, st), device)

def _exp(x):
    return Tensor(np.exp(x._a), x.device)

def _sin(x):
    return Tensor(np.sin(x._a), x.device)

def _cos(x):
    return Tensor(np.cos(x._a), x.device)

def _tensor(data, **kwargs):
    if isinstance(data, Tensor):
        return data
    return Tensor(np.array(data), kwargs.get('device', 'cpu'))

def _from_numpy(a):
    return Tensor(np.asarray(a), 'cpu')

def _randn(*shape):
    return Tensor(np.random.randn(*shape))

torch.zeros = _zeros
torch.arange = _arange
torch.exp = _exp
torch.sin = _sin
torch.cos = _cos
torch.tensor = _tensor
torch.from_numpy = _from_numpy
torch.randn = _randn


# ----------------------------------------------------------------------------
# nn.Module base + layers
# ----------------------------------------------------------------------------
class Module:
    def __init__(self):
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_params', {})
        object.__setattr__(self, '_buffers', {})

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Tensor):
            self._buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def parameters(self):
        out = list(self._params.values())
        for m in self._modules.values():
            out.extend(m.parameters())
        return out

    def __getattr__(self, name):
        # only called when normal attribute lookup fails
        modules = object.__getattribute__(self, '_modules')
        if name in modules:
            return modules[name]
        params = object.__getattribute__(self, '_params')
        if name in params:
            return params[name]
        buffers = object.__getattribute__(self, '_buffers')
        if name in buffers:
            return buffers[name]
        raise AttributeError(name)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        if isinstance(data, Tensor):
            super().__init__(data._a, data.device)
        else:
            super().__init__(data)
        self.requires_grad = requires_grad


class Conv3d(Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        sp = x.shape[2:]
        out_sp = tuple((s + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1 for s in sp)
        return Tensor(np.zeros((x.shape[0], self.out_ch, *out_sp)), x.device)


class BatchNorm3d(Module):
    def __init__(self, num_features, *a, **kw):
        super().__init__()

    def forward(self, x):
        return Tensor(np.zeros(x.shape), x.device)


class GELU(Module):
    def forward(self, x):
        return Tensor(np.zeros(x.shape), x.device)


class Dropout(Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        return Tensor(np.zeros(x.shape), x.device)


class Linear(Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f

    def forward(self, x):
        new = x.shape[:-1] + (self.out_f,)
        return Tensor(np.zeros(new), x.device)


class LayerNorm(Module):
    def __init__(self, normalized_shape, *a, **kw):
        super().__init__()

    def forward(self, x):
        return Tensor(np.zeros(x.shape), x.device)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        object.__setattr__(self, '_layers', list(layers))

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class ModuleList(Module):
    def __init__(self, items):
        super().__init__()
        object.__setattr__(self, '_list', list(items))

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, query, key, value, need_weights=True, **kw):
        out = Tensor(np.zeros(query.shape), query.device)
        if need_weights:
            weights = Tensor(np.zeros((query.shape[0], query.shape[1], key.shape[1])), query.device)
            return out, weights
        return out


nn = types.ModuleType('torch.nn')
for nm, cls in dict(
    Module=Module, Conv3d=Conv3d, BatchNorm3d=BatchNorm3d, GELU=GELU,
    Dropout=Dropout, Linear=Linear, LayerNorm=LayerNorm, Sequential=Sequential,
    ModuleList=ModuleList, MultiheadAttention=MultiheadAttention, Parameter=Parameter,
).items():
    setattr(nn, nm, cls)
torch.nn = nn

nnf = types.ModuleType('torch.nn.functional')

def _mse(a, b):
    return Tensor(np.array(0.0), a.device)

def _ce(a, b):
    return Tensor(np.array(0.0), a.device)

nnf.mse_loss = _mse
nnf.cross_entropy = _ce
torch.nn.functional = nnf


# ----------------------------------------------------------------------------
# einops.rearrange (minimal, handles the 4 patterns used)
# ----------------------------------------------------------------------------
import re

def rearrange(tensor, pattern, **axes):
    lhs, rhs = pattern.split("->")
    tok = lambda s: re.findall(r"\([^)]*\)|\S+", s)
    in_names = tok(lhs)
    out_names = tok(rhs)
    sizes = {}
    for i, nm in enumerate(in_names):
        sizes[nm] = tensor.shape[i]
    for k, v in axes.items():
        sizes[k] = v
    out_shape = []
    for nm in out_names:
        m = re.match(r"\(([^)]+)\)", nm)
        if m:
            prod = 1
            for p in m.group(1).split():
                prod *= sizes[p]
            out_shape.append(prod)
        else:
            out_shape.append(sizes[nm])
    return tensor.reshape(-1).reshape(*out_shape)


einops = types.ModuleType('einops')
einops.rearrange = rearrange

tf = types.ModuleType('transformers')

class _AutoModel:
    @staticmethod
    def from_pretrained(name):
        return Module()

tf.AutoModel = _AutoModel

# register stub packages BEFORE importing production code
sys.modules['torch'] = torch
sys.modules['torch.nn'] = nn
sys.modules['torch.nn.functional'] = nnf
sys.modules['einops'] = einops
sys.modules['transformers'] = tf

# ----------------------------------------------------------------------------
# load the real production modules as part of package `core`
# ----------------------------------------------------------------------------
CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core')
core_pkg = types.ModuleType('core')
core_pkg.__path__ = [CORE_DIR]
core_pkg.__package__ = 'core'
sys.modules['core'] = core_pkg

import importlib.util

def load_core(name):
    path = os.path.join(CORE_DIR, name + '.py')
    spec = importlib.util.spec_from_file_location('core.' + name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'core'
    sys.modules['core.' + name] = mod
    spec.loader.exec_module(mod)
    return mod

load_core('cross_attention')   # must exist before vla_jepa import
voxel_mod = load_core('voxel')
vla_jepa_mod = load_core('vla_jepa')

VoxelJEPAEncoder = voxel_mod.VoxelJEPAEncoder
VoxelSpec = voxel_mod.VoxelSpec
VLAJEPA = vla_jepa_mod.VLAJEPA
CrossAttentionModule = sys.modules['core.cross_attention'].CrossAttentionModule


# ----------------------------------------------------------------------------
# dummy modules for VLAJEPA dependencies
# ----------------------------------------------------------------------------
class MLP(Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = Linear(in_f, out_f)

    def forward(self, x, *a):
        return self.lin(x)


# ----------------------------------------------------------------------------
# TEST 1: encoder unit-level — resolution drives N
# ----------------------------------------------------------------------------
spec16 = VoxelSpec(z_cells=16, y_cells=48, x_cells=48, use_color=False)   # 3ch
enc = VoxelJEPAEncoder(spec=spec16, latent_dim=64, token_dim=128, num_stages=3, base_channels=32)

vox16 = Tensor(np.zeros((2, 3, 16, 48, 48)))
pooled, tokens = enc.encode(vox16)
assert pooled.shape == (2, 64), pooled.shape
N16 = tokens.shape[1]
assert tokens.shape == (2, N16, 128), tokens.shape
assert N16 == 72, f"expected 72, got {N16}"

# backward-compatible forward()
out_fwd = enc.forward(vox16)
assert out_fwd.shape == (2, 64), out_fwd.shape

print(f"[T1] 16x48x48 3ch : pooled{ pooled.shape } tokens{ tokens.shape } N={N16}")

# higher resolution, 6 channels (color on)
spec32 = VoxelSpec(z_cells=32, y_cells=96, x_cells=96, use_color=True)     # 6ch
enc2 = VoxelJEPAEncoder(spec=spec32, latent_dim=64, token_dim=128, num_stages=3, base_channels=32)
vox32 = Tensor(np.zeros((2, 6, 32, 96, 96)))
_, tokens2 = enc2.encode(vox32)
N32 = tokens2.shape[1]
assert tokens2.shape == (2, N32, 128), tokens2.shape
assert N32 == 576, f"expected 576, got {N32}"
print(f"[T1] 32x96x96 6ch : tokens{ tokens2.shape } N={N32}  (ratio {N32//N16}x)")

# ----------------------------------------------------------------------------
# TEST 2: VLAJEPA end-to-end — cross-attn keys == N, action_pred (B,8)
# ----------------------------------------------------------------------------
def build_vlajepa(encoder):
    return VLAJEPA(
        jepa_encoder=encoder,
        jepa_predictor=MLP(64, 64),
        action_encoder=MLP(8, 64),
        projector=MLP(64, 64),
        pred_proj=MLP(64, 64),
        llm_model_name="gpt2",
        llm_dim=128,
        jepa_dim=64,
        jepa_token_dim=128,
        num_heads=8,
        use_adaln=True,
        freeze_llm=True,
    )

# T=4 multi-frame (exercises JEPA predictor branch + batch reshape)
B, T = 2, 4
vlaj = build_vlajepa(enc)
batch = {
    'voxel': Tensor(np.zeros((B, T, 3, 16, 48, 48))),
    'action': Tensor(np.zeros((B,)), device='cpu'),
}
out = vlaj.forward(batch)
ap = out['action_pred']
assert ap.shape == (B, 8), ap.shape
attn_N = out['attn_weights'].shape[2]
assert attn_N == N16, f"attn keys {attn_N} != N {N16}"
print(f"[T2] 16x48x48 T={T}: action_pred{ ap.shape } attn_key_N {attn_N}")

# T=1 single-frame (skips predictor branch)
vlaj1 = build_vlajepa(enc2)
batch1 = {
    'voxel': Tensor(np.zeros((B, 1, 6, 32, 96, 96))),
    'action': Tensor(np.zeros((B,))),
}
out1 = vlaj1.forward(batch1)
ap1 = out1['action_pred']
assert ap1.shape == (B, 8), ap1.shape
attn_N1 = out1['attn_weights'].shape[2]
assert attn_N1 == N32, f"attn keys {attn_N1} != N {N32}"
print(f"[T2] 32x96x96 T=1: action_pred{ ap1.shape } attn_key_N {attn_N1}")

# ----------------------------------------------------------------------------
# TEST 3: VLAJEPA 语言条件路径（llm_tokens -> 冻结 LLM [CLS] -> cross-attn query）
# ----------------------------------------------------------------------------
class FakeLLM(Module):
    """桩：冻结 LLM，forward 直接返回 (B, L, hidden) 的 hidden 表征。"""
    def forward(self, input_ids, **kw):
        B, L = input_ids.shape[0], input_ids.shape[1]
        return Tensor(np.zeros((B, L, 128)), input_ids.device)

Bt, Tt, Lt = 2, 4, 10
vlaj_lang = build_vlajepa(enc)
vlaj_lang.llm = FakeLLM()                      # 替换 stub AutoModel，使其 forward 可用

# 3a) 2D 输入 (B, L) —— 退化情况（num_steps=1 时数据集产出）
batch_lang2d = {
    'voxel': Tensor(np.zeros((Bt, Tt, 3, 16, 48, 48))),
    'action': Tensor(np.zeros((Bt, Tt, 1))),
    'llm_tokens': Tensor(np.zeros((Bt, Lt), dtype=np.int64)),
}
out3a = vlaj_lang.forward(batch_lang2d)
ap3a = out3a['action_pred']
assert ap3a.shape == (Bt, 8), ap3a.shape
attn3a = out3a['attn_weights']
assert attn3a.shape[1] == 1, f"语言 query 应为单 [CLS] 向量，得到 {attn3a.shape[1]}"
assert attn3a.shape[2] == N16, f"attn keys {attn3a.shape[2]} != N {N16}"
print(f"[T3a] lang 2D  T={Tt} L={Lt}: action_pred{ ap3a.shape } attn_q={attn3a.shape[1]} attn_key_N {attn3a.shape[2]}")

# 3b) 3D 输入 (B, T, L) —— 真实情况（num_steps>1 数据集把 llm_tokens 堆叠成 B,T,L）
batch_lang3d = {
    'voxel': Tensor(np.zeros((Bt, Tt, 3, 16, 48, 48))),
    'action': Tensor(np.zeros((Bt, Tt, 1))),
    'llm_tokens': Tensor(np.zeros((Bt, Tt, Lt), dtype=np.int64)),
}
out3b = vlaj_lang.forward(batch_lang3d)
ap3b = out3b['action_pred']
assert ap3b.shape == (Bt, 8), ap3b.shape
attn3b = out3b['attn_weights']
assert attn3b.shape[1] == 1, f"语言 query 应为单 [CLS] 向量，得到 {attn3b.shape[1]}"
assert attn3b.shape[2] == N16, f"attn keys {attn3b.shape[2]} != N {N16}"
print(f"[T3b] lang 3D  T={Tt} L={Lt}: action_pred{ ap3b.shape } attn_q={attn3b.shape[1]} attn_key_N {attn3b.shape[2]}")

# ----------------------------------------------------------------------------
print()
print("=== 全通过：分辨率 16x48x48 -> N=72, 32x96x96 -> N=576（8x），"
      "cross-attn 真正吃真实体素 token；T>1 不再崩溃 ===")
