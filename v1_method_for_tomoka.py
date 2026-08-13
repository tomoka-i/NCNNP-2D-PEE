"""
NCNNP + 2D-PEE，mapping依照 Ou, Li, Zhao, Ni, Shi (2013)
"Pairwise Prediction-Error Expansion for Efficient Reversible Data Hiding"
IEEE TIP, Fig.3 (T=1, proposed pairwise PEE) 精確實作

跟前一版(spiral shift版)的差異：
1. 不是8方向spiral shift，而是「每軸獨立套用經典1D T=1規則」為主幹
2. 只有 A 類型 bin (h1,h2 都在 {-1,0}) 才做 joint 處理，捨棄掉最貴的角落，
   容量從 2 bit 降成 log2(3) bit，用大數進制轉換 (ternary <-> binary) 精確達成
3. 被捨棄的角落本身變成 B 類型 bin，可以自己嵌 1 bit
4. C 類型 (恰一軸在範圍內) 嵌 1 bit；D 類型 (兩軸都不在範圍內) 純 shift 不嵌
5. 落單的leftover pixel用單軸1D規則補
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from math import log10
from skimage.metrics import structural_similarity as ssim

# ============================================================
# 1. 模型定義 (NCNNP + CBAM) -- 不變
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x): return self.block(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_p = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_p = F.adaptive_max_pool2d(x, 1).view(b, c)
        out = self.mlp(avg_p) + self.mlp(max_p)
        return x * self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class NCNNP(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch3 = ConvBlock(1, 32, 3)
        self.branch5 = ConvBlock(1, 32, 5)
        self.branch7 = ConvBlock(1, 32, 7)
        self.cbam = CBAM(32)
        self.refine1 = ConvBlock(32, 32, 3)
        self.refine2 = ConvBlock(32, 32, 3)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    def forward(self, x):
        feat = self.cbam(self.branch3(x) + self.branch5(x) + self.branch7(x))
        r1 = self.refine1(feat) + feat
        r2 = self.refine2(r1) + r1
        return self.out_conv(r2)

def pick_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except: return "cpu"
    return "cpu"

# ============================================================
# 1b. 局部複雜度矩陣 (LC, 用來決定embedding優先順序)
# ============================================================
def get_full_complexity_matrix(img_arr):
    img = img_arr.astype(np.float32)
    h, w = img.shape
    pad = np.pad(img, 2, mode='edge')
    offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 0), (-1, 2),
        (0, -1), (0, 1), (1, -2), (1, 0), (1, 2),
        (2, -1), (2, 1)
    ]
    l_sum = np.zeros_like(img)
    l_sq_sum = np.zeros_like(img)
    for dy, dx in offsets:
        shifted = pad[2+dy : 2+dy+h, 2+dx : 2+dx+w]
        l_sum += shifted
        l_sq_sum += shifted**2
    n = len(offsets)
    mean = l_sum / n
    var = (l_sq_sum / n) - (mean**2)
    return np.sqrt(np.maximum(var, 0))


def get_cross_complexity_matrix(img_arr, target_parity):
    """
    Compute complexity for one checkerboard target set from the opposite set.

    Every offset has odd checkerboard parity, so each valid neighbour is in the
    opposite checkerboard set.  Unlike the legacy full-image implementation,
    out-of-image neighbours are omitted instead of edge-padded: edge padding
    can copy a target pixel at the border and break blind reproducibility.
    """
    if target_parity not in (0, 1):
        raise ValueError(f"target_parity must be 0 or 1, got {target_parity}")

    img = img_arr.astype(np.float32)
    h, w = img.shape
    y_idx, x_idx = np.indices((h, w))
    target_mask = (y_idx + x_idx) % 2 == target_parity
    value_sum = np.zeros_like(img)
    squared_sum = np.zeros_like(img)
    neighbour_count = np.zeros_like(img)
    offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 0), (-1, 2),
        (0, -1), (0, 1), (1, -2), (1, 0), (1, 2),
        (2, -1), (2, 1)
    ]

    for dy, dx in offsets:
        target_y_start, target_y_end = max(0, -dy), min(h, h - dy)
        target_x_start, target_x_end = max(0, -dx), min(w, w - dx)
        source_y_start, source_y_end = target_y_start + dy, target_y_end + dy
        source_x_start, source_x_end = target_x_start + dx, target_x_end + dx

        target_slice = np.s_[target_y_start:target_y_end, target_x_start:target_x_end]
        source_values = img[source_y_start:source_y_end, source_x_start:source_x_end]
        target_values = target_mask[target_slice]
        value_sum[target_slice] += source_values * target_values
        squared_sum[target_slice] += source_values ** 2 * target_values
        neighbour_count[target_slice] += target_values

    complexity = np.full(img.shape, np.nan, dtype=np.float32)
    valid_target = target_mask & (neighbour_count > 0)
    mean = np.zeros_like(img)
    mean[valid_target] = value_sum[valid_target] / neighbour_count[valid_target]
    variance = np.zeros_like(img)
    variance[valid_target] = (
        squared_sum[valid_target] / neighbour_count[valid_target]
    ) - mean[valid_target] ** 2
    complexity[valid_target] = np.sqrt(np.maximum(variance[valid_target], 0))
    return complexity

# ============================================================
# 2. 單軸 1D T=1 規則 (論文 Eq.2, 對 T=1 的特例)
# ============================================================
def f1d_embed(e, b):
    """e 必須 in {-1,0}，b 是要嵌入的bit"""
    if e == 0:
        return b
    else:  # e == -1
        return -1 - b

def f1d_shift(e):
    """e 必須在 {-1,0} 之外，純位移不嵌入"""
    if e > 0:
        return e + 1
    else:  # e < -1
        return e - 1

def f1d_extract(e_new):
    """回傳 (原始e, bit或None)"""
    if e_new == 0:
        return 0, 0
    elif e_new == 1:
        return 0, 1
    elif e_new == -1:
        return -1, 0
    elif e_new == -2:
        return -1, 1
    elif e_new > 1:
        return e_new - 1, None
    else:  # e_new < -2
        return e_new + 1, None


# ============================================================
# 3. Pairwise PEE mapping (論文 Fig.3, T=1)
# ============================================================
IN_RANGE = (-1, 0)

# 四個 anchor 對應被捨棄的角落 (B類型bin的來源)
ANCHOR_TO_CORNER = {
    (0, 0):   (1, 1),
    (0, -1):  (1, -2),
    (-1, 0):  (-2, 1),
    (-1, -1): (-2, -2),
}
CORNER_TO_ANCHOR = {v: k for k, v in ANCHOR_TO_CORNER.items()}

# 每個 anchor 保留的3種 (b1,b2) 組合 (捨棄 b1=1,b2=1)
SYMBOL_TO_BITS = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
BITS_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_BITS.items()}


def classify_pair(h1, h2):
    if (h1, h2) in ANCHOR_TO_CORNER:
        return 'A'
    if (h1, h2) in CORNER_TO_ANCHOR:
        return 'B'
    in1 = h1 in IN_RANGE
    in2 = h2 in IN_RANGE
    if in1 != in2:
        return 'C'
    return 'D'  # in1==in2==False (in1==in2==True 已經被 A 涵蓋)


def embed_pair(h1, h2, kind, value):
    """value: A->symbol(0,1,2); B/C->bit(0,1); D->None"""
    if kind == 'A':
        b1, b2 = SYMBOL_TO_BITS[value]
        return f1d_embed(h1, b1), f1d_embed(h2, b2)
    if kind == 'B':
        if value == 0:
            return h1, h2
        else:
            return f1d_shift(h1), f1d_shift(h2)
    if kind == 'C':
        if h1 in IN_RANGE:
            return f1d_embed(h1, value), f1d_shift(h2)
        else:
            return f1d_shift(h1), f1d_embed(h2, value)
    # D
    return f1d_shift(h1), f1d_shift(h2)


# B 類型位移後的目標座標 (用來做 extract 反查)
B_SHIFTED = {
    (1, 1):   (2, 2),
    (1, -2):  (2, -3),
    (-2, 1):  (-3, 2),
    (-2, -2): (-3, -3),
}
SHIFTED_TO_B_SOURCE = {v: k for k, v in B_SHIFTED.items()}

# A 類型輸出座標 -> (anchor, symbol) 反查表
A_OUTPUT_LOOKUP = {}
for anchor in ANCHOR_TO_CORNER:
    h1, h2 = anchor
    for symbol, (b1, b2) in SYMBOL_TO_BITS.items():
        out = (f1d_embed(h1, b1), f1d_embed(h2, b2))
        A_OUTPUT_LOOKUP[out] = (anchor, symbol)


def extract_pair(nh1, nh2):
    """回傳 (原始h1, 原始h2, kind, value或None)"""
    if (nh1, nh2) in A_OUTPUT_LOOKUP:
        anchor, symbol = A_OUTPUT_LOOKUP[(nh1, nh2)]
        return anchor[0], anchor[1], 'A', symbol
    if (nh1, nh2) in CORNER_TO_ANCHOR:  # B, bit=0 (沒動)
        return nh1, nh2, 'B', 0
    if (nh1, nh2) in SHIFTED_TO_B_SOURCE:  # B, bit=1 (shift過)
        src = SHIFTED_TO_B_SOURCE[(nh1, nh2)]
        return src[0], src[1], 'B', 1
    # 剩下的走 per-axis 獨立反推 (C 或 D)
    o1, bit1 = f1d_extract(nh1)
    o2, bit2 = f1d_extract(nh2)
    if bit1 is not None:
        return o1, o2, 'C', bit1
    if bit2 is not None:
        return o1, o2, 'C', bit2
    return o1, o2, 'D', None


# ============================================================
# 4. 大數進制轉換 (log2(3) 的精確容量分配)
# ============================================================
def bits_to_int(bits):
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x

def int_to_bits(x, n):
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]

def int_to_base3(x, n_digits):
    digits = []
    for _ in range(n_digits):
        digits.append(x % 3)
        x //= 3
    digits.reverse()
    return digits

def base3_to_int(digits):
    x = 0
    for d in digits:
        x = x * 3 + d
    return x

def ternary_capacity_bits(n_A):
    """floor(log2(3^n_A))"""
    if n_A == 0:
        return 0
    return (3 ** n_A).bit_length() - 1


# ============================================================
# 5. valid pixel 取得 / pairing (跟之前相同)
# ============================================================
def get_valid_errors_in_order(img_arr, rounded_pred, target_parity):
    h, w = img_arr.shape
    error_map = img_arr.astype(np.int32) - rounded_pred
    y_idx, x_idx = np.indices((h, w))
    valid_mask = (y_idx + x_idx) % 2 == target_parity
    coords = np.argwhere(valid_mask)
    errors = error_map[valid_mask]
    return coords, errors, valid_mask


def build_pairs(coords, errors):
    n = len(errors)
    n_pairs = n // 2
    leftover = None
    if n % 2 == 1:
        leftover = (coords[-1], errors[-1])
        errors = errors[:-1]
        coords = coords[:-1]
    pair_errors = errors.reshape(n_pairs, 2)
    pair_coords = coords.reshape(n_pairs, 2, 2)
    return pair_errors, pair_coords, leftover


def get_pair_priority_order(pair_coords, comp_matrix):
    """Return pair indices from low to high complexity with a stable tie-break."""
    if len(pair_coords) == 0:
        return np.array([], dtype=int)

    pair_complexity = np.array([
        (comp_matrix[c1[0], c1[1]] + comp_matrix[c2[0], c2[1]]) / 2.0
        for c1, c2 in pair_coords
    ])
    return np.argsort(pair_complexity, kind='stable')


# ============================================================
# 6. Stage-level embed / extract
# ============================================================
def embed_stage_2dpee(img_arr, rounded_pred, target_parity, payload, comp_matrix,
                       bit_ptr_start=0, target_ec=None):
    """
    comp_matrix: target parity用的局部複雜度 (get_cross_complexity_matrix算出來的)，
    數值越小代表越平滑、預測越準。pair的複雜度取兩個pixel複雜度的平均，
    embedding依複雜度由小到大優先處理，達到target_ec就停止，
    複雜度高的區域完全不碰(不shift、不嵌入)。
    """
    coords, errors, _ = get_valid_errors_in_order(img_arr, rounded_pred, target_parity)
    pair_errors, pair_coords, leftover = build_pairs(coords, errors)

    kinds_all = [classify_pair(int(h1), int(h2)) for h1, h2 in pair_errors]

    # 依複雜度排序，決定embedding優先順序 (平滑的pair優先)
    n_pairs = len(pair_errors)
    priority_order = get_pair_priority_order(pair_coords, comp_matrix)

    if target_ec is None:
        needed = float('inf')
    else:
        needed = max(0, target_ec - bit_ptr_start)

    # Pass 1: 依複雜度優先順序累計capacity，找出達到needed所需的pair數 stop_rank
    n_A_running = 0
    n_BC_running = 0
    stop_rank = 0 if needed == 0 else n_pairs
    if needed > 0:
        for rank, idx in enumerate(priority_order):
            k = kinds_all[idx]
            if k == 'A':
                n_A_running += 1
            elif k in ('B', 'C'):
                n_BC_running += 1
            cap_so_far = n_BC_running + ternary_capacity_bits(n_A_running)
            if cap_so_far >= needed:
                stop_rank = rank + 1
                break

    selected_indices = priority_order[:stop_rank]
    used_kinds = [kinds_all[i] for i in selected_indices]
    n_A = used_kinds.count('A')
    n_BC = used_kinds.count('B') + used_kinds.count('C')
    cap_A = ternary_capacity_bits(n_A)
    total_cap = n_BC + cap_A

    avail = payload[bit_ptr_start: bit_ptr_start + total_cap]
    used = len(avail)
    avail = list(avail) + [0] * (total_cap - used)  # payload不夠長才補0(不算進used)

    bc_bits = avail[:n_BC]
    a_bits = avail[n_BC:n_BC + cap_A]
    a_int = bits_to_int(a_bits) if cap_A > 0 else 0
    a_symbols = int_to_base3(a_int, n_A) if n_A > 0 else []

    res = img_arr.copy().astype(np.int32)
    idx_bc, idx_a = 0, 0
    # 依複雜度優先順序embedding，selected_indices以外的pair完全不碰
    for pidx in selected_indices:
        h1, h2 = int(pair_errors[pidx][0]), int(pair_errors[pidx][1])
        c1, c2 = pair_coords[pidx]
        kind = kinds_all[pidx]
        if kind == 'A':
            val = a_symbols[idx_a]; idx_a += 1
        elif kind in ('B', 'C'):
            val = bc_bits[idx_bc]; idx_bc += 1
        else:
            val = None
        nh1, nh2 = embed_pair(h1, h2, kind, val)
        y1, x1 = c1; y2, x2 = c2
        res[y1, x1] = np.clip(rounded_pred[y1, x1] + nh1, 0, 255)
        res[y2, x2] = np.clip(rounded_pred[y2, x2] + nh2, 0, 255)

    new_ptr = bit_ptr_start + used

    # leftover只有在整批pair都處理完(stop_rank==n_pairs)時才會被摸到
    if leftover is not None and stop_rank == n_pairs:
        (ly, lx), le = leftover
        le = int(le)
        if le in IN_RANGE and new_ptr < len(payload):
            b = payload[new_ptr]
            new_e = f1d_embed(le, b)
            new_ptr += 1
        elif le in IN_RANGE:
            new_e = f1d_embed(le, 0)
        else:
            new_e = f1d_shift(le)
        res[ly, lx] = np.clip(rounded_pred[ly, lx] + new_e, 0, 255)

    return Image.fromarray(res.astype(np.uint8)), new_ptr, stop_rank


def extract_stage_2dpee(stego_arr, rounded_pred, target_parity, comp_matrix, stop_rank):
    """
    Reverse only the pairs selected during embedding.

    The decoder must receive the same complexity matrix used by the corresponding
    embedding stage and the stage's stop_rank.  Pairs after stop_rank were not
    modified during embedding, so they must remain untouched here.
    """
    coords, stego_errors, _ = get_valid_errors_in_order(
        stego_arr, rounded_pred, target_parity)
    pair_new, pair_coords, leftover = build_pairs(coords, stego_errors)

    n_pairs = len(pair_new)
    if not 0 <= stop_rank <= n_pairs:
        raise ValueError(
            f"stop_rank must be between 0 and {n_pairs}, got {stop_rank}")

    priority_order = get_pair_priority_order(pair_coords, comp_matrix)
    selected_indices = priority_order[:stop_rank]

    recon = stego_arr.copy().astype(np.int32)
    a_symbols, bc_bits = [], []
    n_A = 0

    for pidx in selected_indices:
        nh1, nh2 = pair_new[pidx]
        c1, c2 = pair_coords[pidx]
        h1, h2, kind, val = extract_pair(int(nh1), int(nh2))
        y1, x1 = c1; y2, x2 = c2
        recon[y1, x1] = np.clip(rounded_pred[y1, x1] + h1, 0, 255)
        recon[y2, x2] = np.clip(rounded_pred[y2, x2] + h2, 0, 255)
        if kind == 'A':
            a_symbols.append(val); n_A += 1
        elif kind in ('B', 'C'):
            bc_bits.append(val)

    cap_A = ternary_capacity_bits(n_A)
    a_int = base3_to_int(a_symbols) if a_symbols else 0
    a_bits = int_to_bits(a_int, cap_A) if cap_A > 0 else []

    extracted_bits = bc_bits + a_bits

    # The leftover pixel is changed only when the embedder processes every pair.
    if leftover is not None and stop_rank == n_pairs:
        (ly, lx), le_new = leftover
        e, bit = f1d_extract(int(le_new))
        recon[ly, lx] = np.clip(rounded_pred[ly, lx] + e, 0, 255)
        if bit is not None:
            extracted_bits = extracted_bits + [bit]

    return Image.fromarray(recon.astype(np.uint8)), extracted_bits


# ============================================================
# 7. 兩階段 (checkerboard) 整合
# ============================================================
def predict_target_parity(model, img_arr, target_parity, device):
    """Predict target_parity pixels using only the opposite checkerboard set."""
    to_tensor = transforms.ToTensor()
    h, w = img_arr.shape
    y_idx, x_idx = np.indices((h, w))

    model_input = img_arr.copy()
    model_input[(y_idx + x_idx) % 2 == target_parity] = 0
    with torch.no_grad():
        prediction_raw = model(
            to_tensor(Image.fromarray(model_input)).unsqueeze(0).to(device)
        )[0, 0] * 255
    return np.round(
        torch.clamp(prediction_raw, 0, 255).cpu().numpy()
    ).astype(np.int32)


def embed_two_stage_2dpee(model, img_arr, full_payload, device, target_ec):
    """
    Embed payload in checkerboard order with decoder-reproducible complexity.

    Stage 1 embeds parity-1 pixels using complexity from original parity-0
    pixels.  Stage 2 embeds parity-0 pixels using complexity from the parity-1
    stego pixels produced by Stage 1.
    """
    stage1_target = 1
    stage2_target = 0

    prediction1 = predict_target_parity(model, img_arr, stage1_target, device)
    complexity1 = get_cross_complexity_matrix(img_arr, stage1_target)
    stego1, used1, stop_rank1 = embed_stage_2dpee(
        img_arr, prediction1, stage1_target, full_payload, complexity1,
        bit_ptr_start=0, target_ec=target_ec)
    stego1_arr = np.array(stego1)

    prediction2 = predict_target_parity(model, stego1_arr, stage2_target, device)
    complexity2 = get_cross_complexity_matrix(stego1_arr, stage2_target)
    stego2, used_total, stop_rank2 = embed_stage_2dpee(
        stego1_arr, prediction2, stage2_target, full_payload, complexity2,
        bit_ptr_start=used1, target_ec=target_ec)

    embedding_info = {
        "payload_length": used_total,
        "stage1_payload_length": used1,
        "stage1_stop_rank": stop_rank1,
        "stage2_stop_rank": stop_rank2,
    }
    return stego2, embedding_info


def extract_two_stage_2dpee(model, stego_arr, device, stage1_stop_rank,
                             stage2_stop_rank, payload_length=None):
    """
    Recover the payload and cover image in the reverse checkerboard order.

    Stage 2 is decoded first.  Its complexity is computed from the parity-1
    stego pixels, which are unchanged by Stage 2 and match the encoder input.
    After restoring Stage 2, Stage 1 is decoded using the restored parity-0
    pixels.  This reproduces both complexity orders without the cover image.
    """
    stage1_target = 1
    stage2_target = 0
    stego_arr = np.asarray(stego_arr)

    prediction2 = predict_target_parity(model, stego_arr, stage2_target, device)
    complexity2 = get_cross_complexity_matrix(stego_arr, stage2_target)
    stego1, stage2_bits = extract_stage_2dpee(
        stego_arr, prediction2, stage2_target, complexity2, stage2_stop_rank)
    stego1_arr = np.array(stego1)

    prediction1 = predict_target_parity(model, stego1_arr, stage1_target, device)
    complexity1 = get_cross_complexity_matrix(stego1_arr, stage1_target)
    recovered, stage1_bits = extract_stage_2dpee(
        stego1_arr, prediction1, stage1_target, complexity1, stage1_stop_rank)

    extracted_payload = stage1_bits + stage2_bits
    if payload_length is not None:
        if payload_length < 0:
            raise ValueError(f"payload_length must be non-negative, got {payload_length}")
        if payload_length > len(extracted_payload):
            raise ValueError(
                "payload_length exceeds the number of extracted bits: "
                f"{payload_length} > {len(extracted_payload)}")
        extracted_payload = extracted_payload[:payload_length]

    return recovered, extracted_payload


def try_embedding_2d(model, img_arr, full_payload, device, target_ec):
    stego2, embedding_info = embed_two_stage_2dpee(
        model, img_arr, full_payload, device, target_ec)

    stego_arr = np.array(stego2)
    mse = np.mean((img_arr.astype(np.float32) - stego_arr.astype(np.float32)) ** 2)
    psnr_val = 10 * log10(255 ** 2 / mse) if mse > 0 else 100
    ssim_val = ssim(img_arr, stego_arr, data_range=255)

    return embedding_info["payload_length"], psnr_val, ssim_val, stego2


# ============================================================
# 8. 主流程 -- EC = 10000 / 20000
# ============================================================
def main():
    device = pick_device()
    img_dir = "images"
    model_path = "ncnnp_imagenette.pth"
    output_csv = "output_results_2dpee_paper_mapping.csv"
    EC_LIST = [10000, 20000]

    model = NCNNP().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    if not os.path.exists(img_dir):
        print(f"[Warning] 找不到資料夾: {img_dir}")
        return

    for file_name in sorted(os.listdir(img_dir)):
        if not file_name.lower().endswith(".bmp"):
            continue
        img_arr = np.array(Image.open(os.path.join(img_dir, file_name)).convert("L"))
        for ec in EC_LIST:
            payload = [np.random.randint(0, 2) for _ in range(ec)]

            print(f"\n[Processing] {file_name} | EC target: {ec}")
            used, psnr, ssim_val, stego = try_embedding_2d(
                model, img_arr, payload, device, target_ec=ec)

            status = "SUCCESS" if used >= ec else "PARTIAL"
            print(f"  Used: {used}/{ec} bits | PSNR: {psnr:.2f} | SSIM: {ssim_val:.4f} | {status}")

            results.append({
                "Image": file_name, "EC_target": ec, "Used_bits": used,
                "PSNR": psnr, "SSIM": ssim_val, "Status": status
            })
            stego.save(f"stego_2dpee_paper_EC{ec}_{file_name}")

    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

            for ec in EC_LIST:
                subset = [r for r in results if r["EC_target"] == ec]
                if subset:
                    avg_psnr = np.mean([r["PSNR"] for r in subset])
                    avg_ssim = np.mean([r["SSIM"] for r in subset])
                    writer.writerow({
                        "Image": f"AVERAGE_EC{ec}", "EC_target": ec, "Used_bits": "",
                        "PSNR": f"{avg_psnr:.2f}", "SSIM": f"{avg_ssim:.4f}", "Status": ""
                    })
                    print(f"\n[EC={ec}] Average PSNR: {avg_psnr:.2f} | Average SSIM: {avg_ssim:.4f}")

        print(f"\n[Done] Saved to {output_csv}")


if __name__ == "__main__":
    main()
