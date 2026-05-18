import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from backbone.DFormer.DFormer import DFormer_Small

class RegionPartitioner(nn.Module):
    """图像区域分割模块（替换原PatchDivider，支持replicate边缘填充）"""

    def __init__(self, padding_mode='replicate'):
        super().__init__()
        assert padding_mode in ['constant', 'replicate', 'reflect'], \
            f"Padding mode must be one of ['constant', 'replicate', 'reflect'], got {padding_mode}"
        self.padding_mode = padding_mode

    def forward(self, x, step, region_size):
        b, c, h, w = x.size()
        # 若输入本身就是region_size尺寸，步长设为region_size（无重叠）
        step = region_size if (h == region_size and w == region_size) else step

        # 计算填充量（确保填充后尺寸能被region_size整除）
        pad_h = (region_size - h % region_size) % region_size
        pad_w = (region_size - w % region_size) % region_size

        # 边缘填充
        if self.padding_mode != 'truncate' and (pad_h > 0 or pad_w > 0):
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.padding_mode)
            h, w = x.shape[2], x.shape[3]

        # 生成区域块
        regions = []
        positions = []  # 保存每个区域块的位置信息
        for i in range(0, h, step):
            for j in range(0, w, step):
                end_i = min(i + region_size, h)
                end_j = min(j + region_size, w)
                # 跳过不足region_size的块（避免尺寸不匹配）
                if self.padding_mode != 'truncate' and (end_i - i < region_size or end_j - j < region_size):
                    continue
                # 如果区域块小于region_size，进行填充
                region = x[:, :, i:end_i, j:end_j]
                rh, rw = region.shape[2], region.shape[3]
                if rh < region_size or rw < region_size:
                    pad_rh = region_size - rh
                    pad_rw = region_size - rw
                    region = F.pad(region, (0, pad_rw, 0, pad_rh), mode=self.padding_mode)
                regions.append(region)
                positions.append((i, end_i, j, end_j, rh, rw))  # 保存原始位置和尺寸

        # 计算区域块的行列数
        row_count = (h + step - 1) // step
        col_count = len(regions) // row_count if row_count != 0 else 0

        # 调整维度：(b, n_regions, c, region_h, region_w)
        regions = torch.stack(regions, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return regions, row_count, col_count, (pad_h, pad_w), positions


class RegionIntegrator(nn.Module):
    """图像区域整合模块（配套RegionPartitioner使用，替换原PatchMerger）"""

    def __init__(self, padding_mode='replicate'):
        super().__init__()
        self.padding_mode = padding_mode

    def forward(self, regions, orig_x, step, region_size, pad_info, positions):
        pad_h, pad_w = pad_info
        b, c, orig_h, orig_w = orig_x.size()
        temp_h = orig_h + pad_h  # 填充后的高度
        temp_w = orig_w + pad_w  # 填充后的宽度

        # 初始化输出和计数掩码（处理重叠区域块的平均融合）
        output = torch.zeros(b, c, temp_h, temp_w, device=orig_x.device)
        count_mask = torch.zeros_like(output)

        # 整合区域块 - 使用保存的位置信息
        for idx in range(regions.shape[1]):  # 遍历所有区域块
            i, end_i, j, end_j, rh, rw = positions[idx]
            # 从区域块中提取原始大小的部分（去除填充）
            region = regions[:, idx, :, :rh, :rw]
            # 将区域块放到正确位置
            output[:, :, i:end_i, j:end_j] += region
            count_mask[:, :, i:end_i, j:end_j] += 1

        # 重叠区域平均化（避免重复累加导致数值偏大）
        output = output / count_mask.clamp(min=1e-8)  # 防止除零

        # 移除填充部分，恢复原始尺寸
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :orig_h, :orig_w]

        return output


class NormalizationWrapper(nn.Module):
    def __init__(self, dim, func, norm_type='layer'):
        super().__init__()
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(dim)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}, choose from ['layer', 'instance', 'batch']")
        self.func = func  # 待包装的函数（如注意力、MLP）

    def forward(self, x, *args, **kwargs):
        # 先归一化，再执行目标函数，传递所有额外参数
        return self.func(self.norm(x), *args, **kwargs)


class AdaptiveAttention(nn.Module):
    """纯自适应多头注意力模块（无分块逻辑）"""

    def __init__(self, dim, heads, qk_dim=None, attn_drop=0.):
        super().__init__()
        self.heads = heads
        self.dim = dim  # 保存输入特征维度

        # 自动确保qk_dim能被头数整除（避免einops拆分错误）
        if qk_dim is not None:
            # 确保qk_dim能被heads整除
            self.qk_dim = ((qk_dim + heads - 1) // heads) * heads
        else:
            # 从dim计算，并确保能被heads整除
            self.qk_dim = ((dim // heads) * heads + heads) if (dim % heads != 0) else (dim // heads) * heads

        assert self.qk_dim % heads == 0, f"qk_dim ({self.qk_dim}) must be divisible by heads ({heads})"
        self.head_dim = self.qk_dim // self.heads  # 每个头的维度

        self.scale = self.head_dim ** -0.5  # 使用每个头的维度计算缩放因子
        # 调整投影层输出维度以匹配修正后的qk_dim
        self.qkv_proj = nn.Linear(dim, self.qk_dim + self.qk_dim + dim, bias=False)  # QKV投影

        # 确保输出投影层的输入维度与合并后的多头维度一致
        self.out_proj = nn.Linear(self.qk_dim, dim)  # 输出投影
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力 dropout

    def forward(self, x, mask=None):
        # x: (b, n, dim)，n为序列长度（如区域块内像素数）
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # 拆分Q/K/V: 各(b, n, qk_dim)或(b, n, dim)

        def adjust_and_rearrange(t):
            batch, seq_len, dim = t.shape
            # 如果维度不匹配，进行调整
            if dim != self.qk_dim:
                # 使用线性投影调整维度
                proj = nn.Linear(dim, self.qk_dim, bias=False).to(t.device)
                t = proj(t)
            # 执行rearrange操作
            return rearrange(t, 'b n (h d) -> b h n d', h=self.heads, d=self.head_dim)

        q, k, v = map(adjust_and_rearrange, qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (b, heads, n, n)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # 掩码无效区域
        attn = self.attn_drop(attn.softmax(dim=-1))  # 注意力归一化+dropout

        # 注意力加权求和 + 输出投影
        out = attn @ v  # (b, heads, n, d)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 合并多头维度，得到(b, n, qk_dim)
        out = self.out_proj(out)  # (b, n, dim) - 维度匹配
        return out


class ConvMLP(nn.Module):
    """卷积MLP模块（带深度卷积的特征变换，替换原ConvolutionMLP）"""

    def __init__(self, dim, mlp_ratio=2., kernel_size=5, dilation=1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)  # MLP隐藏层维度

        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation // 2,  # 保证输入输出尺寸一致
            dilation=dilation, groups=hidden_dim
        )
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        # 线性投影层
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, spatial_size):
        # x: (b, n, dim)，spatial_size: (h, w)（对应n=h*w）
        x = self.proj[0](x)  # 线性升维：(b, n, hidden_dim)
        # 重塑为2D特征图：(b, hidden_dim, h, w)
        x = x.transpose(1, 2).view(x.shape[0], -1, spatial_size[0], spatial_size[1])
        # 深度卷积 + 归一化 + 激活
        x = self.act(self.norm(self.dwconv(x)))
        # 展平回序列维度：(b, n, hidden_dim)
        x = x.flatten(2).transpose(1, 2)
        # 线性降维 + 激活
        x = self.proj[1](x)
        x = self.proj[2](x)  # (b, n, dim)
        return x


class AdaptiveFocusAttention(nn.Module):

    def __init__(self, dim, heads, qk_dim=None, mlp_ratio=2., attn_drop=0., padding_mode='replicate'):
        super().__init__()
        # 区域分割与整合模块
        self.region_partitioner = RegionPartitioner(padding_mode=padding_mode)
        self.region_integrator = RegionIntegrator(padding_mode=padding_mode)
        # 注意力块（带LayerNorm包装）
        self.attn_block = NormalizationWrapper(
            dim, AdaptiveAttention(dim, heads, qk_dim, attn_drop), norm_type='layer'
        )
        # MLP块（带LayerNorm包装）
        self.mlp_block = NormalizationWrapper(
            dim, ConvMLP(dim, mlp_ratio), norm_type='layer'
        )

    def forward(self, x, region_size):
        # x: (b, c, h, w)，region_size: 区域块尺寸
        step = region_size - 2  # 重叠区域步长（重叠2个像素）
        # 1. 图像区域分割 - 获取位置信息
        regions, row_count, col_count, pad_info, positions = self.region_partitioner(x, step, region_size)
        b, n_regions, c, rh, rw = regions.shape  # rh/rw: 单区域块高度/宽度

        # 2. 注意力计算（展平区域块内空间维度）
        region_flat = rearrange(regions, 'b n c h w -> (b n) (h w) c')  # (b*n, h*w, c)
        attn_out = self.attn_block(region_flat) + region_flat  # 残差连接

        # 3. 区域块重组为特征图 - 使用位置信息
        region_reshaped = rearrange(attn_out, '(b n) (h w) c -> b n c h w', n=n_regions, w=rw)
        merged = self.region_integrator(region_reshaped, x, step, region_size, pad_info, positions)  # (b, c, h, w)

        # 4. MLP特征变换
        b, c, h, w = merged.shape
        x_flat = rearrange(merged, 'b c h w -> b (h w) c')  # (b, h*w, c)
        mlp_out = self.mlp_block(x_flat, (h, w)) + x_flat  # 残差连接

        # 5. 恢复原始空间维度
        return rearrange(mlp_out, 'b (h w) c -> b c h w', h=h)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 定义一个一维卷积层，用于进行qkv变换
class qkv_transform(nn.Conv1d):
    """用于qkv变换的Conv1d"""


class AdaptiveSpatialBlock(nn.Module):
    def __init__(self, in_channels, base_gamma=1 / 12, max_dilation=2):
        super(AdaptiveSpatialBlock, self).__init__()
        self.in_channels = in_channels
        self.base_gamma = base_gamma  # 基础gamma系数
        self.max_dilation = max_dilation  # 最大间隔偏移步长

        # 1. 边缘密度估计器（用于动态调整gamma）
        self.edge_estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 初始化Sobel核
        sobel_kernel = self._init_sobel_kernel()
        self.edge_estimator[0].weight.data = sobel_kernel
        self.edge_estimator[0].weight.requires_grad = False

        # 2. 动态gamma映射
        self.gamma_mapper = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # 3. 归一化层
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)

        # 4. 特征增强MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, bias=False)
        )

    def _init_sobel_kernel(self):
        """初始化Sobel核，匹配分组卷积要求"""
        # 创建水平和垂直Sobel核
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

        # 对于分组卷积，每个输入通道需要独立的卷积核
        kernel = torch.zeros(self.in_channels, 1, 3, 3)
        for i in range(self.in_channels):
            if i % 2 == 0:
                kernel[i] = sobel_x.unsqueeze(0)  # 水平核
            else:
                kernel[i] = sobel_y.unsqueeze(0)  # 垂直核
        return kernel

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # 动态计算gamma
        edge_features = self.edge_estimator[0](x)
        edge_magnitude = torch.norm(edge_features, dim=1, keepdim=True)
        complexity = edge_magnitude.mean(dim=[1, 2, 3], keepdim=True)
        gamma_scale = self.gamma_mapper(complexity.view(b, 1)).view(b, 1, 1, 1)
        dynamic_gamma = self.base_gamma * (1 + 2 * gamma_scale)

        # 计算每个方向的偏移通道数
        each_direction_channels = max(1, min(
            (c * dynamic_gamma).mean().int().item(),  # 先求均值并转为整数
            c // 12  # 最大通道数限制
        ))

        # 计算偏移步长
        offset_step = min(self.max_dilation, min(h, w) // 8)

        offset_features = []
        if each_direction_channels > 0:
            # 右偏移
            offset_right = torch.zeros_like(x[:, :each_direction_channels])
            offset_right[:, :, :, offset_step:] = x[:, :each_direction_channels, :, :-offset_step]
            offset_features.append(offset_right)

            # 左偏移
            offset_left = torch.zeros_like(x[:, :each_direction_channels])
            offset_left[:, :, :, :-offset_step] = x[:, :each_direction_channels, :, offset_step:]
            offset_features.append(offset_left)

            # 下偏移
            offset_down = torch.zeros_like(x[:, :each_direction_channels])
            offset_down[:, :, offset_step:, :] = x[:, :each_direction_channels, :-offset_step, :]
            offset_features.append(offset_down)

            # 上偏移
            offset_up = torch.zeros_like(x[:, :each_direction_channels])
            offset_up[:, :, :-offset_step, :] = x[:, :each_direction_channels, offset_step:, :]
            offset_features.append(offset_up)

        # 合并偏移特征和未偏移特征
        num_offset = len(offset_features) * each_direction_channels
        non_offset = x[:, num_offset:]
        x_offset = torch.cat(offset_features + [non_offset], dim=1) if offset_features else x

        # 归一化与特征增强
        x_norm = self.norm(x_offset)
        x_mlp = self.mlp(x_norm)
        output = x_mlp + identity

        return output


class AdaptiveAxisAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=28,
                 stride=1, bias=False, width=False, shift_gamma=1 / 12):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        self.qkv_transform = nn.Conv1d(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        base_size = kernel_size * 2 - 1
        self.register_buffer('base_relative', torch.randn(self.group_planes * 2, base_size, base_size))
        self.relative_scale = nn.Parameter(torch.ones(1))

        self.supported_kernels = {}
        self.max_supported_kernel = kernel_size * 2
        self.min_supported_kernel = 7

        for k in [7, 14, 21, kernel_size]:
            self._init_position_encoding(k)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.spatial_block = AdaptiveSpatialBlock(out_planes, base_gamma=shift_gamma)

        self.kernel_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_planes // 4, 1, 1),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def _init_position_encoding(self, kernel_size):
        if kernel_size in self.supported_kernels:
            return

        adjusted_kernel = max(
            self.min_supported_kernel,
            min(int(round(kernel_size / 2) * 2 + 1), self.max_supported_kernel)
        )

        relative_expanded = self.base_relative.unsqueeze(0)
        relative_resized = F.interpolate(
            relative_expanded,
            size=(adjusted_kernel, adjusted_kernel),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        self.supported_kernels[kernel_size] = {
            'relative': relative_resized,
            'adjusted_kernel': adjusted_kernel,
        }

    def _get_position_encoding(self, kernel_size):
        kernel_size = max(
            self.min_supported_kernel,
            min(int(round(kernel_size / 2) * 2 + 1), self.max_supported_kernel)
        )

        if kernel_size not in self.supported_kernels:
            self._init_position_encoding(kernel_size)

        return self.supported_kernels[kernel_size]

    def forward(self, x):
        batch_size = x.size(0)
        adjust_factor = self.kernel_adjust(x)
        effective_kernels = torch.clamp(
            (self.kernel_size * adjust_factor).int().squeeze(-1).squeeze(-1),
            min=self.min_supported_kernel,
            max=self.kernel_size
        )

        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)

        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        output_batches = []

        for b in range(batch_size):
            effective_kernel = effective_kernels[b, 0].item()
            batch_start = b * W
            batch_end = (b + 1) * W
            batch_q = q[batch_start:batch_end]
            batch_k = k[batch_start:batch_end]
            batch_v = v[batch_start:batch_end]

            pos_encoding = self._get_position_encoding(effective_kernel)
            adjusted_kernel = pos_encoding['adjusted_kernel']
            pos_encoding_tensor = pos_encoding['relative'].to(batch_q.device)

            q_embedding, k_embedding, v_embedding = torch.split(
                pos_encoding_tensor,
                [self.group_planes // 2, self.group_planes // 2, self.group_planes],
                dim=0
            )

            if adjusted_kernel != H:
                batch_q_reshaped = batch_q.reshape(-1, batch_q.size(-2), batch_q.size(-1))
                batch_k_reshaped = batch_k.reshape(-1, batch_k.size(-2), batch_k.size(-1))
                batch_v_reshaped = batch_v.reshape(-1, batch_v.size(-2), batch_v.size(-1))

                batch_q_resized = F.interpolate(batch_q_reshaped, size=adjusted_kernel, mode='linear',
                                                align_corners=False)
                batch_k_resized = F.interpolate(batch_k_reshaped, size=adjusted_kernel, mode='linear',
                                                align_corners=False)
                batch_v_resized = F.interpolate(batch_v_reshaped, size=adjusted_kernel, mode='linear',
                                                align_corners=False)

                batch_q = batch_q_resized.reshape(batch_q.size(0), batch_q.size(1), batch_q_resized.size(-2),
                                                  batch_q_resized.size(-1))
                batch_k = batch_k_resized.reshape(batch_k.size(0), batch_k.size(1), batch_k_resized.size(-2),
                                                  batch_k_resized.size(-1))
                batch_v = batch_v_resized.reshape(batch_v.size(0), batch_v.size(1), batch_v_resized.size(-2),
                                                  batch_v_resized.size(-1))

            qr = torch.einsum('bgci,cij->bgij', batch_q, q_embedding)
            kr = torch.einsum('bgci,cij->bgij', batch_k, k_embedding).transpose(2, 3)
            qk = torch.einsum('bgci, bgcj->bgij', batch_q, batch_k)

            stacked_similarity = torch.cat([qk, qr, kr], dim=1)
            stacked_similarity = self.bn_similarity(stacked_similarity).view(
                W, 3, self.groups, adjusted_kernel, adjusted_kernel).sum(dim=1)

            if adjusted_kernel != H:
                stacked_similarity_reshaped = stacked_similarity.reshape(-1, 1, adjusted_kernel, adjusted_kernel)
                stacked_similarity_resized = F.interpolate(
                    stacked_similarity_reshaped, size=H, mode='bilinear', align_corners=False)
                stacked_similarity = stacked_similarity_resized.reshape(stacked_similarity.size(0),
                                                                        stacked_similarity.size(1),
                                                                        stacked_similarity_resized.size(-2),
                                                                        stacked_similarity_resized.size(-1))

            similarity = F.softmax(stacked_similarity, dim=3)

            if v_embedding.size(1) != similarity.size(2) or v_embedding.size(2) != similarity.size(3):
                v_embedding = F.interpolate(
                    v_embedding.unsqueeze(0),
                    size=(similarity.size(2), similarity.size(3)),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            if adjusted_kernel != H:
                batch_v_reshaped = batch_v.reshape(-1, batch_v.size(-2), batch_v.size(-1))
                batch_v_resized = F.interpolate(batch_v_reshaped, size=H, mode='linear', align_corners=False)
                batch_v = batch_v_resized.reshape(batch_v.size(0), batch_v.size(1), batch_v_resized.size(-2),
                                                  batch_v_resized.size(-1))

            sv = torch.einsum('bgij,bgcj->bgci', similarity, batch_v)
            sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
            batch_output = torch.cat([sv, sve], dim=-1).view(W, self.out_planes * 2, H)
            output_batches.append(batch_output)

        output = torch.cat(output_batches, dim=0)
        output = self.bn_output(output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)
        output = self.spatial_block(output)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.base_relative, 0., math.sqrt(1. / self.group_planes))


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1) -> object:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PPD(nn.Module):
    def __init__(self, channel):
        super(PPD, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 128, 1)

        # 动态融合权重
        self.fusion_weight1 = nn.Parameter(torch.tensor(0.5))
        self.fusion_weight2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.fusion_weight1 * self.conv_upsample1(self.upsample(x1)) + (1 - self.fusion_weight1) * x2

        x3_1 = self.fusion_weight2 * self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) + (1 - self.fusion_weight2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channel):
        super(Decoder, self).__init__()
        self.conv1 = BasicConv2d(128, in_channel, 3, padding=1)
        self.conv2 = BasicConv2d(256, 128, 3, padding=1)
        self.conv3 = BasicConv2d(512, 256, 3, padding=1)
        self.upsample = lambda x, target: F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)

        # 注意力权重
        self.attention_gate1 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_gate2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_gate3 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x0, x1, x2, x3, GM_MA):
        # 计算注意力权重
        attn3 = self.attention_gate3(x3)
        attn2 = self.attention_gate2(x2)
        attn1 = self.attention_gate1(x1)

        out_diff_3 = self.upsample(GM_MA, x3).expand_as(x3)
        x2 = self.upsample(self.conv3(x3 * (1 + out_diff_3) * attn3), x2) * x2 * attn2
        out_diff_2 = self.upsample(GM_MA, x2).expand_as(x2)
        x1 = self.upsample(self.conv2(x2 * (1 + out_diff_2) * attn2), x1) * x1 * attn1
        out_diff_1 = self.upsample(GM_MA, x1).expand_as(x1)
        x0 = self.upsample(self.conv1(x1 * (1 + out_diff_1) * attn1), x0) * x0
        return x0


class ChannelNorm2D(nn.Module):
    """2D通道归一化层（替换原LayerNorm2D）"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(ChannelNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized


class ChannelNorm1D(nn.Module):
    """1D通道归一化层（替换原LayerNorm1D）"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(ChannelNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, L)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, L)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, L)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized


class Conv2DLayer(nn.Module):
    """2D卷积层（替换原ConvLayer2D）"""
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(Conv2DLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class Conv1DLayer(nn.Module):
    """1D卷积层（替换原ConvLayer1D）"""
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(Conv1DLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class NeuroAttention(nn.Module):
    def __init__(self, in_channels, expand_ratio=1, state_size=None):
        super().__init__()
        # 多尺度特征提取：并行分支（3个不同 kernel 尺寸）
        self.scale1 = Conv2DLayer(in_channels, in_channels // 4, kernel_size=3, padding=1)  # 3x3 分支
        self.scale2 = Conv2DLayer(in_channels, in_channels // 4, kernel_size=5, padding=2)  # 5x5 分支
        self.scale3 = Conv2DLayer(in_channels, in_channels // 2, kernel_size=7, padding=3)  # 7x7 分支

        self.expand_ratio = expand_ratio
        self.hidden_dim = int(self.expand_ratio * in_channels)
        self.state_size = state_size if state_size is not None else self.hidden_dim

        # 计算多尺度特征的总通道数（用于后续投影层输入）
        multi_scale_channels = (in_channels // 4) + (in_channels // 4) + (in_channels // 2)  # 应为 in_channels
        # State transformation projections：输入是原始特征+多尺度特征
        self.state_proj = Conv1DLayer(in_channels + multi_scale_channels, 3 * self.state_size, 1, norm=None, act_layer=None)

        # Gating mechanism
        self.feature_gate = nn.Sequential(
            Conv1DLayer(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

        conv_dim = 3 * self.state_size
        self.depth_conv = Conv2DLayer(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None)

        # Feature transformation：输入是原始特征+多尺度特征
        self.feature_proj = Conv1DLayer(in_channels + multi_scale_channels, 2 * self.hidden_dim, 1, norm=None,
                                        act_layer=None)

        self.output_proj = Conv1DLayer(self.hidden_dim, in_channels, 1, norm=None, act_layer=None)

        # State parameters
        decay = torch.empty(self.state_size).uniform_(0.5, 2)
        self.decay_param = torch.nn.Parameter(decay)
        self.activation = nn.SiLU()
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = 1e-5

    def forward(self, x, hidden_state=None):
        batch, channels, height, width = x.shape
        seq_len = height * width

        # 多尺度特征提取：并行计算三个分支，再拼接
        scale1_feat = self.scale1(x)  # (B, in//4, H, W)
        scale2_feat = self.scale2(x)  # (B, in//4, H, W)
        scale3_feat = self.scale3(x)  # (B, in//2, H, W)
        multi_scale = torch.cat([scale1_feat, scale2_feat, scale3_feat], dim=1)  # 拼接多尺度特征

        # 拼接原始特征和多尺度特征（用于后续投影）
        x_concat = torch.cat([x, multi_scale], dim=1)  # (B, in + multi_scale, H, W)

        x_flat = x.reshape(batch, channels, seq_len)  # (B, in, L)
        if hidden_state is None:
            hidden_state = torch.zeros(batch, self.state_size, seq_len, device=x.device, dtype=x.dtype)

        gate = self.feature_gate(x_flat)  # (B, in, L)
        x_gated = x_flat * gate  # (B, in, L)

        x_concat_flat = x_concat.reshape(batch, -1, seq_len)  # (B, in + multi_scale, L)

        state_params = self.state_proj(x_concat_flat)  # (B, 3*state_size, L)
        state_params = state_params.reshape(batch, 3, self.state_size, seq_len)
        input_factor, state_factor, time_step = state_params.unbind(1)  # 各为 (B, state_size, L)

        decay = torch.sigmoid(self.decay_param).reshape(1, -1, 1)  # (1, state_size, 1)
        time_step = torch.sigmoid(time_step)  # (B, state_size, L)
        new_state = hidden_state * torch.exp(-decay * time_step) + input_factor * state_factor  # (B, state_size, L)

        features = self.feature_proj(x_concat_flat)  # (B, 2*hidden_dim, L)
        content, selection = features.chunk(2, dim=1)  # 各为 (B, hidden_dim, L)
        selection = torch.sigmoid(selection)  # (B, hidden_dim, L)

        output = content * selection + new_state * (1 - selection) + self.bias  # (B, hidden_dim, L)
        output = self.output_proj(output)  # (B, in_channels, L)
        output = output.reshape(batch, -1, height, width)  # 恢复空间维度 (B, in_channels, H, W)
        return output, new_state


class BrainTumorBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        expand_ratio = 2
        hidden_dim = int(expand_ratio * in_channels)
        state_size = hidden_dim

        self.attention = NeuroAttention(in_channels, expand_ratio=expand_ratio, state_size=state_size)

        # Enhanced FFN
        self.ffn = nn.Sequential(
            Conv2DLayer(in_channels, in_channels * 2, 1),
            Conv2DLayer(in_channels * 2, in_channels, 1, act_layer=None)
        )

        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.norm2 = nn.InstanceNorm2d(in_channels)

        self.res_conv = nn.Identity()

    def forward(self, x1, x2=None):
        if x2 is not None:
            x = x1 + x2
        else:
            x = x1
        residual = self.res_conv(x)
        x_norm = self.norm1(x)
        x_attn, _ = self.attention(x_norm)  # 传入归一化后的特征
        x = x_attn + residual
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)  # 传入归一化后的特征
        x = x_ffn + residual
        return x

class ACANet(nn.Module):
    def __init__(self, num_classes):
        super(ACANet, self).__init__()
        self.backbone1 = DFormer_Small(pretrained=True)  # [64, 128, 256, 512]
        self.backbone2 = DFormer_Small(pretrained=True)  # [64, 128, 256, 512]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)

        # 多尺度LRSA模块
        self.afa_56 = AdaptiveFocusAttention(dim=64, qk_dim=64, heads=2)
        self.afa_28 = AdaptiveFocusAttention(dim=128, qk_dim=64, heads=4)
        self.afa_14 = AdaptiveFocusAttention(dim=256, qk_dim=128, heads=4)
        self.afa_7 = AdaptiveFocusAttention(dim=512, qk_dim=128, heads=4)
        # 特征融合与解码器
        self.mf0 = BrainTumorBlock(64)
        self.mf1 = BrainTumorBlock(128)
        self.mf2 = BrainTumorBlock(256)
        self.mf3 = BrainTumorBlock(512)
        self.ppd1 = PPD(128)
        self.ppd2 = PPD(128)
        self.decoder = Decoder(in_channel=64)

        # 动态轴向注意力

        self.ax2 = AdaptiveAxisAttention(128, 128, kernel_size=28)

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
        )

    def forward(self, x):
        x0 = x[:, 0:1, :, :]  # 通道0
        x1 = x[:, 1:2, :, :]  # 通道1
        x2 = x[:, 2:3, :, :]  # 通道2
        x3 = x[:, 3:4, :, :]  # 通道3
        x4 = self.conv1(x0)
        x5 = self.conv2(x1)
        x6 = self.conv1(x2)
        x7 = self.conv2(x3)

        outputs1, _ = self.backbone1(x4, x5)
        outputs2, _ = self.backbone2(x6, x7)
        pvt1_0 = self.afa_56(outputs1[0], region_size=32)
        pvt2_0 = self.afa_56(outputs2[0], region_size=32)
        pvt1_1 = self.afa_28(outputs1[1], region_size=16)
        pvt2_1 = self.afa_28(outputs2[1], region_size=16)
        pvt1_2 = self.afa_14(outputs1[2], region_size=12)
        pvt2_2 = self.afa_14(outputs2[2], region_size=12)
        # pvt1_3 = self.afa_7(outputs1[3], region_size=7)
        # pvt2_3 = self.afa_7(outputs2[3], region_size=7)
        pvt1_3 = outputs1[3]
        pvt2_3 = outputs2[3]
        # 特征融合
        F1 = self.mf0(pvt1_0, pvt2_0)
        F2 = self.mf1(pvt1_1, pvt2_1)
        F3 = self.mf2(pvt1_2, pvt2_2)
        F4 = self.mf3(pvt1_3, pvt2_3)

        # 处理GM分支
        pvt1_3 = self.conv3(pvt1_3)
        pvt2_3 = self.conv3(pvt2_3)
        pvt1_2 = self.conv4(pvt1_2)
        pvt2_2 = self.conv4(pvt2_2)

        aux1 = self.ppd1(pvt1_3, pvt1_2, pvt1_1)
        aux2 = self.ppd2(pvt2_3, pvt2_2, pvt2_1)

        aux1 = self.ax2(aux1)
        aux2 = self.ax2(aux2)
        aux1 = self.conv5(aux1)
        aux2 = self.conv5(aux2)

        # 动态融合策略
        lateral_map_GM1 = F.interpolate(aux1, scale_factor=8, mode='bilinear')
        lateral_map_GM2 = F.interpolate(aux2, scale_factor=8, mode='bilinear')

        GM = torch.abs(torch.sigmoid(aux1) - torch.sigmoid(aux2))
        aux = GM[:, 0, None, :, :]

        # 最终解码
        output = self.out(self.decoder(F1, F2, F3, F4, aux))
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)

        return lateral_map_GM1, lateral_map_GM2, output


if __name__ == '__main__':
    net = ACANet(3)
    a = torch.randn(4, 4, 224, 224)
    out = net(a)
    for i in range(len(out)):
        print(f"Output {i} shape: {out[i].shape}")
