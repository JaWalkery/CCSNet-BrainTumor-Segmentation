import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from networks.pvtv2_new import pvt_v2_b2


def dwt_init(x):
    """Haar小波变换：输入(B,C,H,W)→输出(LL, HL, LH, HH)四个子带"""
    h, w = x.shape[2], x.shape[3]
    x = x[..., :h - (h % 2), :w - (w % 2)]  # 裁剪为偶数尺寸
    x01 = x[:, :, 0::2, :] / 2  # 偶数行（H/2行）
    x02 = x[:, :, 1::2, :] / 2  # 奇数行（H/2行，与x01尺寸相同）
    x1 = x01[:, :, :, 0::2]  # 偶数列（W/2列）
    x2 = x02[:, :, :, 0::2]  # 偶数列（与x1尺寸相同）
    x3 = x01[:, :, :, 1::2]  # 奇数列（W/2列）
    x4 = x02[:, :, :, 1::2]  # 奇数列（与x3尺寸相同）

    # 四个子带尺寸完全匹配，可安全相加
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    """逆Haar小波变换：输入(B,4C,H,W)→输出重构特征(B,C,2H,2W)"""
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()

    if in_channel % 4 != 0:
        raise ValueError(f"输入通道数必须是4的倍数，当前为{in_channel}")

    x1 = x[:, 0::4, :, :]  # LL子带
    x2 = x[:, 1::4, :, :]  # HL子带
    x3 = x[:, 2::4, :, :]  # LH子带
    x4 = x[:, 3::4, :, :]  # HH子带

    out_batch = in_batch
    out_channel = in_channel // 4
    out_height = r * in_height
    out_width = r * in_width
    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device)

    # 填充重构（确保尺寸匹配）
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class WaveletFusionBlock(nn.Module):
    def __init__(self, in_channels, split_num=16, levels=2):
        super(WaveletFusionBlock, self).__init__()
        self.in_channels = in_channels
        self.split_num = split_num  # 通道分割块数
        self.wavelet_levels = levels  # 小波变换级数
        self.split_channels = in_channels // self.split_num  # 每块的通道数

        # 动态调整通道数为4的倍数（适配小波子带）
        self.adjust_conv = nn.Conv2d(
            self.split_channels,
            ((self.split_channels + 3) // 4) * 4,  # 向上取整到4的倍数
            kernel_size=1
        )

        self.dwt = DWT()
        self.iwt = IWT()
        self.subband_weights = nn.Parameter(torch.tensor([1.2, 1.0, 1.0, 0.8]))

        # SE风格频率注意力
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 初始融合卷积
        self.init_fusion_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.channel_adjust_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def dwt_transform(self, x):
        """带动态Padding的多级小波变换"""
        coeffs = []
        current = x
        for _ in range(self.wavelet_levels):
            h_curr, w_curr = current.shape[2], current.shape[3]
            pad_h = (2 - h_curr % 2) % 2
            pad_w = (2 - w_curr % 2) % 2
            current = F.pad(current, (0, pad_w, 0, pad_h), mode='replicate')
            ll, hl, lh, hh = self.dwt(current)
            coeffs.append((ll, hl, lh, hh))
            current = ll
        return coeffs

    def forward(self, feat_a, feat_b):

        original_size = feat_a.shape[2:]
        concat_feat = torch.cat([feat_a, feat_b], dim=1)  # 拼接特征 (B, 2C, H, W)
        init_fused_feat = self.init_fusion_conv(concat_feat)  # 初始融合特征 (B, C, H, W)

        # 2. 通道分割与小波频率特征提取
        split_feats = torch.split(init_fused_feat, self.split_channels, dim=1)
        freq_feat_list = []
        for part in split_feats:
            part = self.adjust_conv(part)  # 调整通道数适配小波变换
            wavelet_coeffs = self.dwt_transform(part)
            level_feats = []
            for ll, hl, lh, hh in wavelet_coeffs:
                # 子带池化与加权融合
                ll_pool = F.adaptive_avg_pool2d(ll, (1, 1))
                hl_pool = F.adaptive_avg_pool2d(hl, (1, 1))
                lh_pool = F.adaptive_avg_pool2d(lh, (1, 1))
                hh_pool = F.adaptive_avg_pool2d(hh, (1, 1))

                weights = F.softmax(self.subband_weights, dim=0)
                level_feat = (
                        weights[0] * ll_pool +
                        weights[1] * hl_pool +
                        weights[2] * lh_pool +
                        weights[3] * hh_pool
                )
                level_feats.append(level_feat)

            # 层级融合
            if len(level_feats) > 1:
                level_weights = torch.linspace(0.5, 1.0, len(level_feats), device=part.device)
                level_weights = level_weights / level_weights.sum()
                freq_feat = sum(f * w for f, w in zip(level_feats, level_weights))
            else:
                freq_feat = level_feats[0]
            freq_feat_list.append(freq_feat)

        # 3. 频率注意力加权
        freq_feat = torch.cat(freq_feat_list, dim=1)
        freq_weight = self.freq_attention(freq_feat)  # 频率注意力权重 (B, C, 1, 1)
        freq_weight_comp = 1 - freq_weight  # 互补权重

        # 4. 特征校准
        feat_b_calib = freq_weight * feat_b  # 校准特征B
        feat_a_calib = freq_weight_comp * feat_a  # 校准特征A
        fused_feat = self.channel_adjust_conv(feat_a_calib + feat_b_calib)  # 融合校准特征

        # 5. 空间注意力加权
        spatial_feat = self.channel_adjust_conv(fused_feat + init_fused_feat)  # 空间注意力输入
        spatial_weight = self.spatial_attention(spatial_feat)  # 空间注意力权重 (B, 1, H, W)
        fused_feat_att = spatial_weight * fused_feat  # 空间加权融合特征
        feat_b_att = spatial_weight * feat_b  # 空间加权特征B

        # 6. 输出最终融合特征（带残差连接）
        output = self.output_conv(fused_feat_att + feat_b_att)
        return output + self.res_scale * init_fused_feat



# 频谱特征处理模块
class FFParser_n(nn.Module):
    """2D傅里叶变换解析器（彻底解决复数插值问题）"""

    def __init__(self, dim, init_h=32, init_w=32):
        super().__init__()
        # 分别存储实部和虚部权重（均为实数类型）
        self.real_weight = nn.Parameter(torch.randn(dim, init_h, init_w) * 0.02)  # 实部权重
        self.imag_weight = nn.Parameter(torch.randn(dim, init_h, init_w) * 0.02)  # 虚部权重
        self.init_h = init_h
        self.init_w = init_w

    def forward(self, x):
        B, C, H, W = x.shape  # 输入形状: (B, C, H, W)

        # 1. 计算输入的傅里叶变换（得到复数频谱）
        x = x.to(torch.float32)
        x_fft = torch.fft.rfftn(x, dim=(2, 3), norm='ortho')  # 形状: (B, C, H, W_rfft)，其中W_rfft = W//2 + 1

        # 2. 对实部和虚部权重进行插值（纯实数操作）
        # 实部权重插值
        real_weight = F.interpolate(
            self.real_weight.unsqueeze(0),  # 增加批次维度: (1, C, init_h, init_w)
            size=(H, x_fft.shape[-1]),  # 目标尺寸: (H, W_rfft)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # 移除批次维度: (C, H, W_rfft)

        # 虚部权重插值
        imag_weight = F.interpolate(
            self.imag_weight.unsqueeze(0),  # 增加批次维度: (1, C, init_h, init_w)
            size=(H, x_fft.shape[-1]),  # 目标尺寸: (H, W_rfft)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # 移除批次维度: (C, H, W_rfft)

        # 3. 将实数权重转换为复数权重，并确保设备一致
        complex_weight = torch.complex(
            real_weight.to(x_fft.device),  # 移动到与输入频谱相同的设备
            imag_weight.to(x_fft.device)
        )  # 形状: (C, H, W_rfft)

        # 4. 频谱调制（复数乘法）
        x_fft = x_fft * complex_weight

        # 5. 逆傅里叶变换转回空间域
        x = torch.fft.irfftn(x_fft, s=(H, W), dim=(2, 3), norm='ortho')

        return x.reshape(B, C, H, W)


class LayerNorm(nn.Module):
    r"""适配2D影像的LayerNorm，支持channels_last和channels_first格式"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 2D适配
            return x


class MlpChannel(nn.Module):
    """2D版本通道注意力MLP层"""

    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)  # 2D卷积
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)  # 2D卷积

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Spectral_Layer(nn.Module):
    """适配2D影像的频谱特征处理层"""

    def __init__(self, dim, stage=1, in_shape=[128, 128]):
        super().__init__()
        self.dim = dim

        # 计算初始权重尺寸（根据输入尺寸和stage）
        self.init_h = in_shape[0] // (2 ** (stage + 1))
        self.init_w = in_shape[1] // (2 ** (stage + 1))
        self.init_h = max(1, self.init_h)
        self.init_w = max(1, self.init_w)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim // 2)
        self.ffp_module = FFParser_n(dim, self.init_h, self.init_w)

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim, "输入通道数与Layer维度不匹配"

        n_tokens = x.shape[2:].numel()  # H*W
        img_dims = x.shape[2:]  # (H, W)

        x_reshape = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
        norm1_x = self.norm1(x_reshape)
        norm1_x = norm1_x.reshape(B, C, *img_dims)
        x_fft = self.ffp_module(norm1_x)

        x_fft_flat = x_fft.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
        norm2_x_fft = self.norm2(x_fft_flat)
        x_spatial = self.mlp(norm2_x_fft.transpose(-1, -2).reshape(B, C, *img_dims))  # 回到2D
        out = x + x_spatial
        return out


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


class SimpleContext(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SimpleContext, self).__init__()
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)
        self.branch1 = BasicConv2d(in_channel, out_channel, 3, padding=3, dilation=3)
        self.branch2 = BasicConv2d(in_channel, out_channel, 3, padding=5, dilation=5)
        self.branch3 = BasicConv2d(in_channel, out_channel, 3, padding=7, dilation=7)

        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.spectral = Spectral_Layer(dim=out_channel)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat([x0, x1, x2, x3], dim=1)
        x_cat = self.conv_cat(x_cat)

        # 加入频谱特征处理
        x_spectral = self.spectral(x_cat)

        x_out = x_spectral + self.conv_res(x)
        return x_out


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
        self.conv5 = nn.Conv2d(3 * channel, 3, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

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
        self.conv2 = BasicConv2d(320, 128, 3, padding=1)
        self.conv3 = BasicConv2d(512, 320, 3, padding=1)
        self.upsample = lambda x, target: F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x0, x1, x2, x3, GM_MA):
        out_diff_3 = self.upsample(GM_MA, x3).expand_as(x3)
        x2 = self.upsample(self.conv3(x3 + x3 * out_diff_3), x2) * x2
        out_diff_2 = self.upsample(GM_MA, x2).expand_as(x2)
        x1 = self.upsample(self.conv2(x2 + x2 * out_diff_2), x1) * x1
        out_diff_1 = self.upsample(GM_MA, x1).expand_as(x1)
        x0 = self.upsample(self.conv1(x1 + x1 * out_diff_1), x0) * x0
        return x0

class BoundaryAttention(nn.Module):
    def __init__(self, channels):
        super(BoundaryAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        sobel_kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.sobel_kernel = nn.Parameter(
            torch.cat([sobel_kernel_x.repeat(channels, 1, 1, 1),
                       sobel_kernel_y.repeat(channels, 1, 1, 1)]),
            requires_grad=False
        )

    def forward(self, x):
        B, C, H, W = x.shape  # 明确4D输入维度
        # 双方向边缘检测（输出4D：[B, 2*C, H, W]）
        edge = torch.abs(F.conv2d(x, self.sobel_kernel, padding=1, groups=C))
        edge_x = edge[:, :C, :, :]  # x方向边缘：[B, C, H, W]
        edge_y = edge[:, C:, :, :]  # y方向边缘：[B, C, H, W]
        edge = (edge_x + edge_y).mean(dim=1, keepdim=True)  # 通道平均→[B, 1, H, W]（4D，与attn维度匹配）
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)  # 输出4D：[B, 1, H, W]

        attn = self.sigmoid(attn + edge)
        return x * attn + x  # 输出仍为4D：[B, C, H, W]

def estimate_tumor_center(x):
    B, C, H, W = x.shape
    feat_score = x.mean(dim=1)  # 通道平均生成特征响应图：[B, H, W]
    # 生成坐标网格
    h_grid = torch.arange(H, device=x.device).view(1, H, 1).repeat(B, 1, W)
    w_grid = torch.arange(W, device=x.device).view(1, 1, W).repeat(B, H, 1)
    # 加权平均计算中心
    total_score = feat_score.sum(dim=[1, 2], keepdim=True) + 1e-6
    h_center = (feat_score * h_grid).sum(dim=[1, 2], keepdim=True) / total_score  # [B, 1, 1]
    w_center = (feat_score * w_grid).sum(dim=[1, 2], keepdim=True) / total_score  # [B, 1, 1]

    return torch.cat([h_center, w_center], dim=1).unsqueeze(2)  # (B, 2, 1, 1)


class Dynamic1DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, flow_type='radial'):
        super(Dynamic1DConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.flow_type = flow_type
        assert kernel_size % 2 == 1, "卷积核必须为奇数"
        self.padding = kernel_size // 2

        self.kernel_gen = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1
        )

    def forward(self, x, offsets, tumor_center=None):
        B, C, H, W = x.shape
        K = self.kernel_size
        device = x.device

        # 生成动态采样网格
        if self.flow_type == 'radial':
            assert tumor_center is not None, "径向流必须输入肿瘤中心坐标"
            h_center, w_center = tumor_center[:, 0], tumor_center[:, 1]

            h_idx = torch.arange(H, device=device).view(1, H, 1).repeat(B, 1, W)
            w_idx = torch.arange(W, device=device).view(1, 1, W).repeat(B, H, 1)
            dh = h_idx - h_center
            dw = w_idx - w_center
            radial_norm = torch.sqrt(dh ** 2 + dw ** 2 + 1e-6)
            dir_h = dh / radial_norm
            dir_w = dw / radial_norm

            # 径向采样基
            radial_base = torch.linspace(-self.padding, self.padding, K, device=device).view(1, K, 1, 1)
            radial_base = radial_base.repeat(B, 1, H, W)
            h_grid = radial_base * dir_h.unsqueeze(1) + offsets
            w_grid = radial_base * dir_w.unsqueeze(1) + offsets

        else:  # 局部流
            grad_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean(dim=1, keepdim=True)
            grad_h = F.pad(grad_h, (0, 0, 1, 0))
            grad_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean(dim=1, keepdim=True)
            grad_w = F.pad(grad_w, (1, 0, 0, 0))

            # 局部采样基
            local_base = torch.linspace(-self.padding, self.padding, K, device=device).view(1, K, 1, 1)
            local_base = local_base.repeat(B, 1, H, W)
            h_grid = local_base * grad_h + offsets
            w_grid = local_base * grad_w + offsets

        # 坐标归一化
        h_grid = h_grid / (H - 1) * 2
        w_grid = w_grid / (W - 1) * 2

        # 特征采样
        grid = torch.stack([w_grid, h_grid], dim=-1)
        grid = grid.permute(0, 2, 3, 1, 4)
        grid_reshaped = grid.reshape(B * H * W, 1, K, 2)

        x_reshaped = x.view(B * H * W, C, 1, 1)
        sampled = F.grid_sample(
            x_reshaped,
            grid_reshaped,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(2)
        # 动态卷积计算
        kernels = self.kernel_gen(x)
        kernels = kernels.view(B, self.out_channels, K, H, W)
        kernels = kernels.permute(0, 3, 4, 1, 2)
        kernels = kernels.reshape(B * H * W, self.out_channels, K)
        kernels = kernels.permute(0, 2, 1)

        out = torch.bmm(sampled, kernels)
        out = out.view(B, H, W, C, self.out_channels)
        out = out.permute(0, 4, 1, 2, 3)
        out = out.sum(dim=4)

        assert out.shape[1] == self.out_channels, "动态卷积通道错误"
        return out

class FlowGuidedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 7]):
        super(FlowGuidedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        assert all(k % 2 == 1 for k in kernel_sizes), "卷积核必须为奇数"
        # 偏移量生成
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2 * sum(kernel_sizes), kernel_size=3, padding=1)
        )
        self.tanh = nn.Tanh()
        # 辅助模块
        self.boundary_attn = BoundaryAttention(in_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        # 通道分配
        self.radial_out = out_channels // 2
        self.local_out = out_channels - self.radial_out
        assert self.radial_out % self.num_scales == 0, f"径向流通道{self.radial_out}需被{self.num_scales}整除"
        assert self.local_out % self.num_scales == 0, f"局部流通道{self.local_out}需被{self.num_scales}整除"

        # 双分支动态1D卷积
        self.radial_convs = nn.ModuleList([
            Dynamic1DConv(
                in_channels=in_channels,
                out_channels=self.radial_out // self.num_scales,
                kernel_size=k,
                flow_type='radial'
            ) for k in kernel_sizes
        ])
        self.local_convs = nn.ModuleList([
            Dynamic1DConv(
                in_channels=in_channels,
                out_channels=self.local_out // self.num_scales,
                kernel_size=k,
                flow_type='local'
            ) for k in kernel_sizes
        ])
        # 特征融合
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(out_channels, out_channels, 1)

    def iterative_cumulative_constraint(self, offsets, half_k):
        B, K, H, W = offsets.size()
        constrained = torch.zeros_like(offsets)
        constrained[:, half_k, :, :] = offsets[:, half_k, :, :]
        # 迭代累积约束
        for i in range(1, half_k + 1):
            constrained[:, half_k - i, :, :] = 0.5 * constrained[:, half_k - i + 1, :, :] + 0.5 * offsets[:, half_k - i,
                                                                                                  :, :]
            constrained[:, half_k + i, :, :] = 0.5 * constrained[:, half_k + i - 1, :, :] + 0.5 * offsets[:, half_k + i,
                                                                                                  :, :]

        # 分组卷积平滑
        conv_kernel = torch.ones(K, 1, 3, 3, device=constrained.device) / 9
        constrained = F.conv2d(constrained, conv_kernel, padding=1, groups=K)
        return constrained

    def forward(self, x):
        # 仅接受4D输入 (B, C, H, W)
        assert x.dim() == 4, f"输入必须是4D张量，实际输入维度: {x.dim()}"
        B, C, H, W = x.shape
        assert C == self.in_channels, f"输入通道数不匹配: 预期{self.in_channels}, 实际{C}"
        residual = self.residual(x)
        x = self.boundary_attn(x)  # 修复后，此处输出仍为4D
        # 生成并约束偏移量（输入为4D，无维度错误）
        raw_offsets = self.offset_conv(x)
        radial_offsets = []
        local_offsets = []
        offset_idx = 0

        for k in self.kernel_sizes:
            half_k = k // 2
            radial_raw = raw_offsets[:, offset_idx:offset_idx + k, :, :]
            local_raw = raw_offsets[:, offset_idx + sum(self.kernel_sizes):offset_idx + sum(self.kernel_sizes) + k, :,
                        :]

            radial_norm = self.tanh(radial_raw)
            local_norm = self.tanh(local_raw)

            radial_constrained = self.iterative_cumulative_constraint(radial_norm, half_k)
            local_constrained = self.iterative_cumulative_constraint(local_norm, half_k)

            radial_offsets.append(radial_constrained)
            local_offsets.append(local_constrained)
            offset_idx += k

        # 估计肿瘤中心
        tumor_center = estimate_tumor_center(x)
        # 双分支特征计算
        radial_features = [self.radial_convs[i](x, radial_offsets[i], tumor_center) for i in range(self.num_scales)]
        local_features = [self.local_convs[i](x, local_offsets[i]) for i in range(self.num_scales)]
        radial_cat = torch.cat(radial_features, dim=1)
        local_cat = torch.cat(local_features, dim=1)
        # 特征融合与输出
        fusion_input = torch.cat([radial_cat, local_cat], dim=1)
        fusion_out = self.scale_fusion(fusion_input)
        final_out = self.final_conv(fusion_out)
        return final_out + residual


class WaveletFusionNet(nn.Module):
    def __init__(self, num_classes):
        super(WaveletFusionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
        self.backbone1 = pvt_v2_b2()  # [64, 128, 320, 512]
        self.backbone2 = pvt_v2_b2()  # [64, 128, 320, 512]

        # 加载预训练权重
        path = '/media/user/sdb1/sy/ACANet1/networks/pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone1.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone1.load_state_dict(model_dict)
        self.backbone2.load_state_dict(model_dict)

        self.wff1 = WaveletFusionBlock(in_channels=64)
        self.wff2 = WaveletFusionBlock(in_channels=128)
        self.wff3 = WaveletFusionBlock(in_channels=320)
        self.wff4 = WaveletFusionBlock(in_channels=512)  # 512/32=16，是4的倍数

        self.ppd1 = PPD(128)
        # self.context_11 = SimpleContext(128, 128)  # 已包含频谱处理
        # self.context_12 = SimpleContext(320, 128)  # 已包含频谱处理
        # self.context_13 = SimpleContext(512, 128)  # 已包含频谱处理
        self.ppd2 = PPD(128)
        # self.context_21 = SimpleContext(128, 128)  # 已包含频谱处理
        # self.context_22 = SimpleContext(320, 128)  # 已包含频谱处理
        # self.context_23 = SimpleContext(512, 128)  # 已包含频谱处理
        self.conv3 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
        )
        self.decoder = Decoder(in_channel=64)
        self.fgb = FlowGuidedBlock(128, 128, [5, 7])
        # self.fgb_320 = FlowGuidedBlock(
        #             in_channels=320,
        #             out_channels=128,
        #             kernel_sizes=[3,5]  #
        #         )
        # self.fgb_512 = FlowGuidedBlock(
        #             in_channels=512,
        #             out_channels=128,
        #             kernel_sizes=[3,3]
        #         )

        self.conv3 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = torch.cat((x[:, 1, None, :, :], x[:, 2, None, :, :]), dim=1)
        x2 = torch.cat((x[:, 0, None, :, :], x[:, 3, None, :, :]), dim=1)
        x1 = self.conv1(x1)  # 1,3,224,224
        x2 = self.conv2(x2)

        pvt1 = self.backbone1(x1)
        pvt2 = self.backbone2(x2)

        F1 = self.wff1(pvt1[0], pvt2[0])
        F2 = self.wff2(pvt1[1], pvt2[1])
        F3 = self.wff3(pvt1[2], pvt2[2])
        F4 = self.wff4(pvt1[3], pvt2[3])

        # 处理GM分支
        pvt1_3 = self.conv4(pvt1[3])
        pvt2_3 = self.conv4(pvt2[3])
        pvt1_2 = self.conv3(pvt1[2])
        pvt2_2 = self.conv3(pvt2[2])
        # pvt1_3 = self.fgb_512(pvt1[3])
        # pvt2_3 = self.fgb_512(pvt2[3])
        # pvt1_2 = self.fgb_320(pvt1[2])
        # pvt2_2 = self.fgb_320(pvt2[2])
        pvt1_1 = self.fgb(pvt1[1])
        pvt2_1 = self.fgb(pvt2[1])

        aux1 = self.ppd1(pvt1_3, pvt1_2, pvt1_1)
        aux2 = self.ppd2(pvt2_3, pvt2_2, pvt2_1)
        # f11 = self.context_11(pvt1[1])
        # f12 = self.context_12(pvt1[2])
        # f13 = self.context_13(pvt1[3])
        # #
        # # f21 = self.context_21(pvt2[1])
        # # f22 = self.context_22(pvt2[2])
        # f23 = self.context_23(pvt2[3])
        aux1 = F.interpolate(aux1, scale_factor=8, mode='bilinear')
        aux2 = F.interpolate(aux2, scale_factor=8, mode='bilinear')

        m5 = torch.abs(torch.sigmoid(aux1) - torch.sigmoid(aux2))  # [bs,3,28,28]
        m5 = m5[:, 0, None, :, :]

        # 解码输出
        logits = self.out(self.decoder(F1, F2, F3, F4, m5))
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        return  aux1,  aux2, logits

if __name__ == '__main__':
    # 检查是否有可用的CUDA设备
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    net = WaveletFusionNet(3).to(device)  # 将模型移动到CUDA设备
    a = torch.randn(4, 4, 224, 224).to(device)  # 将输入数据移动到CUDA设备
    out = net(a)
    for i in range(len(out)):
        print(out[i].shape)
    from thop import profile

    net = WaveletFusionNet(3)
    input = torch.randn(1, 4, 224, 224)
    flops, params = profile(net, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")