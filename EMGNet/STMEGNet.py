import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
import glob
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from torch.nn import functional as F
import json
import gc
import pandas as pd



# MEGNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.elu = nn.ELU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.elu(out)
        return out

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ELU()
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
    
    def forward(self, features):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            lateral = lateral_conv(features[i])
            laterals.append(lateral)
        outputs = []
        for i, output_conv in enumerate(self.output_convs):
            outputs.append(output_conv(laterals[i]))
        return outputs

class MEGNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=4, Samples=18000, dropout_rate=0.5):
        super(MEGNet, self).__init__()
        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )
        # Backbone
        self.residual1 = ResidualBlock(16, 32, stride=2)
        self.residual2 = ResidualBlock(32, 64, stride=2)
        self.residual3 = ResidualBlock(64, 64, stride=2)

        # 注意力模块（放在global pooling之前）
        # 1) 通道注意力（SE）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # 全局汇聚得到每个通道的全局统计
            nn.Conv2d(64, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, 1),
            nn.Sigmoid()
        )
        # 2) 时间注意力（先沿H做均值，得到 N×C×T；再 Conv1d(C->1) 得到 N×1×T）
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 多尺度时间池化（保留时间维后再做1×1/1×2/1×4）
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 4))

        # 分类头：拼接后 -> 两层全连接 (ELU + Dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (1 + 2 + 4), 256),  # 64 * 7
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, nb_classes)
        )

        # 残差捷径以稳定训练
        self.shortcut = nn.Conv2d(64, 64, kernel_size=1, bias=True)

    def forward(self, x):
        # 输入支持 (B, E, C, T) 或 (B, C, T)
        if x.dim() == 4:
            batch, epochs, chans, times = x.shape
            x = x.reshape(batch * epochs, chans, times)
            merge_epochs = True
        else:
            merge_epochs = False

        # (N, 1, C, T)
        x = x.unsqueeze(1)

        # Backbone 输出 (N, 64, H, T)
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)  # -> (N, 64, H, T)

        # 注意力在全局池化之前
        # 通道注意力（SE）
        se_w = self.se(x)            # (N, 64, 1, 1)
        x = x * se_w                 # 广播到 (N, 64, H, T)

        # 时间注意力（沿H做均值 -> Conv1d(64->1)）
        x_mean_H = x.mean(dim=2)     # (N, 64, T)
        t_w = self.temporal_attention(x_mean_H)  # (N, 1, T)
        t_w = t_w.unsqueeze(2)       # (N, 1, 1, T)
        x = x * t_w                  # 广播到 (N, 64, H, T)

        # 残差捷径
        x = x + self.shortcut(x)

        # 多尺度池化（时间维 1/2/4，先于任何global pooling进行）
        p1 = self.pool1(x)           # (N, 64, 1, 1)
        p2 = self.pool2(x)           # (N, 64, 1, 2)
        p3 = self.pool3(x)           # (N, 64, 1, 4)
        x = torch.cat([p1, p2, p3], dim=3)  # (N, 64, 1, 7)

        # 分类
        x = self.classifier(x)       # (N, nb_classes)

        # 若输入包含多个epoch，做epoch维平均得到subject级别预测
        if merge_epochs:
            x = x.view(-1, epochs, x.size(-1)).mean(dim=1)  # (B, nb_classes)

        return x

    def __init__(self, nb_classes=4, Chans=4, Samples=18000, dropout_rate=0.5):
        super(MEGNet, self).__init__()
        self.final_channels = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )
        self.residual1 = ResidualBlock(16, 32, stride=2)
        self.residual2 = ResidualBlock(32, 64, stride=2)
        self.residual3 = ResidualBlock(64, 64, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7, 256),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, nb_classes)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, 1),
            nn.Sigmoid()
        )
        self.shortcut = nn.Conv2d(64, 64, 1)
    
    def forward(self, x):
        if x.dim() == 4:
            batch, epochs, chans, times = x.shape
            x = x.reshape(batch * epochs, chans, times)
            merge_epochs = True
        else:
            merge_epochs = False
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        features = [x]
        x = self.global_pool(x)
        se = self.se(x)
        x = x * se
        temporal_weights = self.temporal_attention(x.squeeze(2))
        x = x * temporal_weights.unsqueeze(2)
        shortcut = self.shortcut(x)
        x = x + shortcut
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        x = torch.cat([p1, p2, p3], dim=3)
        x = self.classifier(x)
        if merge_epochs:
            x = x.view(batch, epochs, -1).mean(dim=1)
        return x



    def __init__(self, target_length=72000, downsample_factor=12):
        self.train_mean = None
        self.train_std = None
        self.val_mean = None
        self.val_std = None
        self.target_length = int(target_length)
        self.downsample_factor = int(downsample_factor)
        self.n_channels = 260
        self._warning_printed = False
        self.common_channels = [
            'MLC11-44', 'MLC12-44', 'MLC13-44', 'MLC14-44', 'MLC15-44', 'MLC16-44', 'MLC17-44', 'MLC21-44',
            'MLC22-44', 'MLC23-44', 'MLC24-44', 'MLC31-44', 'MLC32-44', 'MLC41-44', 'MLC42-44', 'MLC51-44',
            'MLC53-44', 'MLC54-44', 'MLC55-44', 'MLC61-44', 'MLF11-44', 'MLF12-44', 'MLF13-44', 'MLF14-44',
            'MLF21-44', 'MLF22-44', 'MLF23-44', 'MLF24-44', 'MLF25-44', 'MLF31-44', 'MLF32-44', 'MLF33-44',
            'MLF34-44', 'MLF35-44', 'MLF41-44', 'MLF42-44', 'MLF43-44', 'MLF44-44', 'MLF45-44', 'MLF46-44',
            'MLF51-44', 'MLF52-44', 'MLF53-44', 'MLF54-44', 'MLF55-44', 'MLF56-44', 'MLF61-44', 'MLF62-44',
            'MLF63-44', 'MLF64-44', 'MLF65-44', 'MLF66-44', 'MLF67-44', 'MLO11-44', 'MLO12-44', 'MLO13-44',
            'MLO14-44', 'MLO21-44', 'MLO22-44', 'MLO23-44', 'MLO24-44', 'MLO31-44', 'MLO32-44', 'MLO33-44',
            'MLO34-44', 'MLO41-44', 'MLO42-44', 'MLO43-44', 'MLO44-44', 'MLO51-44', 'MLP11-44', 'MLP12-44',
            'MLP21-44', 'MLP22-44', 'MLP23-44', 'MLP31-44', 'MLP32-44', 'MLP33-44', 'MLP34-44', 'MLP35-44',
            'MLP41-44', 'MLP42-44', 'MLP43-44', 'MLP44-44', 'MLP45-44', 'MLP51-44', 'MLP52-44', 'MLP53-44',
            'MLP54-44', 'MLP56-44', 'MLP57-44', 'MLT11-44', 'MLT12-44', 'MLT13-44', 'MLT14-44', 'MLT15-44',
            'MLT16-44', 'MLT21-44', 'MLT22-44', 'MLT23-44', 'MLT24-44', 'MLT25-44', 'MLT27-44', 'MLT31-44',
            'MLT32-44', 'MLT33-44', 'MLT34-44', 'MLT35-44', 'MLT36-44', 'MLT37-44', 'MLT41-44', 'MLT42-44',
            'MLT43-44', 'MLT44-44', 'MLT45-44', 'MLT46-44', 'MLT47-44', 'MLT51-44', 'MLT52-44', 'MLT53-44',
            'MLT54-44', 'MLT56-44', 'MLT57-44', 'MRC11-44', 'MRC12-44', 'MRC13-44', 'MRC14-44', 'MRC15-44',
            'MRC16-44', 'MRC17-44', 'MRC21-44', 'MRC22-44', 'MRC23-44', 'MRC24-44', 'MRC25-44', 'MRC31-44',
            'MRC32-44', 'MRC41-44', 'MRC42-44', 'MRC51-44', 'MRC52-44', 'MRC53-44', 'MRC54-44', 'MRC55-44',
            'MRC61-44', 'MRC62-44', 'MRC63-44', 'MRF11-44', 'MRF12-44', 'MRF13-44', 'MRF14-44', 'MRF21-44',
            'MRF22-44', 'MRF23-44', 'MRF24-44', 'MRF25-44', 'MRF31-44', 'MRF32-44', 'MRF33-44', 'MRF34-44',
            'MRF35-44', 'MRF41-44', 'MRF42-44', 'MRF43-44', 'MRF44-44', 'MRF45-44', 'MRF46-44', 'MRF51-44',
            'MRF52-44', 'MRF53-44', 'MRF54-44', 'MRF55-44', 'MRF56-44', 'MRF61-44', 'MRF62-44', 'MRF63-44',
            'MRF65-44', 'MRF66-44', 'MRF67-44', 'MRO11-44', 'MRO12-44', 'MRO13-44', 'MRO14-44', 'MRO21-44',
            'MRO22-44', 'MRO23-44', 'MRO24-44', 'MRO31-44', 'MRO32-44', 'MRO33-44', 'MRO34-44', 'MRO41-44',
            'MRO42-44', 'MRO43-44', 'MRO44-44', 'MRO51-44', 'MRO52-44', 'MRO53-44', 'MRP11-44', 'MRP12-44',
            'MRP21-44', 'MRP22-44', 'MRP23-44', 'MRP31-44', 'MRP32-44', 'MRP33-44', 'MRP34-44', 'MRP35-44',
            'MRP41-44', 'MRP42-44', 'MRP43-44', 'MRP44-44', 'MRP45-44', 'MRP51-44', 'MRP54-44', 'MRP55-44',
            'MRP56-44', 'MRP57-44', 'MRT11-44', 'MRT12-44', 'MRT13-44', 'MRT14-44', 'MRT15-44', 'MRT16-44',
            'MRT21-44', 'MRT22-44', 'MRT23-44', 'MRT24-44', 'MRT25-44', 'MRT26-44', 'MRT31-44', 'MRT32-44',
            'MRT33-44', 'MRT34-44', 'MRT35-44', 'MRT36-44', 'MRT37-44', 'MRT41-44', 'MRT42-44', 'MRT43-44',
            'MRT44-44', 'MRT45-44', 'MRT46-44', 'MRT52-44', 'MRT53-44', 'MRT54-44', 'MRT55-44', 'MRT56-44',
            'MZC01-44', 'MZC02-44', 'MZC03-44', 'MZC04-44', 'MZF01-44', 'MZF02-44', 'MZF03-44', 'MZO01-44',
            'MZO02-44', 'MZO03-44', 'MZP01-44'
        ]

    def fit(self, data, ch_names=None, is_train=True):
        if data is None or data.size == 0:
            raise ValueError("输入数据为空")
        if ch_names is not None:
            common_indices, common_names = self.get_common_channel_indices(ch_names)
            if len(common_indices) > 0:
                data = data[:, common_indices, :]
                self.n_channels = len(common_indices)
                print(f"使用 {self.n_channels} 个共同通道")
                print(f"前5个共同通道: {common_names[:5]}")
            else:
                print(f"警告：未找到任何共同通道，使用所有通道")
        if is_train:
            self.train_mean = np.mean(data, axis=(0, 2))
            self.train_std = np.std(data, axis=(0, 2)) + 1e-8
        else:
            self.val_mean = np.mean(data, axis=(0, 2))
            self.val_std = np.std(data, axis=(0, 2)) + 1e-8

    def transform(self, data, ch_names=None, is_train=True):
        try:
            if data is None or data.size == 0:
                raise ValueError("输入数据为空")
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if len(data.shape) < 3:
                raise ValueError(f"数据形状不正确: {data.shape}")
            if ch_names is not None:
                common_indices, common_names = self.get_common_channel_indices(ch_names)
                if len(common_indices) > 0:
                    data = data[:, common_indices, :]
                    print(f"使用 {len(common_indices)} 个共同通道")
                    print(f"前5个共同通道: {common_names[:5]}")
                else:
                    print(f"警告：未找到任何共同通道，使用所有通道")
            if is_train:
                mean = self.train_mean
                std = self.train_std
            else:
                mean = self.val_mean
                std = self.val_std
            if mean is None or std is None:
                raise ValueError("请先调用fit方法进行数据拟合")
            mean = mean.reshape(1, -1, 1)
            std = std.reshape(1, -1, 1)
            if mean.shape[1] != data.shape[1]:
                print(f"警告：mean的通道数({mean.shape[1]})与数据通道数({data.shape[1]})不匹配")
                mean = np.mean(data, axis=(0, 2), keepdims=True)
                std = np.std(data, axis=(0, 2), keepdims=True) + 1e-8
            data = (data - mean) / std
            if data.shape[2] > self.target_length:
                start = (data.shape[2] - self.target_length) // 2
                data = data[:, :, start:start + self.target_length]
            elif data.shape[2] < self.target_length:
                pad_length = self.target_length - data.shape[2]
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                data = np.pad(data, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')
            if self.downsample_factor > 1:
                data = data[:, :, ::self.downsample_factor]
            print(f"最终数据形状: {data.shape}")
            print(f"目标时间步数: {self.target_length // self.downsample_factor}")
            if np.isnan(data).any() or np.isinf(data).any():
                print("警告：处理后的数据包含NaN或Inf值")
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            return data
        except Exception as e:
            print(f"数据转换过程中出错: {str(e)}")
            print(f"数据形状: {data.shape if hasattr(data, 'shape') else 'unknown'}")
            print(f"mean形状: {mean.shape if 'mean' in locals() else 'unknown'}")
            print(f"std形状: {std.shape if 'std' in locals() else 'unknown'}")
            print(f"target_length: {self.target_length}")
            print(f"downsample_factor: {self.downsample_factor}")
            print(f"n_channels: {self.n_channels}")
            raise

    def get_common_channel_indices(self, ch_names):
        common_indices = []
        common_names = []
        ch_names = [ch.decode('utf-8') if isinstance(ch, bytes) else ch for ch in ch_names]
        ch_name_to_index = {ch_name: i for i, ch_name in enumerate(ch_names)}
        for common_ch in self.common_channels:
            prefix = common_ch[:8]
            matching_channels = [ch for ch in ch_names if ch.startswith(prefix)]
            if matching_channels:
                ch_name = matching_channels[0]
                if ch_name not in common_names:
                    idx = ch_name_to_index[ch_name]
                    if idx < 260:
                        common_indices.append(idx)
                        common_names.append(ch_name)
                    elif not self._warning_printed:
                        print(f"警告：通道 {ch_name} 的索引 {idx} 超出范围 (0-259)")
                        self._warning_printed = True
            elif not self._warning_printed:
                print(f"警告：未找到匹配通道: {common_ch}")
                self._warning_printed = True
        if len(common_indices) == 0:
            print("错误：没有找到任何匹配的通道！")
            print("请检查通道名称格式是否正确。")
            print("数据中的通道名称示例:")
            for i in range(min(10, len(ch_names))):
                print(f"  {ch_names[i]}")
            print("\n共同通道列表中的前10个通道:")
            for i in range(min(10, len(self.common_channels))):
                print(f"  {self.common_channels[i]}")
            return [], []
        elif len(common_indices) != len(self.common_channels) and not self._warning_printed:
            print(f"警告：找到的通道数量 ({len(common_indices)}) 与期望的共同通道数量 ({len(self.common_channels)}) 不匹配")
            print("缺失的通道:")
            missing_channels = set(self.common_channels) - {ch[:8] for ch in common_names}
            for ch in list(missing_channels)[:10]:
                print(f"  {ch}")
            if len(missing_channels) > 10:
                print(f"  ... 等 {len(missing_channels)} 个通道")
            self._warning_printed = True
        common_indices = [idx for idx in common_indices if idx < 260]
        if len(common_indices) == 0:
            print("错误：所有通道索引都超出范围！")
            print(f"通道数量: {len(ch_names)}")
            return [], []
        return common_indices, common_names

    def augment_data(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        scale = torch.clamp(1.0 + torch.randn(1) * 0.2, min=0.8, max=1.2)
        noise = torch.randn_like(data) * 0.1
        shift = np.random.randint(-30, 30)
        mask = torch.bernoulli(torch.ones_like(data) * 0.9)
        augmented_data = data * scale + noise
        augmented_data = torch.roll(augmented_data, shifts=shift, dims=-1)
        augmented_data = augmented_data * mask
        return augmented_data


    def __init__(self, data_dir, processor, train=True, cache_dir=None, batch_size=2):
        self.data_files = glob.glob(os.path.join(data_dir, "*.h5"))
        if not self.data_files:
            raise ValueError(f"在目录 {data_dir} 中没有找到.h5文件")
        print(f"找到 {len(self.data_files)} 个.h5文件")
        self.train = train
        self.processor = processor
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.label_dict = {
            'sub-0006': 0, 'sub-0007': 0, 'sub-0010': 0, 'sub-0011': 0, 'sub-0012': 0, 'sub-0013': 0, 'sub-0014': 0,
            'sub-0015': 0, 'sub-0016': 0, 'sub-0017': 0, 'sub-0018': 0, 'sub-0019': 0, 'sub-0020': 0, 'sub-0022': 0,
            'sub-0023': 0, 'sub-0024': 0, 'sub-0026': 0, 'sub-0028': 0, 'sub-0029': 0, 'sub-0030': 0, 'sub-0032': 0,
            'sub-0034': 0, 'sub-0035': 0, 'sub-0036': 0, 'sub-0037': 0, 'sub-0038': 0, 'sub-0039': 0, 'sub-0041': 0,
            'sub-0042': 0, 'sub-0044': 0, 'sub-0045': 0, 'sub-0047': 0, 'sub-0048': 0, 'sub-0049': 0, 'sub-0050': 0,
            'sub-0051': 0, 'sub-0054': 0, 'sub-0055': 0, 'sub-0056': 0, 'sub-0058': 0, 'sub-0059': 0, 'sub-0061': 0,
            'sub-0062': 0, 'sub-0063': 0, 'sub-0064': 0, 'sub-0065': 0, 'sub-0067': 0, 'sub-0069': 0, 'sub-0070': 0,
            'sub-0071': 0, 'sub-0074': 0, 'sub-0075': 0, 'sub-0076': 0, 'sub-0077': 0, 'sub-0078': 0, 'sub-0080': 0,
            'sub-0081': 0, 'sub-0083': 0, 'sub-0084': 0, 'sub-0085': 0, 'sub-0086': 0, 'sub-0087': 0, 'sub-0098': 0,
            'sub-0099': 0, 'sub-0100': 0, 'sub-0101': 0, 'sub-0102': 0, 'sub-0103': 0, 'sub-0104': 0, 'sub-0105': 0,
            'sub-0106': 0, 'sub-0107': 0, 'sub-0108': 1, 'sub-0109': 1, 'sub-0110': 1, 'sub-0114': 1, 'sub-0118': 1,
            'sub-0119': 1, 'sub-0120': 1, 'sub-0122': 1, 'sub-0123': 1, 'sub-0128': 1, 'sub-0129': 1, 'sub-0130': 1,
            'sub-0132': 1, 'sub-0138': 0, 'sub-0144': 0, 'sub-0145': 0, 'sub-0146': 0, 'sub-0147': 0, 'sub-0149': 0,
            'sub-0151': 0, 'sub-0152': 0, 'sub-0153': 0, 'sub-0154': 0, 'sub-0155': 0, 'sub-0156': 0, 'sub-0158': 0,
            'sub-0159': 0, 'sub-0160': 0, 'sub-0161': 0, 'sub-0162': 0, 'sub-0163': 0, 'sub-0168': 0, 'sub-0169': 0,
            'sub-0170': 0, 'sub-0171': 0, 'sub-0172': 0, 'sub-0174': 0, 'sub-0175': 0, 'sub-0176': 0, 'sub-0177': 0,
            'sub-0179': 0, 'sub-0180': 0, 'sub-0181': 0, 'sub-0182': 0, 'sub-0183': 2, 'sub-0184': 0, 'sub-0185': 0,
            'sub-0186': 2, 'sub-0187': 2, 'sub-0189': 2, 'sub-0192': 0, 'sub-0193': 0, 'sub-0194': 2, 'sub-0195': 0,
            'sub-0196': 0, 'sub-0197': 0, 'sub-0199': 2, 'sub-0200': 0, 'sub-0217': 0, 'sub-0218': 0, 'sub-0219': 2,
            'sub-0220': 0, 'sub-0221': 2, 'sub-0223': 2, 'sub-0224': 2, 'sub-0225': 0, 'sub-0226': 2, 'sub-0227': 2,
            'sub-0228': 2, 'sub-0627': 2, 'sub-PD0146': 3, 'sub-PD0196': 3, 'sub-PD0208': 3, 'sub-PD0215': 3,
            'sub-PD0219': 3, 'sub-PD0260': 3, 'sub-PD0267': 3, 'sub-PD0289': 3, 'sub-PD0339': 3, 'sub-PD0439': 3,
            'sub-PD0471': 3, 'sub-PD0509': 3, 'sub-PD0576': 3, 'sub-PD0620': 3, 'sub-PD0622': 3, 'sub-PD0647': 3,
            'sub-PD0649': 3, 'sub-PD0653': 3, 'sub-PD0660': 3, 'sub-PD0666': 3, 'sub-PD0757': 3, 'sub-PD0760': 3,
            'sub-PD0782': 3, 'sub-PD0786': 3, 'sub-PD0792': 3, 'sub-PD0890': 3, 'sub-PD0952': 0, 'sub-PD0953': 3,
            'sub-PD0955': 3, 'sub-PD0959': 3, 'sub-PD0979': 3, 'sub-PD0980': 3, 'sub-0021': 0, 'sub-0025': 0,
            'sub-0031': 0, 'sub-0033': 0, 'sub-0040': 0, 'sub-0043': 0, 'sub-0046': 0, 'sub-0052': 0, 'sub-0053': 0,
            'sub-0057': 0, 'sub-0060': 0, 'sub-0066': 0, 'sub-0068': 0, 'sub-0072': 0, 'sub-0073': 0, 'sub-0079': 0,
            'sub-0082': 0, 'sub-0088': 0, 'sub-0089': 0, 'sub-0097': 0, 'sub-0111': 1, 'sub-0116': 1, 'sub-0134': 0,
            'sub-0148': 0, 'sub-0150': 0, 'sub-0157': 0, 'sub-0164': 0, 'sub-0165': 0, 'sub-0166': 0, 'sub-0167': 0,
            'sub-0173': 0, 'sub-0178': 0, 'sub-0188': 2, 'sub-0190': 2, 'sub-0198': 0, 'sub-0216': 0, 'sub-0222': 2,
            'sub-PD0256': 3, 'sub-PD0406': 3, 'sub-PD0458': 3, 'sub-PD0472': 3, 'sub-PD0512': 3, 'sub-PD0583': 3,
            'sub-PD0869': 3, 'sub-PD0950': 3
        }
        self.file_paths = []
        self.labels = []
        self._preprocess_and_cache_files()
        self.labels = torch.LongTensor(self.labels)
        self.file_handles = {}
        self.max_open_files = 10
        if len(self.file_paths) == 0:
            raise ValueError(f"数据集为空，请检查数据预处理过程。数据目录: {data_dir}")
        print(f"数据集初始化完成: {len(self.file_paths)} 个样本")

    def _preprocess_and_cache_files(self):
        if not self.cache_dir:
            self.file_paths = self.data_files
            self.labels = [self.label_dict[os.path.basename(f).split('_')[0]] for f in self.data_files]
            return
        self.cache_dir = os.path.abspath(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        temp_dir = r"D:\data_cache"
        os.makedirs(temp_dir, exist_ok=True)
        valid_files = []
        total_files = len(self.data_files)
        from tqdm import tqdm
        print("开始预处理文件...")
        for file_path in tqdm(self.data_files, total=total_files, desc="处理文件"):
            try:
                cache_file = os.path.join(self.cache_dir, os.path.basename(file_path).replace('.h5', '.npy'))
                if os.path.exists(cache_file):
                    try:
                        data = np.load(cache_file, allow_pickle=True)
                        if data.shape[1] == self.processor.n_channels:
                            if len(data.shape) == 3 and data.shape[2] == self.processor.target_length // self.processor.downsample_factor:
                                valid_files.append(file_path)
                                continue
                    except Exception:
                        if os.path.exists(cache_file):
                            os.remove(cache_file)
                with h5py.File(file_path, 'r') as hf:
                    ch_names = hf['ch_names'][:]
                    epochs_data = hf['epochs_data']
                    total_epochs = epochs_data.shape[0]
                    processed_data = np.zeros((total_epochs, self.processor.n_channels, 
                                            self.processor.target_length // self.processor.downsample_factor),
                                           dtype=np.float32)
                    batch_size = min(100, total_epochs)
                    for i in range(0, total_epochs, batch_size):
                        end_idx = min(i + batch_size, total_epochs)
                        batch_data = epochs_data[i:end_idx]
                        processed_batch = self.processor.transform(batch_data, ch_names)
                        processed_data[i:end_idx] = processed_batch
                        del batch_data
                        del processed_batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                temp_file = os.path.join(temp_dir, f"temp_{os.path.basename(cache_file)}")
                np.save(temp_file, processed_data)
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                os.rename(temp_file, cache_file)
                valid_files.append(file_path)
                del processed_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n处理文件 {file_path} 时出错: {str(e)}")
                continue
        self.file_paths = valid_files
        self.labels = [self.label_dict[os.path.basename(f).split('_')[0]] for f in valid_files]
        print(f"\n预处理完成: {len(valid_files)} 个有效文件")
        if len(self.file_paths) == 0:
            raise ValueError("没有有效的文件被处理，请检查数据预处理过程")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        ch_names = None
        try:
            cache_file = os.path.join(self.cache_dir, os.path.basename(file_path).replace('.h5', '.npy')) if self.cache_dir else None
            if cache_file and os.path.exists(cache_file):
                data = np.load(cache_file, allow_pickle=True)
                with h5py.File(file_path, 'r') as hf:
                    ch_names = hf['ch_names'][:]
            else:
                with h5py.File(file_path, 'r') as hf:
                    ch_names = hf['ch_names'][:]
                    epochs_data = hf['epochs_data'][:]
                    if epochs_data is None or epochs_data.size == 0:
                        raise ValueError(f"文件 {file_path} 中的epochs_data为空")
                    if len(epochs_data.shape) != 3:
                        raise ValueError(f"数据形状不正确: {epochs_data.shape}")
                    if epochs_data.shape[1] != self.processor.n_channels:
                        print(f"警告：文件 {file_path} 的通道数({epochs_data.shape[1]})与处理器通道数({self.processor.n_channels})不匹配")
                        epochs_data = epochs_data[:, :self.processor.n_channels, :]
                    data = self.processor.transform(epochs_data, ch_names)
                    if cache_file:
                        np.save(cache_file, data)
            if len(data.shape) != 3:
                raise ValueError(f"数据形状不正确: {data.shape}")
            target_epochs = 10
            if data.shape[0] > target_epochs:
                indices = np.random.choice(data.shape[0], target_epochs, replace=False)
                data = data[indices]
            elif data.shape[0] < target_epochs:
                repeat_times = target_epochs // data.shape[0] + 1
                data = np.repeat(data, repeat_times, axis=0)
                data = data[:target_epochs]
            data = torch.from_numpy(data).float()
            filename = os.path.basename(file_path)
            subject_id = filename.split('_')[0]
            label = self.label_dict[subject_id]
            return data, label, subject_id  # 返回subject_id作为样本标识符
        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {str(e)}")
            print(f"文件路径: {file_path}")
            print(f"数据形状: {data.shape if 'data' in locals() else 'unknown'}")
            raise

    def print_data_stats(self):
        print("\n=== 详细数据统计信息 ===")
        print(f"总文件数: {self.data_stats['total_files']}")
        print(f"有效文件数: {self.data_stats['valid_files']}")
        print(f"无效文件数: {self.data_stats['invalid_files']}")
        print("\n形状不匹配的文件:")
        for filename, error in self.data_stats['shape_mismatch_files']:
            print(f"  - {filename}: {error}")
        print("\n通道不匹配的文件:")
        for filename, error in self.data_stats['channel_mismatch_files']:
            print(f"  - {filename}: {error}")
        print("\n时间序列不匹配的文件:")
        for filename, error in self.data_stats['time_series_mismatch_files']:
            print(f"  - {filename}: {error}")
        print("\n其他错误文件:")
        for filename, error in self.data_stats['error_files']:
            print(f"  - {filename}: {error}")
        print("=======================\n")


    torch.multiprocessing.set_sharing_strategy('file_system')
    main()