"""borrowed and modified from https://github.com/CompVis/taming-transformers"""
import torch.nn.init as init
from torch import nn

input2hid = {
    512: 256,
    256: 256,
    1536: 512
}


class TwoLayerMLP(nn.Module):
    # def __init__(self, input_size=1536, hidden_size=512, output_size=1536, dropout=0.5):
    # def __init__(self, input_size=512, hidden_size=256, output_size=512, dropout=0.1):
    # def __init__(self, input_size=1024, hidden_size=512, output_size=1024, dropout=0.1):
    def __init__(self, input_size=256, hidden_size=256, output_size=256, dropout=0.5):
        # def __init__(self, input_size=128, hidden_size=128, output_size=128, dropout=0.5):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.ReLU() nn.GELU() nn.LeakyReLU() nn.SELU() nn.CELU() nn.SiLU() nn.Mish()
        self.relu = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self._initialize_weights() # 加入会增大clip_recon_loss，先弃用

    def _initialize_weights(self):
        # 初始化带有ReLU的层 (fc1 和 fc3)
        for layer in [self.fc1, self.fc3]:
            init.kaiming_normal_(
                layer.weight,
                mode='fan_in',  # 补偿前向传播的方差
                nonlinearity='relu'  # 针对ReLU的校正
            )
            init.zeros_(layer.bias)  # 偏置初始化为0

        # 初始化输出层 (fc2)
        std = 1.0 / (self.fc2.in_features ** 0.5)  # 1/sqrt(256) = 1/16
        init.normal_(self.fc2.weight, mean=0, std=std)
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout1(out)
        out = self.fc3(out)
        out = self.relu2(out)
        # out = self.dropout2(out)
        out = self.fc2(out)
        return out

    def get_layer_output(self, x):
        out1 = self.fc1(x)
        out1 = self.relu(out1)
        out2 = self.fc3(out1)
        out2 = self.relu2(out2)
        out3 = self.fc2(out2)
        return out1, out2, out3


class OneLayerMLP(nn.Module):
    def __init__(self, input_size=256, output_size=256):
        super(OneLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        return out


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, input_dim=512, double_z=True, **ignore_kwargs):
        super().__init__()
        hidden_size = input2hid.get(input_dim, 256)
        self.encoder = TwoLayerMLP(input_size=input_dim, hidden_size=hidden_size, output_size=z_channels)

    def forward(self, x):
        h = self.encoder(x)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, input_dim=512, give_pre_end=False, **ignorekwargs):
        super().__init__()
        hidden_size = input2hid.get(input_dim, 256)
        self.decoder = TwoLayerMLP(input_size=z_channels, hidden_size=hidden_size, output_size=input_dim)

    def forward(self, z):
        h = self.decoder(z)
        return h


class Decoder_Concat(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, input_dim=512, give_pre_end=False, **ignorekwargs):
        super().__init__()
        hidden_size = input2hid.get(input_dim, 256)
        self.decoder = TwoLayerMLP(input_size=z_channels * 3, hidden_size=hidden_size, output_size=input_dim)

    def forward(self, z):
        h = self.decoder(z)
        return h
