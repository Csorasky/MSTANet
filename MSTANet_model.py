import torch.nn.functional as F
import torch.nn as nn
import torch
from timm.models.layers import DropPath, trunc_normal_
import math

def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / 4)
        s_width = int(d_width * 2)
        s_height = int(d_height * 2)
        t_1 = output.contiguous().view(batch_size, temp, d_height, d_width, 4, s_depth)
        spl = t_1.split(2, 4)
        stack = [t_t.contiguous().view(batch_size, temp, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).transpose(1, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, temp, s_height, s_width, s_depth)
        output = output.permute(0, 4, 1, 2, 3)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size,temp, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 2)
        output = output.permute(0, 4, 1, 3, 2)
        return output.contiguous()



class SimVP_Model(nn.Module):

    def __init__(self, in_shape, hid_S=64, hid_T=512, N_S=4, N_T=8, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            pass
        else:
            self.hid = MidMetaNet(640, 510, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)



    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        # encoder
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        # predictor
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        # decoder
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        return Y




def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """Encoder for MSTANet"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel):

        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC( C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        latent = x
        print("x", x.shape)
        for i in range(0, len(self.enc)-1):
            latent = self.enc[i](latent)
            print(latent.shape)

        enc1 = self.enc[-1](latent)
        print(enc1.shape)

        return enc1, enc1


class Decoder(nn.Module):
    """Decoder for MSTANet"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        samplings = sampling_generator(N_S+1, reverse=False)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        self.sigmoid = self.activate = nn.LeakyReLU(0.2)
        self.num_hidden = C_hid
        self.conv1 = nn.Conv2d(
            C_hid, C_hid*2, kernel_size=3,
            stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            C_hid, C_hid*2, kernel_size=3,
            stride=1, padding=1)

    def forward(self, hid, enc1=None):
            v = hid
            v = self.dec[0](v)
            z = enc1
            v_concat = self.conv1(v)
            z_concat = self.conv2(z)
            v_f, v_i = torch.split(v_concat, self.num_hidden, dim=1)
            z_f, z_i = torch.split(z_concat, self.num_hidden, dim=1)
            f = self.sigmoid(v_f + z_f)
            i = self.sigmoid(v_i + z_i)
            hid = f * z + i * v
            print("enc",hid.shape)
            for i in range(1, len(self.dec)):
                hid = self.dec[i](hid)
                print("enc",hid.shape)
            Y = self.readout(hid)

            return Y



class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class AttentionModule(nn.Module):
    """Large Kernel Attention for MSTANet"""

    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)  # depth-wise conv
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution

        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MixMlp(nn.Module):
    def __init__(self,
                 in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(hidden_features)                  # CFF: Convlutional feed-forward network
        self.act = act_layer()                                 # GELU
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1) # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class MetaBlock(nn.Module):

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'van':
            self.block = Predictor(n_feats=in_channels)

        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):


    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):

        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample 640-510
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers 510-510
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample 512-640
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x

        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SGAB(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor=15, attn='GLKA'):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut

class FeSFR(nn.Module):
    def __init__(self, n_feats,k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2 * n_feats
        c = n_feats//10
        t = 10
        gama = 2

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale sp Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats , n_feats , 7, 1, 7 // 2, groups=n_feats ),
            nn.Conv2d(n_feats , n_feats, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats, dilation=4),
            nn.Conv2d(n_feats , n_feats , 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats , n_feats , 5, 1, 5 // 2, groups=n_feats ),
            nn.Conv2d(n_feats , n_feats , 7, stride=1, padding=(7 // 2) * 3, groups=n_feats , dilation=3),
            nn.Conv2d(n_feats , n_feats , 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats , n_feats , 3, 1, 1, groups=n_feats ),
            nn.Conv2d(n_feats , n_feats , 5, stride=1, padding=(5 // 2) * 2, groups=n_feats , dilation=2),
            nn.Conv2d(n_feats , n_feats , 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats , n_feats , 3, 1, 1, groups=n_feats)
        self.X5 = nn.Conv2d(n_feats , n_feats , 5, 1, 5 // 2, groups=n_feats )
        self.X7 = nn.Conv2d(n_feats , n_feats , 7, 1, 7 // 2, groups=n_feats )

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        # Multiscale time Attention
        self.pool_conv = nn.Conv2d(n_feats,n_feats,16,stride=16,groups=n_feats)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.local_conv = nn.Sequential(
            nn.Conv1d(c, c, 1, groups=c),
            nn.Conv1d(c, c, 3, padding=1, groups=c, dilation=1),
            nn.Conv1d(c, c, 1))
        self.local_conv = nn.Conv1d(c,c,3,padding=1)

        self.global_conv1 = nn.Conv1d(c,gama*c,1)

        self.global_conv2 = nn.Conv1d(gama*c,c,1)

        self.global_conv3 = nn.Conv1d(t,gama*t,1)

        self.global_conv4 = nn.Conv1d(gama*t,t,1)

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        a_1 = a.clone()
        a_2 = a.clone()
        a_3 = a.clone()

        a = self.LKA3(a_1) * self.X3(a_1)+self.LKA5(a_2) * self.X5(a_2)+self.LKA7(a_3) * self.X7(a_3)

        b, c, H, W = x.size()
        c = c//10
        time_attn = self.pool_conv(x)
        time_attn = self.avg_pool(x).view(b, c,10)
        # local time attention
        time_attn = self.local_conv(time_attn)
        #global time attention
        time_attn = self.global_conv1(time_attn)
        time_attn = self.global_conv2(time_attn).view(b,10,c)
        time_attn = self.global_conv3(time_attn)
        time_attn = self.global_conv4(time_attn).view(b,10*c,1,1)
        attn = time_attn* a
        x = self.proj_last(x * attn)
        x = x  * self.scale + shortcut

        return x




class Predictor(nn.Module):

    def __init__(
            self, n_feats):
        super().__init__()

        self.LKA = FeSFR(n_feats)

        self.LFE = SGAB(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x





