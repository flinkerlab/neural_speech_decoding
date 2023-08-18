import os
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np
import utils.lreq as ln
import utils.lreq_causal as ln_c
from utils.registry import *
from modules.Swin3D_blocks import *

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
def db(x, noise=-80, slope=35, powerdb=True):
    if powerdb:
        return (
            (2 * torchaudio.transforms.AmplitudeToDB()(x)).clamp(min=noise)
            - slope
            - noise
        ) / slope
    else:
        return (
            (torchaudio.transforms.AmplitudeToDB()(x)).clamp(min=noise) - slope - noise
        ) / slope


def to_db(x, noise_db=-60, max_db=35):
    return (torchaudio.transforms.AmplitudeToDB()(x) - noise_db) / (
        max_db - noise_db
    ) * 2 - 1


def amplitude(x, noise_db=-60, max_db=35, trim_noise=False):
    if trim_noise:
        x_db = (x + 1) / 2 * (max_db - noise_db) + noise_db
        if type(x) is np.ndarray:
            return 10 ** (x_db / 10) * (x_db > noise_db).astype(np.float32)
        else:
            return 10 ** (x_db / 10) * (x_db > noise_db).float()
    else:
        return 10 ** (((x + 1) / 2 * (max_db - noise_db) + noise_db) / 10)


def wave2spec(
    wave,
    n_fft=256,
    wave_fr=16000,
    spec_fr=125,
    noise_db=-60,
    max_db=35,
    to_db=True,
    power=2,
):
    if to_db:
        return (
            torchaudio.transforms.AmplitudeToDB()(
                torchaudio.transforms.Spectrogram(
                    n_fft * 2 - 1,
                    win_length=n_fft * 2 - 1,
                    hop_length=int(wave_fr / spec_fr),
                    power=power,
                )(wave)
            )
            .clamp(min=noise_db, max=max_db)
            .transpose(-2, -1)
            - noise_db
        ) / (max_db - noise_db) * 2 - 1
    else:
        return torchaudio.transforms.Spectrogram(
            n_fft * 2 - 1,
            win_length=n_fft * 2 - 1,
            hop_length=int(wave_fr / spec_fr),
            power=power,
        )(wave).transpose(-2, -1)


def mel_scale(n_mels, hz, min_octave=-31.0, max_octave=102.0, pt=True):
    if pt:
        return (
            (torch.log2(hz / 440.0) - min_octave / 24.0)
            * 24
            * n_mels
            / (max_octave - min_octave)
        )
    else:
        return (
            (np.log2(hz / 440.0) - min_octave / 24.0)
            * 24
            * n_mels
            / (max_octave - min_octave)
        )


def inverse_mel_scale(mel, min_octave=-31.0, max_octave=102.0):
    return 440 * 2 ** (mel * (max_octave - min_octave) / 24.0 + min_octave / 24.0)


def ind2hz(ind, n_fft, max_freq=8000.0):
    # input abs ind, output abs hz
    return ind / (1.0 * n_fft) * max_freq


def hz2ind(hz, n_fft, max_freq=8000.0):
    return hz / (1.0 * max_freq) * n_fft


def bandwidth_mel(freqs_hz, bandwidth_hz, n_mels):
    bandwidth_upper = freqs_hz + bandwidth_hz / 2.0
    bandwidth_lower = torch.clamp(freqs_hz - bandwidth_hz / 2.0, min=1)
    bandwidth = mel_scale(n_mels, bandwidth_upper) - mel_scale(n_mels, bandwidth_lower)
    return bandwidth


def torch_P2R(radii, angles):
    return radii * torch.cos(angles), radii * torch.sin(angles)


def inverse_spec_to_audio(
    spec, n_fft=511, win_length=511, hop_length=128, power_synth=True
):
    """
    generate random phase, then use istft to inverse spec to audio
    """
    window = torch.hann_window(win_length)
    angles = torch.randn_like(spec).uniform_(
        0, np.pi * 2
    )  # torch.zeros_like(spec)#torch.randn_like(spec).uniform_(0, np.pi*2)
    spec = spec**0.5 if power_synth else spec
    spec_complex = torch.stack(
        torch_P2R(spec, angles), dim=-1
    )  # real and image in same dim
    return torch.istft(
        spec_complex,
        n_fft=n_fft,
        window=window,
        center=True,
        win_length=win_length,
        hop_length=hop_length,
    )


def h_poly_helper(tt):
    A = torch.tensor(
        [
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1],
        ],
        dtype=tt[-1].dtype,
    )
    return torch.matmul(tt, A.transpose(0, 1))


def h_poly(t):
    tt = torch.stack([torch.ones_like(t), t, t**2, t**3], -1)
    return h_poly_helper(tt)


def H_poly(t):
    tt = [None for _ in range(4)]
    tt[0] = t
    for i in range(1, 4):
        tt[i] = tt[i - 1] * t * i / (i + 1)
    return h_poly_helper(tt)


def interp(y, xs):
    y = y[0].squeeze()
    y = F.pad(y, [1, 1])
    x = torch.linspace(-1.0, 1.0, y.shape[0] - 2)
    x = torch.cat([x[[0]] + x[0] - x[1], x, x[[-1]] + x[-1] - x[-2]])
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    I = torch.searchsorted(x[1:], xs)
    x = torch.cat([x, x[-1:] + 1])
    y = F.pad(y, [0, 1])
    m = F.pad(m, [0, 1])
    dx = x[I + 1] - x[I]
    t = (xs - x[I]) / dx
    hh = h_poly(t)
    legal_range = (t >= 0).float() * (I < x.shape[0] - 2).float()
    yy = torch.stack([y[I], m[I] * dx, y[I + 1], m[I + 1] * dx], -1)
    return ((hh * yy).sum(-1)) * legal_range


class Comp_Spec_CNN_learnx(nn.Module):
    def __init__(self, n_fft=20, hidden_dim=1):
        super(Comp_Spec_CNN_learnx, self).__init__()
        self.conv1 = ln.Conv1d(hidden_dim, 4, 1, 1, 0)
        self.norm1 = nn.GroupNorm(2, hidden_dim)
        self.conv2 = ln.Conv1d(4, 8, 1, 1, 0)
        self.norm2 = nn.GroupNorm(4, 4)
        self.conv3 = ln.Conv1d(8, 16, 1, 1, 0)
        self.norm3 = nn.GroupNorm(4, 8)
        self.conv4 = ln.Conv1d(16, 32, 1, 1, 0)
        self.norm4 = nn.GroupNorm(4, 16)
        self.conv5 = ln.Conv1d(32, n_fft, 1, 1, 0)
        self.norm5 = nn.GroupNorm(4, 32)
        self.drop = nn.Dropout(p=0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.leaky_relu(x))
        x = self.conv3(F.leaky_relu(x))
        x = self.conv4(F.leaky_relu(x))
        x = self.conv5(F.leaky_relu(x))
        return x.permute(0, 2, 1).unsqueeze(1)


class Comp_Spec_CNN_learnbandwidth(nn.Module):
    def __init__(self, n_fft=1, hidden_dim=6):
        super(Comp_Spec_CNN_learnbandwidth, self).__init__()
        self.conv1 = ln.Conv1d(hidden_dim, 4, 1, 1, 0)
        self.norm1 = nn.GroupNorm(2, hidden_dim)
        self.conv2 = ln.Conv1d(4, 8, 1, 1, 0)
        self.norm2 = nn.GroupNorm(4, 4)
        self.conv3 = ln.Conv1d(8, 16, 1, 1, 0)
        self.norm3 = nn.GroupNorm(4, 8)
        self.conv4 = ln.Conv1d(16, 8, 1, 1, 0)
        self.norm4 = nn.GroupNorm(4, 16)
        self.conv5 = ln.Conv1d(8, n_fft, 1, 1, 0)
        self.norm5 = nn.GroupNorm(4, 8)
        self.drop = nn.Dropout(p=0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.leaky_relu(x))
        x = self.conv3(F.leaky_relu(x))
        x = self.conv4(F.leaky_relu(x))
        x = F.softplus(self.conv5(F.leaky_relu(x)))
        return x


class Comp_Spec_CNN_learnx_fromspec(nn.Module):
    def __init__(self, n_fft=20, hidden_dim=1, reverse_order=1):
        # for both harmonic and noise
        super(Comp_Spec_CNN_learnx_fromspec, self).__init__()
        self.reverse_order = reverse_order
        if reverse_order:
            self.conv1 = ln.Conv1d(hidden_dim, 4, 1, 1, 0)
            self.norm1 = nn.GroupNorm(2, hidden_dim)
            self.conv2 = ln.Conv1d(4, 8, 1, 1, 0)
            self.norm2 = nn.GroupNorm(4, 4)
            self.conv3 = ln.Conv1d(8, 16, 1, 1, 0)
            self.norm3 = nn.GroupNorm(4, 8)
            self.conv4 = ln.Conv1d(16, 32, 1, 1, 0)
            self.norm4 = nn.GroupNorm(4, 16)
            self.conv5 = ln.Conv1d(32, n_fft, 1, 1, 0)
            self.norm5 = nn.GroupNorm(4, 32)
        else:
            self.conv1 = ln.Conv1d(hidden_dim, 256, 1, 1, 0)
            self.norm1 = nn.GroupNorm(32, hidden_dim)
            self.conv2 = ln.Conv1d(256, 128, 1, 1, 0)
            self.norm2 = nn.GroupNorm(32, 256)
            self.conv3 = ln.Conv1d(128, 64, 1, 1, 0)
            self.norm3 = nn.GroupNorm(32, 128)
            self.conv5 = ln.Conv1d(64, n_fft, 1, 1, 0)
            self.norm5 = nn.GroupNorm(16, 64)

        self.drop = nn.Dropout(p=0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.leaky_relu(x))
        x = self.conv3(F.leaky_relu(x))
        x = self.conv4(F.leaky_relu(x))
        x = self.conv5(F.leaky_relu(x))
        return x.permute(0, 2, 1).unsqueeze(1)


class LearntFormantFilter(nn.Module):
    def __init__(
        self,
        nfft,
        nfilter=50,
        max_filter_freq=2000.0,
        max_frequency=8000.0,
        cubic=False,
        noise=False,
        dynamic=False,
        learnedbandwidth=False,
        temporal_invariant=True,
        reverse_order=True,
    ):
        super(LearntFormantFilter, self).__init__()
        self.max_frequency = max_frequency
        self.nfft = nfft
        self.nfilter = int(nfilter)
        self.max_filter_freq = max_filter_freq
        self.cubic = cubic
        self.noise = noise
        self.dynamic = dynamic
        self.learnedbandwidth = learnedbandwidth
        self.temporal_invariant = temporal_invariant
        if dynamic:
            self.net_learnx = Comp_Spec_CNN_learnx_fromspec(
                hidden_dim=nfft, n_fft=self.nfilter, reverse_order=reverse_order
            )
        else:
            self.x = torch.nn.Parameter(
                torch.ones(1, 1, 1, self.nfilter)
            )

        if learnedbandwidth:
            self.net_learn_band = Comp_Spec_CNN_learnbandwidth(hidden_dim=1)

    def forward(self, fi, bi, f_sampling=None, spec=None):
        if self.dynamic:
            self.x = self.net_learnx(spec)
            if self.temporal_invariant:
                self.x = self.x.max(-2)[0]
                self.x = self.x.unsqueeze(dim=-2)
        x1 = F.softmax(self.x[:, :, :, : self.nfilter // 2], dim=-1)
        x2 = F.softmax(self.x[:, :, :, self.nfilter // 2 :], dim=-1)
        x1_cum = torch.cumsum(x1, dim=-1)
        x2_cum = torch.flip(torch.cumsum(x2, dim=-1), [-1])
        x_cum = torch.cat((x1_cum, x2_cum), -1)
        if self.learnedbandwidth and not self.noise:
            bi = self.net_learn_band(fi)
        self.x_cum = x_cum
        x1_3db_value = x1_cum.max() / np.sqrt(2.0)
        _, x1_3db_idxs = torch.abs((x1_cum - x1_3db_value)).topk(2, -1, False)
        x1_3db_values0 = (x1_cum[0, 0, 0, x1_3db_idxs[0, 0, 0, 0]] - x1_3db_value).abs()
        x1_3db_values1 = (x1_cum[0, 0, 0, x1_3db_idxs[0, 0, 0, 1]] - x1_3db_value).abs()
        x1_3db_idx = x1_3db_idxs[0, 0, 0, 0].float() + x1_3db_values0 / (
            x1_3db_values0 + x1_3db_values1
        ) * (x1_3db_idxs[0, 0, 0, 1].float() - x1_3db_idxs[0, 0, 0, 0].float())
        x2_3db_value = x2_cum.max() / np.sqrt(2.0)
        _, x2_3db_idxs = torch.abs((x2_cum - x2_3db_value)).topk(2, -1, False)
        x2_3db_values0 = (x2_cum[0, 0, 0, x2_3db_idxs[0, 0, 0, 0]] - x2_3db_value).abs()
        x2_3db_values1 = (x2_cum[0, 0, 0, x2_3db_idxs[0, 0, 0, 1]] - x2_3db_value).abs()
        x2_3db_idx = x2_3db_idxs[0, 0, 0, 0].float() + x2_3db_values0 / (
            x2_3db_values0 + x2_3db_values1
        ) * (x2_3db_idxs[0, 0, 0, 1].float() - x2_3db_idxs[0, 0, 0, 0].float())
        b_x = (
            (self.nfilter // 2 - 1 - x1_3db_idx + x2_3db_idx).float()
            / self.nfilter
            * self.max_filter_freq
        )
        if f_sampling is None:
            grid_x, grid_y = torch.meshgrid(
                torch.arange(self.nfft).float(), torch.zeros(fi.shape[2])
            )
            grid_x = grid_x.unsqueeze(0)  # 1 x nfft x T
            grid_y = grid_y.unsqueeze(0)
        else:
            grid_x = f_sampling / self.max_frequency * self.nfft
            grid_y = torch.zeros_like(grid_x)
        a = (2 * b_x / self.max_filter_freq) / (
            (bi + 1e-8) / self.max_frequency * self.nfft
        )
        b = -2 * b_x / self.max_filter_freq * fi / (bi + 1e-8)
        grid_x = grid_x * a + b
        grid = torch.stack([grid_x, grid_y.repeat([fi.shape[0], 1, 1])], -1)
        if not self.dynamic:
            x_cum = x_cum.repeat(
                [fi.shape[0], 1, 1, 1]
            )
        if self.cubic:
            sampled_filter = interp(x_cum, grid_x)
        else:
            sampled_filter = F.grid_sample(
                x_cum, grid, align_corners=None
            )
        sampled_filter = sampled_filter.transpose(-1, -2)
        return sampled_filter


class LearntFormantFilterMultiple(nn.Module):
    def __init__(
        self,
        n_formants,
        n_formants_noise,
        nfft,
        nfilter=50,
        max_filter_freq=2000.0,
        max_frequency=8000.0,
        dynamic=False,
        learnedbandwidth=False,
        temporal_invariant=True,
        reverse_order=True,
    ):
        super(LearntFormantFilterMultiple, self).__init__()
        self.n_formants = n_formants
        self.n_formants_noise = n_formants_noise
        self.filter = nn.ModuleList([])
        self.temporal_invariant = temporal_invariant
        self.dynamic = dynamic
        for i in range(n_formants):
            self.filter.append(
                LearntFormantFilter(
                    nfft=nfft,
                    nfilter=nfilter,
                    max_filter_freq=max_filter_freq,
                    max_frequency=max_frequency,
                    noise=False,
                    dynamic=dynamic,
                    learnedbandwidth=learnedbandwidth,
                    temporal_invariant=temporal_invariant,
                    reverse_order=reverse_order,
                )
            )

        self.filter_noise = nn.ModuleList([])
        for i in range(n_formants_noise):
            self.filter_noise.append(
                LearntFormantFilter(
                    nfft=nfft,
                    nfilter=nfilter,
                    max_filter_freq=max_filter_freq,
                    max_frequency=max_frequency,
                    noise=True,
                    dynamic=dynamic,
                    learnedbandwidth=learnedbandwidth,
                    temporal_invariant=temporal_invariant,
                    reverse_order=reverse_order,
                )
            )

    def forward(self, fi, bi, f_sampling=None, spec=None):
        total_formants = fi.shape[-1]
        is_harm = True if (total_formants == self.n_formants) else False
        fi = torch.unbind(fi, -1)
        bi = torch.unbind(bi, -1)
        if f_sampling is not None:
            f_sampling = f_sampling.squeeze(-1).transpose(-1, -2)
        combined_filter = torch.stack(
            [
                self.filter[i](
                    fi[i].transpose(-1, -2),
                    bi[i].transpose(-1, -2),
                    f_sampling=f_sampling,
                    spec=spec,
                )
                for i in range(self.n_formants)
            ],
            -1,
        )
        combined_filter = combined_filter.squeeze(1)
        if not is_harm:
            combined_filter_noise = torch.stack(
                [
                    self.filter_noise[i](
                        fi[i + self.n_formants].transpose(-1, -2),
                        bi[i + self.n_formants].transpose(-1, -2),
                        f_sampling=f_sampling,
                        spec=spec,
                    )
                    for i in range(self.n_formants_noise)
                ],
                -1,
            )
            combined_filter_noise = combined_filter_noise.squeeze(
                1
            )
            combined_filter = torch.cat([combined_filter, combined_filter_noise], -1)
        return combined_filter


class DoubleConv(nn.Module):
    """(convolution => [norm] => LeakyReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        groups=16,
        causal=0,
        anticausal=0,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ln_c.Conv1d(
                in_channels, mid_channels, 3, 1, 1, causal=causal, anticausal=anticausal
            ),
            GroupNormXDim(groups, mid_channels),
            nn.LeakyReLU(),
            ln_c.Conv1d(
                mid_channels,
                out_channels,
                3,
                1,
                1,
                causal=causal,
                anticausal=anticausal,
            ),
            GroupNormXDim(groups, out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Upsample_Block(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, bilinear=False, causal=False, anticausal=False
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = ln_c.ConvTranspose1d(
                in_channels,
                out_channels,
                3,
                2,
                1,
                transform_kernel=True,
                causal=causal,
                anticausal=anticausal,
            )
    def forward(self, x):
        return self.up(x)


class GroupNormXDim(nn.Module):
    def __init__(self, num_groups, num_channels, **kargrs):
        super(GroupNormXDim, self).__init__()
        self.norm = nn.GroupNorm(1, num_channels, **kargrs)

    def forward(self, x):
        x = x.permute([0, 2, 1] + list(range(3, x.ndim)))
        szs = x.shape
        x = x.reshape([szs[0] * szs[1], *szs[2:]])
        x = self.norm(x)
        x = x.reshape(szs)
        x = x.permute([0, 2, 1] + list(range(3, x.ndim)))
        return x


class BatchNormXDim(nn.Module):
    def __init__(self, num_groups, num_channels, **kargrs):
        super(BatchNormXDim, self).__init__()
        self.norm1d = nn.BatchNorm1d(num_channels, **kargrs)
        self.norm2d = nn.BatchNorm2d(num_channels, **kargrs)
        self.norm3d = nn.BatchNorm3d(num_channels, **kargrs)

    def forward(self, x):
        szs = x.shape
        x = x.reshape([szs[0], szs[1] * szs[2], *szs[3:], 1])
        if x.dim == 3:
            x = self.norm1d(x)
        if x.dim == 4:
            x = self.norm2d(x)
        if x.dim == 5:
            x = self.norm3d(x)
        x = x.reshape(szs)
        return x


@GENERATORS.register("GeneratorFormant")
class FormantSysth(nn.Module):
    def __init__(
        self,
        n_mels=64,
        k=100,
        wavebased=False,
        n_fft=256,
        noise_db=-50,
        max_db=22.5,
        dbbased=False,
        add_bgnoise=True,
        log10=False,
        noise_from_data=False,
        return_wave=False,
        power_synth=False,
        n_formants=6,
        normed_mask=False,
        dummy_formant=False,
        learned_mask=False,
        n_filter_samples=40,
        dynamic_filter_shape=False,
        learnedbandwidth=False,
        return_filtershape=False,
        spec_fr=125,
        reverse_order=True,
        quantfilename=None,
        #'/scratch/xc1490/projects/ecog/ALAE_1023/neural_speech_decoding/data/models/869model/NY869_kmeans_model_Cluster100_Window2_6formants.pkl'
    ):
        super(FormantSysth, self).__init__()
        self.wave_fr = 16e3
        self.spec_fr = spec_fr
        self.n_fft = n_fft
        self.noise_db = noise_db
        self.max_db = max_db
        self.n_mels = n_mels
        self.k = k
        self.dbbased = dbbased
        self.log10 = log10
        self.add_bgnoise = add_bgnoise
        self.wavebased = wavebased
        self.noise_from_data = noise_from_data
        self.linear_scale = wavebased
        self.return_wave = return_wave
        self.power_synth = power_synth
        self.n_formants = n_formants
        self.normed_mask = normed_mask
        self.dummy_formant = dummy_formant
        self.learnedbandwidth = learnedbandwidth
        self.dynamic_filter_shape = dynamic_filter_shape
        self.timbre = Parameter(torch.Tensor(1, 1, n_mels))
        self.return_filtershape = return_filtershape
        self.timbre_mapping = nn.Sequential(
            ln.Conv1d(1, 128, 1),
            nn.LeakyReLU(0.2),
            ln.Conv1d(128, 128, 1),
            nn.LeakyReLU(0.2),
            ln.Conv1d(128, 2, 1),
        )
        self.bgnoise_mapping = nn.Sequential(
            ln.Conv2d(2, 2, [1, 5], padding=[0, 2], gain=1, bias=False),
        )
        self.noise_mapping = nn.Sequential(
            ln.Conv2d(2, 2, [1, 5], padding=[0, 2], gain=1, bias=False),
        )

        self.bgnoise_mapping2 = nn.Sequential(
            ln.Conv1d(1, 128, 1, 1),
            nn.LeakyReLU(0.2),
            ln.Conv1d(128, 128, 1, 1),
            nn.LeakyReLU(0.2),
            ln.Conv1d(128, 1, 1, gain=1, bias=False),
        )
        self.noise_mapping2 = nn.Sequential(
            ln.Conv1d(1, 128, 1, 1),
            nn.LeakyReLU(0.2),
            ln.Conv1d(128, 128, 1, 1),
            nn.LeakyReLU(0.2),
            ln.Conv1d(128, 1, 1, 1, gain=1, bias=False),
        )
        self.prior_exp = np.array([0.4963, 0.0745, 1.9018])
        self.timbre_parameter = Parameter(torch.Tensor(2))
        self.wave_noise_amplifier = Parameter(torch.Tensor(1))
        self.wave_hamon_amplifier = Parameter(torch.Tensor(1))

        if noise_from_data:
            self.bgnoise_amp = Parameter(torch.Tensor(1))
            with torch.no_grad():
                nn.init.constant_(self.bgnoise_amp, 1)
        else:
            self.bgnoise_dist = Parameter(
                torch.Tensor(1, 1, 1, self.n_fft if self.wavebased else self.n_mels)
            )
            with torch.no_grad():
                nn.init.constant_(self.bgnoise_dist, 1.0)
        self.silient = -1
        if self.dummy_formant:
            self.dummy_f = Parameter(torch.Tensor(1, 1, 1, 1))
            self.dummy_w = Parameter(torch.Tensor(1, 1, 1, 1))
            self.dummy_a = Parameter(torch.Tensor(1, 1, 1, 1))
        with torch.no_grad():
            nn.init.constant_(self.timbre, 1.0)
            nn.init.constant_(self.timbre_parameter[0], 7)
            nn.init.constant_(self.timbre_parameter[1], 0.004)
            nn.init.constant_(self.wave_noise_amplifier, 1)
            nn.init.constant_(self.wave_hamon_amplifier, 4.0)
            if self.dummy_formant:
                nn.init.constant_(self.dummy_f, 1.0)
                nn.init.constant_(self.dummy_w, 1.0)
                nn.init.constant_(self.dummy_a, -1.0)

        self.learned_mask = learned_mask
        self.quantfilename = quantfilename
        
        if learned_mask:
            if learnedbandwidth or dynamic_filter_shape:
                self.formant_masks_hamon = LearntFormantFilterMultiple(
                    n_formants,
                    0,
                    n_fft,
                    n_filter_samples,
                    2000.0,
                    8000.0,
                    dynamic=dynamic_filter_shape,
                    learnedbandwidth=learnedbandwidth,
                    reverse_order=reverse_order,
                )
                self.formant_masks_noise = LearntFormantFilterMultiple(
                    n_formants,
                    1,
                    n_fft,
                    n_filter_samples,
                    2000.0,
                    8000.0,
                    dynamic=dynamic_filter_shape,
                    learnedbandwidth=learnedbandwidth,
                    reverse_order=reverse_order,
                )
            else:
                self.formant_masks = LearntFormantFilterMultiple(
                    n_formants,
                    1,
                    n_fft,
                    n_filter_samples,
                    2000.0,
                    8000.0,
                    reverse_order=reverse_order,
                )
            if dummy_formant:
                self.formant_masks_dummy = LearntFormantFilterMultiple(
                    1,
                    0,
                    n_fft,
                    n_filter_samples,
                    2000.0,
                    8000.0,
                    dynamic=False,
                    reverse_order=reverse_order,
                )
        if quantfilename is not '':
            print ('performing quantization')
            self.quantizer = self.quantizer_model(quantfilename)
        
        
    def formant_mask(
        self,
        freq_hz,
        bandwith_hz,
        amplitude,
        linear=False,
        triangle_mask=False,
        duomask=True,
        n_formant_noise=1,
        f0_hz=None,
        noise=False,
        learned_mask=None,
        spec=None,
    ):
        freq_cord = torch.arange(self.n_fft if linear else self.n_mels)
        time_cord = torch.arange(freq_hz.shape[2])
        grid_time, grid_freq = torch.meshgrid(time_cord, freq_cord)
        grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1)
        grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1)
        grid_freq_hz = (
            ind2hz(grid_freq, self.n_fft, self.wave_fr / 2)
            if linear
            else inverse_mel_scale(grid_freq / (self.n_mels * 1.0))
        )
        freq_hz = freq_hz.permute([0, 2, 1]).unsqueeze(dim=-2)
        bandwith_hz = bandwith_hz.permute([0, 2, 1]).unsqueeze(
            dim=-2
        ) 
        amplitude = amplitude.permute([0, 2, 1]).unsqueeze(dim=-2) 
        if self.dummy_formant:
            dummy_freq = 250 * torch.sigmoid(self.dummy_f)
            dummy_band = 250 * torch.sigmoid(self.dummy_w) + 50
            dummy_amp = torch.sigmoid(self.dummy_a)
        if self.power_synth:
            amplitude = amplitude
        alpha = 2 * np.sqrt(2 * np.log(np.sqrt(2)))

        if self.return_wave:
            t = torch.arange(int(f0_hz.shape[2] / self.spec_fr * self.wave_fr)) / (
                1.0 * self.wave_fr
            )
            t = t.unsqueeze(dim=0).unsqueeze(dim=0)
            k = (torch.arange(self.k) + 1).reshape([1, self.k, 1])
            k_f0 = k * f0_hz
            k_f0 = k_f0.permute([0, 2, 1]).unsqueeze(-1)
            if self.learned_mask:
                hamonic_dist = (
                    (
                        amplitude
                        * learned_mask(freq_hz, bandwith_hz + 0.01, k_f0, spec=spec)
                    )
                    .sqrt()
                    .sum(-1)
                    .permute([0, 2, 1])
                )
            else:
                hamonic_dist = (
                    (
                        amplitude
                        * torch.exp(
                            -(((k_f0 - freq_hz)) ** 2)
                            / (2 * (bandwith_hz / alpha + 0.01) ** 2)
                        )
                    )
                    .sqrt()
                    .sum(-1)
                    .permute([0, 2, 1])
                )
            if self.dummy_formant:
                if self.learned_mask:
                    dummy = dummy_amp * self.formant_masks_dummy(
                        dummy_freq, dummy_band, k_f0, spec=spec
                    ).sum(-1).permute(
                        [0, 2, 1]
                    )
                else:
                    dummy = dummy_amp * torch.exp(
                        -0.5
                        * (
                            ((k_f0 - dummy_freq)) ** 2
                            / (
                                (dummy_band / (2 * (2 * np.log(2)) ** (0.5)) + 0.01)
                                ** 2
                            )
                            + 1e-8
                        )
                    ).sum(-1).permute(
                        [0, 2, 1]
                    )
                hamonic_dist = hamonic_dist + dummy.squeeze(0)
            hamonic_dist = F.interpolate(
                hamonic_dist,
                int(f0_hz.shape[2] / self.spec_fr * self.wave_fr),
                mode="linear",
                align_corners=False,
            )

        if triangle_mask:
            if duomask:
                if self.learned_mask:
                    masks_hamon = amplitude[..., :-n_formant_noise] * learned_mask(
                        freq_hz[..., :-n_formant_noise],
                        bandwith_hz[..., :-n_formant_noise] + 0.01,
                        spec=spec,
                    )
                    bw = bandwith_hz[..., -n_formant_noise:]
                    masks_noise = amplitude[..., -n_formant_noise:] * learned_mask(
                        freq_hz[..., -n_formant_noise:] + 0.01, bw, spec=spec
                    )
                else:
                    masks_hamon = amplitude[..., :-n_formant_noise] * torch.exp(
                        -(((grid_freq_hz - freq_hz[..., :-n_formant_noise])) ** 2)
                        / (
                            2
                            * (bandwith_hz[..., :-n_formant_noise] / alpha + 0.01) ** 2
                        )
                    )
                    bw = bandwith_hz[..., -n_formant_noise:]
                    masks_noise = F.relu(
                        amplitude[..., -n_formant_noise:]
                        * (
                            1
                            - (1 - 1 / np.sqrt(2))
                            * 2
                            / (bw + 0.01)
                            * (grid_freq_hz - freq_hz[..., -n_formant_noise:]).abs()
                        )
                    )
                masks = torch.cat([masks_hamon, masks_noise], dim=-1)
            else:
                if self.learned_mask:
                    masks_hamon = amplitude * learned_mask(
                        freq_hz, bandwith_hz + 0.01, spec=spec
                    )
                else:
                    masks = F.relu(
                        amplitude
                        * (
                            1
                            - (1 - 1 / np.sqrt(2))
                            * 2
                            / (bandwith_hz + 0.01)
                            * (grid_freq_hz - freq_hz).abs()
                        )
                    )
        else:
            if self.power_synth:
                if self.learned_mask:
                    masks = amplitude * learned_mask(
                        freq_hz, bandwith_hz + 0.01, spec=spec
                    )
                else:
                    masks = amplitude * torch.exp(
                        -(((grid_freq_hz - freq_hz)) ** 2)
                        / (2 * (bandwith_hz / alpha + 0.01) ** 2)
                    )
                if self.dummy_formant and not noise:
                    if self.learned_mask:
                        dummy = dummy_amp * self.formant_masks_dummy(
                            dummy_freq, dummy_band + 0.01, spec=spec
                        )
                    else:
                        dummy = dummy_amp * torch.exp(
                            -0.5
                            * (
                                ((grid_freq_hz - dummy_freq)) ** 2
                                / (
                                    (dummy_band / (2 * (2 * np.log(2)) ** (0.5)) + 0.01)
                                    ** 2
                                )
                                + 1e-8
                            )
                        )
                    masks = torch.cat(
                        [dummy.expand(masks.shape[0], masks.shape[1], -1, -1), masks],
                        -1,
                    )
            else:
                if self.learned_mask:
                    masks = (
                        amplitude
                        * learned_mask(freq_hz, bandwith_hz + 0.01, spec=spec).sqrt()
                    )
                else:
                    masks = (
                        amplitude
                        * (
                            torch.exp(
                                -(((grid_freq_hz - freq_hz)) ** 2)
                                / (2 * (bandwith_hz / alpha + 0.01) ** 2)
                            )
                            + 1e-6
                        ).sqrt()
                    )
        masks = masks.unsqueeze(dim=1)
        if self.return_wave:
            return masks, hamonic_dist
        else:
            if self.return_filtershape:
                if self.learned_mask:
                    return masks, learned_mask(freq_hz, bandwith_hz + 0.01)
                else:
                    return masks
            else:
                return masks

    def voicing_wavebased(self, f0_hz):
        t = torch.arange(int(f0_hz.shape[2] / self.spec_fr * self.wave_fr)) / (
            1.0 * self.wave_fr
        )
        t = t.unsqueeze(dim=0).unsqueeze(dim=0)  # 1, 1, time
        k = (torch.arange(self.k) + 1).reshape([1, self.k, 1])
        f0_hz_interp = F.interpolate(
            f0_hz, t.shape[-1], mode="linear", align_corners=False
        )
        k_f0 = k * f0_hz_interp
        k_f0_sum = 2 * np.pi * torch.cumsum(k_f0, -1) / (1.0 * self.wave_fr)
        wave_k = (
            np.sqrt(2) * torch.sin(k_f0_sum) * (-torch.sign(k_f0 - 7800) * 0.5 + 0.5)
        )
        wave = wave_k.sum(dim=1, keepdim=True)
        spec = wave2spec(
            wave,
            self.n_fft,
            self.wave_fr,
            self.spec_fr,
            self.noise_db,
            self.max_db,
            to_db=self.dbbased,
            power=2.0 if self.power_synth else 1.0,
        )
        if self.return_wave:
            return spec, wave_k
        else:
            return spec

    def unvoicing_wavebased(self, f0_hz, bg=False, mapping=True):
        if bg:
            noise = torch.randn(
                [1, 1, int(f0_hz.shape[2] / self.spec_fr * self.wave_fr)]
            )
            if mapping:
                noise = self.bgnoise_mapping2(noise)
        else:
            noise = np.sqrt(3.0) * (
                2
                * torch.rand([1, 1, int(f0_hz.shape[2] / self.spec_fr * self.wave_fr)])
                - 1
            )
            if mapping:
                noise = self.noise_mapping2(noise)
        return wave2spec(
            noise,
            self.n_fft,
            self.wave_fr,
            self.spec_fr,
            self.noise_db,
            self.max_db,
            to_db=self.dbbased,
            power=2.0 if self.power_synth else 1.0,
        )

    def voicing_linear(self, f0_hz, bandwith=2.5):
        freq_cord = torch.arange(self.n_fft)
        time_cord = torch.arange(f0_hz.shape[2])
        grid_time, grid_freq = torch.meshgrid(time_cord, freq_cord)
        grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1)
        grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1)
        f0_hz = f0_hz.permute([0, 2, 1]).unsqueeze(dim=-2)
        f0_hz = f0_hz.repeat([1, 1, 1, self.k])
        f0_hz = f0_hz * (torch.arange(self.k) + 1).reshape([1, 1, 1, self.k])
        f0 = hz2ind(f0_hz, self.n_fft)
        freq_cord_reshape = freq_cord.reshape([1, 1, 1, self.n_fft])
        hamonics = (1 - 2 / bandwith * (grid_freq - f0).abs()) * (
            -torch.sign(torch.abs(grid_freq - f0) / (bandwith) - 0.5) * 0.5 + 0.5
        )
        hamonics = (hamonics.sum(dim=-1)).unsqueeze(dim=1)
        return hamonics

    def voicing(self, f0_hz):
        freq_cord = torch.arange(self.n_mels)
        time_cord = torch.arange(f0_hz.shape[2])
        grid_time, grid_freq = torch.meshgrid(time_cord, freq_cord)
        grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) 
        grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1)
        grid_freq_hz = inverse_mel_scale(grid_freq / (self.n_mels * 1.0))
        f0_hz = f0_hz.permute([0, 2, 1]).unsqueeze(dim=-2)
        f0_hz = f0_hz.repeat([1, 1, 1, self.k]) 
        f0_hz = f0_hz * (torch.arange(self.k) + 1).reshape([1, 1, 1, self.k])
        if self.log10:
            f0_mel = mel_scale(self.n_mels, f0_hz)
            band_low_hz = inverse_mel_scale(
                (f0_mel - 1) / (self.n_mels * 1.0), n_mels=self.n_mels
            )
            band_up_hz = inverse_mel_scale(
                (f0_mel + 1) / (self.n_mels * 1.0), n_mels=self.n_mels
            )
            bandwith_hz = band_up_hz - band_low_hz
            band_low_mel = mel_scale(self.n_mels, band_low_hz)
            band_up_mel = mel_scale(self.n_mels, band_up_hz)
            bandwith = band_up_mel - band_low_mel
        else:
            bandwith_hz = 24.7 * (f0_hz * 4.37 / 1000 + 1)
            bandwith = bandwidth_mel(f0_hz, bandwith_hz, self.n_mels)
        f0 = mel_scale(self.n_mels, f0_hz)
        switch = mel_scale(
            self.n_mels, torch.abs(self.timbre_parameter[0]) * f0_hz[..., 0]
        ).unsqueeze(1)
        slop = (torch.abs(self.timbre_parameter[1]) * f0_hz[..., 0]).unsqueeze(1)
        freq_cord_reshape = freq_cord.reshape([1, 1, 1, self.n_mels])
        if not self.dbbased:
            sigma = bandwith / (2 * np.sqrt(2 * np.log(2)))
            hamonics = torch.exp(
                -((grid_freq - f0) ** 2) / (2 * sigma**2)
            ) 
        else:
            hamonics = (1 - ((grid_freq - f0) / (2.5 * bandwith / 2)) ** 2) * (
                -torch.sign(torch.abs(grid_freq - f0) / (2.5 * bandwith) - 0.5) * 0.5
                + 0.5
            )
        timbre_parameter = (
            self.timbre_mapping(f0_hz[..., 0, 0].unsqueeze(1))
            .permute([0, 2, 1])
            .unsqueeze(1)
        )
        condition = (
            torch.sign(
                freq_cord_reshape
                - torch.sigmoid(timbre_parameter[..., 0:1]) * self.n_mels
            )
            * 0.5
            + 0.5
        )
        amp = (
            F.softplus(self.wave_hamon_amplifier)
            if self.dbbased
            else 180 * F.softplus(self.wave_hamon_amplifier)
        )
        hamonics = (
            amp
            * ((hamonics.sum(dim=-1)).unsqueeze(dim=1))
            * (
                1
                + (
                    torch.exp(
                        -0.01
                        * torch.sigmoid(timbre_parameter[..., 1:2])
                        * (
                            freq_cord_reshape
                            - torch.sigmoid(timbre_parameter[..., 0:1]) * self.n_mels
                        )
                        * condition
                    )
                    - 1
                )
                * condition
            )
        ) 
        return hamonics

    def unvoicing(self, f0, bg=False, mapping=True):
        rnd = torch.randn(
            [f0.shape[0], 2, f0.shape[2], self.n_fft if self.wavebased else self.n_mels]
        )
        if mapping:
            rnd = self.bgnoise_mapping(rnd) if bg else self.noise_mapping(rnd)
        real = rnd[:, 0:1]
        img = rnd[:, 1:2]
        if self.dbbased:
            return (
                2
                * torchaudio.transforms.AmplitudeToDB()(
                    torch.sqrt(real**2 + img**2 + 1e-10)
                )
                + 80
            ).clamp(min=0) / 35
        else:
            return (
                180
                * F.softplus(self.wave_noise_amplifier)
                * torch.sqrt(real**2 + img**2 + 1e-10)
            )

    
    def forward(
        self,
        components,
        onstage,
        enable_hamon_excitation=True,
        enable_noise_excitation=True,
        enable_bgnoise=True,
        n_iter=0,
        save_path="",
    ):

        if self.dynamic_filter_shape:
            spec_smooth = components["spec_smooth"]
        else:
            spec_smooth = None

        amplitudes = components["amplitudes"].unsqueeze(dim=-1)
        #amplitudes_h = components["amplitudes_h"].unsqueeze(dim=-1)
        loudness = components["loudness"].unsqueeze(dim=-1)
        f0_hz = components["f0_hz"]
        if not self.noise_from_data:
            noise_dist_learned = (
                components["noisebg"]
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
                .permute(0, 3, 2, 1)
            ) 
        if self.wavebased:
            self.hamonics = self.voicing_wavebased(f0_hz)
            self.noise = self.unvoicing_wavebased(f0_hz, bg=False, mapping=False)
            self.bgnoise = self.unvoicing_wavebased(f0_hz, bg=True)
        else:
            self.hamonics = self.voicing(f0_hz)
            self.noise = self.unvoicing(f0_hz, bg=False)
            self.bgnoise = self.unvoicing(f0_hz, bg=True)
        self.excitation_noise = (
            loudness * (amplitudes[:, -1:]) * self.noise
            if self.power_synth
            else loudness * amplitudes[:, -1:] * self.noise
        )
        duomask = (
            components["freq_formants_noise_hz"].shape[1]
            > components["freq_formants_hamon_hz"].shape[1]
        )
        n_formant_noise = (
            (
                components["freq_formants_noise_hz"].shape[1]
                - components["freq_formants_hamon_hz"].shape[1]
            )
            if duomask
            else components["freq_formants_noise_hz"].shape[1]
        )

        if self.learned_mask:
            if self.learnedbandwidth or self.dynamic_filter_shape:
                self.mask_hamon = self.formant_mask(
                    components["freq_formants_hamon_hz"],
                    components["bandwidth_formants_hamon_hz"],
                    components["amplitude_formants_hamon"],
                    linear=self.linear_scale,
                    f0_hz=f0_hz,
                    learned_mask=self.formant_masks_hamon,
                    spec=spec_smooth,
                )
                self.mask_noise = self.formant_mask(
                    components["freq_formants_noise_hz"],
                    components["bandwidth_formants_noise_hz"],
                    components["amplitude_formants_noise"],
                    linear=self.linear_scale,
                    triangle_mask=False if self.wavebased else True,
                    duomask=duomask,
                    n_formant_noise=n_formant_noise,
                    f0_hz=f0_hz,
                    noise=True,
                    learned_mask=self.formant_masks_noise,
                    spec=spec_smooth,
                )
            else:
                self.mask_hamon = self.formant_mask(
                    components["freq_formants_hamon_hz"],
                    components["bandwidth_formants_hamon_hz"],
                    components["amplitude_formants_hamon"],
                    linear=self.linear_scale,
                    f0_hz=f0_hz,
                    learned_mask=self.formant_masks,
                )
                self.mask_noise = self.formant_mask(
                    components["freq_formants_noise_hz"],
                    components["bandwidth_formants_noise_hz"],
                    components["amplitude_formants_noise"],
                    linear=self.linear_scale,
                    triangle_mask=False if self.wavebased else True,
                    duomask=duomask,
                    n_formant_noise=n_formant_noise,
                    f0_hz=f0_hz,
                    noise=True,
                    learned_mask=self.formant_masks,
                )
        else:
            self.mask_hamon = self.formant_mask(
                components["freq_formants_hamon_hz"],
                components["bandwidth_formants_hamon_hz"],
                components["amplitude_formants_hamon"],
                linear=self.linear_scale,
                f0_hz=f0_hz,
            )
            self.mask_noise = self.formant_mask(
                components["freq_formants_noise_hz"],
                components["bandwidth_formants_noise_hz"],
                components["amplitude_formants_noise"],
                linear=self.linear_scale,
                triangle_mask=False if self.wavebased else True,
                duomask=duomask,
                n_formant_noise=n_formant_noise,
                f0_hz=f0_hz,
                noise=True,
            )

        if n_iter != 0:
            if self.return_filtershape:
                if not os.path.exists(save_path + "/filtershape/"):
                    os.makedirs(save_path + "/filtershape/")
                np.save(
                    save_path + "/filtershape/hamon_{}".format(n_iter),
                    self.mask_hamon.detach().cpu().numpy(),
                )
                np.save(
                    save_path + "/filtershape/noise_{}".format(n_iter),
                    self.mask_noise.detach().cpu().numpy(),
                )

        if self.return_wave:
            self.hamonics, self.hamonics_wave = self.hamonics
            self.mask_hamon, self.hamonic_dist = self.mask_hamon
            self.mask_noise, self.mask_noise_only = self.mask_noise
            if self.power_synth:
                self.excitation_hamon_wave = (
                    F.interpolate(
                        (loudness[..., -1] * amplitudes[:, 0:1][..., -1]).sqrt(),
                        self.hamonics_wave.shape[-1],
                        mode="linear",
                        align_corners=False,
                    )
                    * self.hamonics_wave
                )
            else:
                self.excitation_hamon_wave = (
                    F.interpolate(
                        loudness[..., -1] * amplitudes[:, 0:1][..., -1],
                        self.hamonics_wave.shape[-1],
                        mode="linear",
                        align_corners=False,
                    )
                    * self.hamonics_wave
                )
            self.hamonics_wave_ = (self.excitation_hamon_wave * self.hamonic_dist).sum(
                1, keepdim=True
            )
        self.mask_hamon_sum = self.mask_hamon.sum(dim=-1)
        self.mask_noise_sum = self.mask_noise.sum(dim=-1)
        bgdist = (
            F.softplus(self.bgnoise_amp) * self.noise_dist
            if self.noise_from_data
            else noise_dist_learned
        )
        if self.power_synth:
            self.excitation_hamon = loudness * (amplitudes[:, 0:1]) * self.hamonics
        else:
            self.excitation_hamon = loudness * amplitudes[:, 0:1] * self.hamonics
        self.noise_excitation = self.excitation_noise * self.mask_noise_sum
        if self.return_wave:
            self.noise_excitation_wave = 2 * inverse_spec_to_audio(
                self.noise_excitation.squeeze(1).permute(0, 2, 1),
                n_fft=self.n_fft * 2 - 1,
                power_synth=self.power_synth,
            )
            self.noise_excitation_wave = F.pad(
                self.noise_excitation_wave,
                [0, self.hamonics_wave_.shape[2] - self.noise_excitation_wave.shape[1]],
            )
            self.noise_excitation_wave = self.noise_excitation_wave.unsqueeze(1)
            self.rec_wave = self.noise_excitation_wave + self.hamonics_wave_
        if self.wavebased:
            bgn = (
                bgdist * self.bgnoise * 0.0003
                if (self.add_bgnoise and enable_bgnoise)
                else 0
            )
            if not self.noise_from_data:
                bgn = bgn * onstage.unsqueeze(-1)
            speech = (
                (
                    (self.excitation_hamon * self.mask_hamon_sum)
                    if enable_hamon_excitation
                    else torch.zeros(self.excitation_hamon.shape)
                )
                + (self.noise_excitation if enable_noise_excitation else 0)
                + bgn
            )
            speech = (
                torchaudio.transforms.AmplitudeToDB()(speech).clamp(min=self.noise_db)
                - self.noise_db
            ) / (self.max_db - self.noise_db) * 2 - 1
        else:
            speech = (
                (
                    (self.excitation_hamon * self.mask_hamon_sum)
                    if enable_hamon_excitation
                    else torch.zeros(self.excitation_hamon.shape)
                )
                + (self.noise_excitation if enable_noise_excitation else 0)
                + (
                    (
                        (bgdist * self.bgnoise * 0.0003)
                        if not self.dbbased
                        else (
                            2
                            * torchaudio.transforms.AmplitudeToDB()(bgdist * 0.0003)
                            / 35.0
                            + self.bgnoise
                        )
                    )
                    if (self.add_bgnoise and enable_bgnoise)
                    else 0
                )
                + (
                    self.silient * torch.ones(self.mask_hamon_sum.shape)
                    if self.dbbased
                    else 0
                )
            )
            if not self.dbbased:
                speech = db(speech)
        if self.return_wave:
            return speech, self.rec_wave
        else:
            return speech




@ENCODERS.register("EncoderFormant")
class FormantEncoder(nn.Module):
    def __init__(
        self,
        n_mels=64,
        n_formants=4,
        n_formants_noise=2,
        min_octave=-31,
        max_octave=96,
        wavebased=False,
        n_fft=256,
        noise_db=-50,
        max_db=22.5,
        broud=True,
        power_synth=False,
        hop_length=128,
        patient="NY742",
        gender_patient="Female",
        larger_capacity=False,
        unified=False,
    ):
        super(FormantEncoder, self).__init__()
        self.unified = unified
        self.wavebased = wavebased
        self.n_mels = n_mels
        self.n_formants = n_formants
        self.n_formants_noise = n_formants_noise
        self.min_octave = min_octave
        self.max_octave = max_octave
        self.noise_db = noise_db
        self.max_db = max_db
        self.broud = broud
        self.n_fft = n_fft
        self.power_synth = power_synth
        self.formant_freq_limits_diff = torch.tensor([950.0, 2450.0, 2100.0]).reshape(
            [1, 3, 1]
        ) 
        self.formant_freq_limits_diff_low = torch.tensor([300.0, 300.0, 0.0]).reshape(
            [1, 3, 1]
        )
        self.patient = patient
        print("patient for audio encoder:", patient)
        print("gender_patient", gender_patient)
        if unified:
            self.formant_freq_limits_abs = torch.tensor(
                [950.0, 3400.0, 3800.0, 5000.0, 6000.0, 7500.0]
            ).reshape(
                [1, 6, 1]
            )
            self.formant_freq_limits_abs_low = torch.tensor(
                [200.0, 500.0, 1400.0, 3000, 4000.0, 4500.0]
            ).reshape(
                [1, 6, 1]
            )
            self.formant_freq_limits_abs_male = torch.tensor(
                [950.0, 3400.0, 3800.0, 5000.0, 6000.0, 7500.0]
            ).reshape(
                [1, 6, 1]
            )
            self.formant_freq_limits_abs_low_male = torch.tensor(
                [200.0, 500.0, 1400.0, 3000, 4000.0, 4500.0]
            ).reshape(
                [1, 6, 1]
            )
        else:
            self.formant_freq_limits_abs = torch.tensor(
                [950.0, 3400.0, 3800.0, 5000.0, 6000.0, 7000.0]
            ).reshape(
                [1, 6, 1]
            )
            self.formant_freq_limits_abs_low = torch.tensor(
                [300.0, 700.0, 1800.0, 3400, 5000.0, 6000.0]
            ).reshape(
                [1, 6, 1]
            )
            self.formant_freq_limits_abs_male = torch.tensor(
                [850.0, 3000.0, 3400.0, 4600.0, 6000.0, 7500.0]
            ).reshape(
                [1, 6, 1]
            )
            self.formant_freq_limits_abs_low_male = torch.tensor(
                [200.0, 500.0, 1400.0, 3000, 4000.0, 4500.0]
            ).reshape(
                [1, 6, 1]
            )
        self.formant_freq_limits_abs_noise = torch.tensor(
            [8000.0, 7000.0, 7000.0]
        ).reshape(
            [1, 3, 1]
        ) 
        self.formant_freq_limits_abs_noise_low = torch.tensor(
            [4000.0, 3000.0, 3000.0]
        ).reshape(
            [1, 3, 1]
        ) 
        self.formant_bandwitdh_bias = Parameter(torch.Tensor(1))
        self.formant_bandwitdh_slop = Parameter(torch.Tensor(1))
        self.formant_bandwitdh_thres = Parameter(torch.Tensor(1))
        with torch.no_grad():
            nn.init.constant_(self.formant_bandwitdh_bias, 0)
            nn.init.constant_(self.formant_bandwitdh_slop, 0)
            nn.init.constant_(self.formant_bandwitdh_thres, 0)
        if broud:
            if wavebased:
                self.conv1_mel = ln.Conv1d(128, 64, 3, 1, 1)
                self.norm1_mel = nn.GroupNorm(32, 64)
                self.conv2_mel = ln.Conv1d(64, 128, 3, 1, 1)
                self.norm2_mel = nn.GroupNorm(32, 128)
                self.conv_fundementals_mel = ln.Conv1d(128, 128, 3, 1, 1)
                self.norm_fundementals_mel = nn.GroupNorm(32, 128)
                self.f0_drop_mel = nn.Dropout()
            if gender_patient == "Female":
                self.conv1_narrow = nn.Sequential(
                    ln.Conv1d(n_fft, 64, 3, 1, 1),
                    nn.GroupNorm(32, 64),
                    nn.LeakyReLU(0.2),
                    ln.Conv1d(64, 128, 3, 1, 1),
                    nn.GroupNorm(32, 128),
                    nn.LeakyReLU(0.2),
                )
            else:
                if larger_capacity:
                    self.conv1_narrow = nn.Sequential(
                        ln.Conv1d(n_fft, 256, 5, 1, 2),
                        nn.GroupNorm(32, 256),
                        nn.LeakyReLU(0.2),
                        ln.Conv1d(256, 128, 5, 1, 2),
                        nn.GroupNorm(32, 128),
                        nn.LeakyReLU(0.2),
                        ln.Conv1d(128, 64, 5, 1, 2),
                        nn.GroupNorm(32, 64),
                        nn.LeakyReLU(0.2),
                        ln.Conv1d(64, 128, 5, 1, 2),
                        nn.GroupNorm(32, 128),
                        nn.LeakyReLU(0.2),
                    )
                else:
                    self.conv1_narrow = nn.Sequential(
                        ln.Conv1d(n_fft, 128, 3, 1, 1),
                        nn.GroupNorm(32, 128),
                        nn.LeakyReLU(
                            0.2
                        ), 
                        ln.Conv1d(128, 64, 3, 1, 1),
                        nn.GroupNorm(32, 64),
                        nn.LeakyReLU(0.2),
                        ln.Conv1d(64, 128, 3, 1, 1),
                        nn.GroupNorm(32, 128),
                        nn.LeakyReLU(0.2),
                    )

            self.conv_fundementals_narrow = ln.Conv1d(128, 128, 3, 1, 1)
            self.norm_fundementals_narrow = nn.GroupNorm(32, 128)
            self.f0_drop_narrow = nn.Dropout()
            if wavebased:
                self.conv_f0_narrow = ln.Conv1d(256, 1, 1, 1, 0)
            else:
                self.conv_f0_narrow = ln.Conv1d(128, 1, 1, 1, 0)
            self.conv_amplitudes_narrow = ln.Conv1d(128, 2, 1, 1, 0)
            self.conv_amplitudes_h_narrow = ln.Conv1d(128, 2, 1, 1, 0)
        if wavebased:
            self.conv1 = ln.Conv1d(n_fft, 64, 3, 1, 1)
        else:
            self.conv1 = ln.Conv1d(n_mels, 64, 3, 1, 1)
        self.norm1 = nn.GroupNorm(32, 64)
        self.conv2 = ln.Conv1d(64, 128, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, 128)

        if gender_patient == "Female":
            self.conv1 = nn.Sequential(
                ln.Conv1d(n_fft, 64, 3, 1, 1),
                nn.GroupNorm(32, 64),
                nn.LeakyReLU(0.2),
                ln.Conv1d(64, 128, 3, 1, 1),
                nn.GroupNorm(32, 128),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv1 = nn.Sequential(
                ln.Conv1d(n_fft, 128, 3, 1, 1), 
                nn.GroupNorm(32, 128),
                nn.LeakyReLU(0.2),
                ln.Conv1d(128, 64, 3, 1, 1),
                nn.GroupNorm(32, 64),
                nn.LeakyReLU(0.2),
                ln.Conv1d(64, 128, 3, 1, 1),
                nn.GroupNorm(32, 128),
                nn.LeakyReLU(0.2),
            )

        self.conv_fundementals = ln.Conv1d(128, 128, 3, 1, 1)
        self.norm_fundementals = nn.GroupNorm(32, 128)
        self.f0_drop = nn.Dropout()
        self.conv_f0 = ln.Conv1d(128, 1, 1, 1, 0)
        self.conv_amplitudes = ln.Conv1d(128, 2, 1, 1, 0)
        self.conv_amplitudes_h = ln.Conv1d(128, 2, 1, 1, 0)
        print(
            "encoder loudness,",
            n_fft,
        )

        self.conv_loudness = nn.Sequential(
            ln.Conv1d(
                n_fft if wavebased else n_mels, 128, 1, 1, 0
            ),  
            nn.LeakyReLU(0.2),
            ln.Conv1d(128, 128, 1, 1, 0),
            nn.LeakyReLU(0.2),
            ln.Conv1d(128, 1, 1, 1, 0, bias_initial=-9.0 if power_synth else -4.6),
        )

        if self.broud:
            self.conv_formants = ln.Conv1d(128, 128, 3, 1, 1)
        else:
            self.conv_formants = ln.Conv1d(128, 128, 3, 1, 1)
        self.norm_formants = nn.GroupNorm(32, 128)
        self.conv_formants_freqs = ln.Conv1d(128, n_formants, 1, 1, 0)
        self.conv_formants_bandwidth = ln.Conv1d(128, n_formants, 1, 1, 0)
        self.conv_formants_amplitude = ln.Conv1d(128, n_formants, 1, 1, 0)

        self.conv_formants_freqs_noise = ln.Conv1d(128, self.n_formants_noise, 1, 1, 0)
        self.conv_formants_bandwidth_noise = ln.Conv1d(
            128, self.n_formants_noise, 1, 1, 0
        )
        self.conv_formants_amplitude_noise = ln.Conv1d(
            128, self.n_formants_noise, 1, 1, 0
        )
        self.conv1_bgnoise = ln.Conv1d(128, 128, 1, 1, 0)
        self.conv2_bgnoise = ln.Conv1d(
            128, n_fft if wavebased else n_mels, 1, 1, 0
        ) 
        self.norm1_bgnoise = nn.GroupNorm(32, 128 if wavebased else n_mels)
        
        self.norm2_bgnoise = nn.GroupNorm(
            32, n_fft if wavebased else n_mels
        ) 

        self.amplifier = Parameter(torch.Tensor(1))
        self.bias = Parameter(torch.Tensor(1))
        with torch.no_grad():
            nn.init.constant_(self.amplifier, 1.0)
            nn.init.constant_(self.bias, -50.0)
        print("self.noise_db,self.max_db", self.noise_db, self.max_db)

    def forward(
        self,
        x,
        x_denoise=None,
        duomask=False,
        noise_level=None,
        x_amp=None,
        gender="Female",
        onstage=None,
    ):
        x = x.squeeze(dim=1).permute(0, 2, 1) 
        if x_denoise is not None:
            x_denoise = x_denoise.squeeze(dim=1).permute(0, 2, 1)
        if x_amp is None:
            x_amp = amplitude(x, self.noise_db, self.max_db, trim_noise=True)
        else:
            x_amp = x_amp.squeeze(dim=1).permute(0, 2, 1)
        win = 4 * self.n_fft // 256 + 1 
        hann_win = torch.hann_window(win, periodic=False).reshape([1, 1, win, 1])
        x_smooth = (
            F.conv2d(
                x.unsqueeze(1).transpose(-2, -1), hann_win, padding=[(win - 1) // 2, 0]
            )
            .transpose(-2, -1)
            .squeeze(1)
        )
        if self.power_synth:
            loudness = F.softplus(
                (1.0 if self.wavebased else 1.0) * self.conv_loudness(x_smooth)
            )
        else:
            loudness = F.softplus(
                (1.0 if self.wavebased else 1.0) * self.conv_loudness(x_smooth)
            )
        if self.broud:
            x_narrow = x
            x_common_narrow = self.conv1_narrow(x_narrow)
            amplitudes = F.softmax(self.conv_amplitudes_narrow(x_common_narrow), dim=1)
            amplitudes_h = F.softmax(
                self.conv_amplitudes_h_narrow(x_common_narrow), dim=1
            )
            x_fundementals_narrow = self.f0_drop_narrow(
                F.leaky_relu(
                    self.norm_fundementals_narrow(
                        self.conv_fundementals_narrow(x_common_narrow)
                    ),
                    0.2,
                )
            )
            noisebg = F.softmax(
                self.norm1_bgnoise(self.conv1_bgnoise(x_common_narrow)), dim=1
            )
            noisebg = F.softmax(
                self.norm2_bgnoise(self.conv2_bgnoise(noisebg)), dim=1
            ).max(-1)[0]
            x_amp = amplitude(x.unsqueeze(1), self.noise_db, self.max_db).transpose(
                -2, -1
            )
            x_mel = to_db(
                torchaudio.transforms.MelScale(f_max=8000, n_stft=self.n_fft)(
                    x_amp.transpose(-2, -1)
                ),
                self.noise_db,
                self.max_db,
            ).squeeze(1)
            x = F.leaky_relu(self.norm1_mel(self.conv1_mel(x_mel)), 0.2)
            x_common_mel = F.leaky_relu(self.norm2_mel(self.conv2_mel(x)), 0.2)
            x_fundementals_mel = self.f0_drop_mel(
                F.leaky_relu(
                    self.norm_fundementals_mel(
                        self.conv_fundementals_mel(x_common_mel)
                    ),
                    0.2,
                )
            )
            if self.unified:  
                f0_hz = (
                    torch.sigmoid(
                        self.conv_f0_narrow(
                            torch.cat(
                                [x_fundementals_narrow, x_fundementals_mel], dim=1
                            )
                        )
                    )
                    * 220
                    + 80
                )  # 80-300
            else:
                f0_hz = (
                    torch.sigmoid(
                        self.conv_f0_narrow(
                            torch.cat(
                                [x_fundementals_narrow, x_fundementals_mel], dim=1
                            )
                        )
                    )
                    * 120
                    + 80
                    + 100 * gender.unsqueeze(-1)
                ) 
            f0 = torch.clamp(
                mel_scale(self.n_mels, f0_hz) / (self.n_mels * 1.0), min=0.0001
            )
            win = 20 * self.n_fft // 256 + 1
            hann_win = torch.hann_window(win, periodic=False).reshape([1, 1, win, 1])
            x = to_db(
                F.conv2d(x_amp, hann_win, padding=[(win - 1) // 2, 0]).transpose(
                    -2, -1
                ),
                self.noise_db,
                self.max_db,
            ).squeeze(1)
            x_common = self.conv1(x)
        x_formants = F.leaky_relu(self.norm_formants(self.conv_formants(x_common)), 0.2)
        formants_freqs = torch.sigmoid(self.conv_formants_freqs(x_formants))
        formants_freqs_hz = (
            formants_freqs
            * (
                self.formant_freq_limits_abs[:, : self.n_formants]
                - self.formant_freq_limits_abs_low[:, : self.n_formants]
            )
            + self.formant_freq_limits_abs_low[:, : self.n_formants]
        )
        formants_freqs_hz_male = (
            formants_freqs
            * (
                self.formant_freq_limits_abs_male[:, : self.n_formants]
                - self.formant_freq_limits_abs_low_male[:, : self.n_formants]
            )
            + self.formant_freq_limits_abs_low_male[:, : self.n_formants]
        )
        formants_freqs_hz_male_female = torch.stack(
            [formants_freqs_hz_male, formants_freqs_hz], dim=-1
        )
        dummy = (
            gender.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                gender.size(0), formants_freqs_hz.size(1), formants_freqs_hz.size(2), 2
            )
        )
        formants_freqs_hz = formants_freqs_hz_male_female.gather(-1, dummy.long())[
            ..., 0
        ]
        formants_freqs = torch.clamp(
            mel_scale(self.n_mels, formants_freqs_hz) / (self.n_mels * 1.0), min=0
        )
        formants_bandwidth_hz = 0.65 * (
            0.00625 * torch.relu(formants_freqs_hz) + 375
        ) 
        formants_bandwidth_hz_male = (
            0.55 * 0.65 * (0.00625 * torch.relu(formants_freqs_hz) + 300)
        )
        formants_bandwidth_hz_male_female = torch.stack(
            [formants_bandwidth_hz_male, formants_bandwidth_hz], dim=-1
        )
        dummy = (
            gender.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                gender.size(0),
                formants_bandwidth_hz.size(1),
                formants_bandwidth_hz.size(2),
                2,
            )
        )
        formants_bandwidth_hz = formants_bandwidth_hz_male_female.gather(
            -1, dummy.long()
        )[..., 0]
        formants_bandwidth = bandwidth_mel(
            formants_freqs_hz, formants_bandwidth_hz, self.n_mels
        )
        formants_amplitude_logit = self.conv_formants_amplitude(x_formants)
        formants_amplitude = F.softmax(formants_amplitude_logit, dim=1)
        formants_freqs_noise = torch.sigmoid(self.conv_formants_freqs_noise(x_formants))
        formants_freqs_hz_noise = (
            formants_freqs_noise
            * (
                self.formant_freq_limits_abs_noise[:, : self.n_formants_noise]
                - self.formant_freq_limits_abs_noise_low[:, : self.n_formants_noise]
            )
            + self.formant_freq_limits_abs_noise_low[:, : self.n_formants_noise]
        )
        if duomask:
            formants_freqs_hz_noise = torch.cat(
                [formants_freqs_hz, formants_freqs_hz_noise], dim=1
            )
        formants_freqs_noise = torch.clamp(
            mel_scale(self.n_mels, formants_freqs_hz_noise) / (self.n_mels * 1.0), min=0
        )
        formants_bandwidth_hz_noise = self.conv_formants_bandwidth_noise(x_formants)
        formants_bandwidth_hz_noise_1 = (
            F.softplus(formants_bandwidth_hz_noise[:, :1]) * 2344 + 586
        ) 
        formants_bandwidth_hz_noise_2 = (
            torch.sigmoid(formants_bandwidth_hz_noise[:, 1:]) * 586
        ) 
        formants_bandwidth_hz_noise = torch.cat(
            [formants_bandwidth_hz_noise_1, formants_bandwidth_hz_noise_2], dim=1
        )
        if duomask:
            formants_bandwidth_hz_noise = torch.cat(
                [formants_bandwidth_hz, formants_bandwidth_hz_noise], dim=1
            )
        formants_bandwidth_noise = bandwidth_mel(
            formants_freqs_hz_noise, formants_bandwidth_hz_noise, self.n_mels
        )
        formants_amplitude_noise_logit = self.conv_formants_amplitude_noise(x_formants)
        if duomask:
            formants_amplitude_noise_logit = torch.cat(
                [formants_amplitude_logit, formants_amplitude_noise_logit], dim=1
            )
        formants_amplitude_noise = F.softmax(formants_amplitude_noise_logit, dim=1)

        components = {
            "f0": f0,
            "f0_hz": f0_hz,
            "loudness": loudness,
            "amplitudes": amplitudes,
            "amplitudes_h": amplitudes_h,
            "freq_formants_hamon": formants_freqs,
            "bandwidth_formants_hamon": formants_bandwidth,
            "freq_formants_hamon_hz": formants_freqs_hz,
            "bandwidth_formants_hamon_hz": formants_bandwidth_hz,
            "amplitude_formants_hamon": formants_amplitude,
            "freq_formants_noise": formants_freqs_noise,
            "bandwidth_formants_noise": formants_bandwidth_noise,
            "freq_formants_noise_hz": formants_freqs_hz_noise,
            "bandwidth_formants_noise_hz": formants_bandwidth_hz_noise,
            "amplitude_formants_noise": formants_amplitude_noise,
            "noisebg": noisebg,
            "spec_smooth": x_smooth,
        }
        return components


##############################################################################################################
#########################################ECoG Decoder#########################################################
##############################################################################################################


class FromECoG(nn.Module):
    def __init__(self, outputs, residual=False, shape="3D"):
        super().__init__()
        self.residual = residual
        if shape == "3D":
            self.from_ecog = ln.Conv3d(1, outputs, [9, 1, 1], 1, [4, 0, 0])
        else:
            self.from_ecog = ln.Conv2d(1, outputs, [9, 1], 1, [4, 0])

    def forward(self, x):
        x = self.from_ecog(x)
        if not self.residual:
            x = F.leaky_relu(x, 0.2)
        return x


class FromECoG_causal(nn.Module):
    def __init__(
        self,
        outputs,
        residual=False,
        shape="3D",
        causal=False,
        anticausal=False,
        in_channel=1,
    ):
        super().__init__()
        self.residual = residual
        if shape == "3D":
            self.from_ecog = ln_c.Conv3d(
                in_channel,
                outputs,
                [(5 if causal else 9), 1, 1],
                1,
                [4, 0, 0],
                causal=causal,
                anticausal=anticausal,
            )
        else:
            self.from_ecog = ln_c.Conv2d(
                in_channel,
                outputs,
                [(5 if causal else 9), 1],
                1,
                [4, 0],
                causal=causal,
                anticausal=anticausal,
            )

    def forward(self, x):
        x = self.from_ecog(x)
        if not self.residual:
            x = F.leaky_relu(x, 0.2)
        return x


class ECoGMappingBlock_causal(nn.Module):
    def __init__(
        self,
        inputs,
        outputs,
        kernel_size,
        dilation=1,
        fused_scale=True,
        residual=False,
        resample=[],
        pool=None,
        shape="3D",
        causal=False,
        anticausal=False,
        norm=nn.GroupNorm,
    ):
        super(ECoGMappingBlock_causal, self).__init__()
        self.residual = residual
        self.pool = pool
        self.inputs_resample = resample
        self.dim_missmatch = inputs != outputs
        self.resample = resample
        if not self.resample:
            self.resample = 1
        self.padding = list(np.array(dilation) * (np.array(kernel_size) - 1) // 2)
        if shape == "1D":
            conv = ln_c.Conv1d
            maxpool = nn.MaxPool1d
            avgpool = nn.AvgPool1d
        if shape == "2D":
            conv = ln_c.Conv2d
            maxpool = nn.MaxPool2d
            avgpool = nn.AvgPool2d
        if shape == "3D":
            conv = ln_c.Conv3d
            maxpool = nn.MaxPool3d
            avgpool = nn.AvgPool3d
        if residual:
            self.norm1 = norm(inputs, inputs)
        else:
            self.norm1 = norm(outputs, outputs)
        if pool is None:
            self.conv1 = conv(
                inputs,
                outputs,
                kernel_size,
                self.resample,
                self.padding,
                dilation=dilation,
                bias=False,
                causal=causal,
                anticausal=anticausal,
            )
        else:
            self.conv1 = conv(
                inputs,
                outputs,
                kernel_size,
                1,
                self.padding,
                dilation=dilation,
                bias=False,
                causal=causal,
                anticausal=anticausal,
            )
            self.pool1 = (
                maxpool(self.resample, self.resample)
                if self.pool == "Max"
                else avgpool(self.resample, self.resample)
            )
        if self.inputs_resample or self.dim_missmatch:
            if pool is None:
                self.convskip = conv(
                    inputs,
                    outputs,
                    kernel_size,
                    self.resample,
                    self.padding,
                    dilation=dilation,
                    bias=False,
                    causal=causal,
                    anticausal=anticausal,
                )
            else:
                self.convskip = conv(
                    inputs,
                    outputs,
                    kernel_size,
                    1,
                    self.padding,
                    dilation=dilation,
                    bias=False,
                    causal=causal,
                    anticausal=anticausal,
                )
                self.poolskip = (
                    maxpool(self.resample, self.resample)
                    if self.pool == "Max"
                    else avgpool(self.resample, self.resample)
                )

        self.conv2 = conv(
            outputs,
            outputs,
            kernel_size,
            1,
            self.padding,
            dilation=dilation,
            bias=False,
            causal=causal,
            anticausal=anticausal,
        )
        self.norm2 = norm(outputs, outputs)

    def forward(self, x):
        if self.residual:
            x = F.leaky_relu(self.norm1(x), 0.2)
            if self.inputs_resample or self.dim_missmatch:
                # x_skip = F.avg_pool3d(x,self.resample,self.resample)
                x_skip = self.convskip(x)
                if self.pool is not None:
                    x_skip = self.poolskip(x_skip)
            else:
                x_skip = x
            x = F.leaky_relu(self.norm2(self.conv1(x)), 0.2)
            if self.pool is not None:
                x = self.poolskip(x)
            x = self.conv2(x)
            x = x_skip + x

        else:
            x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
            x = F.leaky_relu(self.norm2(self.conv2(x)), 0.2)
        return x

formant_freq_limits_diff = torch.tensor([950.,2450.,2100.]).reshape([1,3,1]).to(device) #freq difference
formant_freq_limits_diff_low = torch.tensor([300.,300.,0.]).reshape([1,3,1]).to(device) #freq difference
formant_freq_limits_abs = torch.tensor([950.,3400.,3800.,5000.,6000.,7000.]).reshape([1,6,1]).to(device) #freq difference
formant_freq_limits_abs_low = torch.tensor([300.,700.,1800.,3400,5000.,6000.]).reshape([1,6,1]).to(device) #freq difference
formant_freq_limits_abs_noise = torch.tensor([8000.,7000.,7000.]).reshape([1,3,1]).to(device) #freq difference
formant_freq_limits_abs_noise_low = torch.tensor([4000.,3000.,3000.]).reshape([1,3,1]).to(device) #freq difference
formant_bandwitdh_ratio = Parameter(torch.Tensor(1)).to(device)
formant_bandwitdh_slop = Parameter(torch.Tensor(1)).to(device)
class Speech_Para_Prediction(nn.Module):
    def __init__(self, causal, anticausal, compute_db_loudness=False, n_formants= 6,n_formants_noise=1,\
                 n_mels=32,network_db=False,input_channel=32):
        super(Speech_Para_Prediction, self).__init__()
        norm = GroupNormXDim
        
        self.n_formants = n_formants
        self.n_mels = n_mels
        self.n_formants_noise = n_formants_noise
        self.compute_db_loudness = compute_db_loudness
        self.network_db = network_db
        self.formant_freq_limits_diff = formant_freq_limits_diff
        self.formant_freq_limits_diff_low = formant_freq_limits_diff_low
        self.formant_freq_limits_abs = formant_freq_limits_abs
        self.formant_freq_limits_abs_low = formant_freq_limits_abs_low
        self.formant_freq_limits_abs_noise = formant_freq_limits_abs_noise
        self.formant_freq_limits_abs_noise_low = formant_freq_limits_abs_noise_low
        self.formant_bandwitdh_ratio = formant_bandwitdh_ratio
        self.formant_bandwitdh_slop = formant_bandwitdh_slop
        with torch.no_grad():
            nn.init.constant_(self.formant_bandwitdh_ratio,0)
            nn.init.constant_(self.formant_bandwitdh_slop,0)
        
        self.conv_fundementals = ln_c.Conv1d(input_channel,32,3,1,1 ,causal=causal,anticausal = anticausal)
        self.norm_fundementals = norm(32,32)
        self.f0_drop = nn.Dropout()
        self.conv_f0 = ln_c.Conv1d(32,1,1,1,0 ,causal=causal,anticausal = anticausal)
        self.conv_amplitudes = ln_c.Conv1d(input_channel,2,1,1,0 ,causal=causal,anticausal = anticausal)
        self.conv_amplitudes_h = ln_c.Conv1d(input_channel,2,1,1,0 ,causal=causal,anticausal = anticausal)
        if compute_db_loudness:
            self.conv_loudness = ln_c.Conv1d(input_channel,1,1,1,0 ,causal=causal,anticausal = anticausal)
        else:
            self.conv_loudness = ln_c.Conv1d(input_channel,1,1,1,0,bias_initial=-9.  ,causal=causal,anticausal = anticausal)
        self.conv_formants = ln_c.Conv1d(input_channel,32,3,1,1 ,causal=causal,anticausal = anticausal)
        self.norm_formants = norm(32,32)
        self.conv_formants_freqs = ln_c.Conv1d(32,n_formants,1,1,0 ,causal=causal,anticausal = anticausal)
        self.conv_formants_bandwidth = ln_c.Conv1d(32,n_formants,1,1,0 ,causal=causal,anticausal = anticausal)
        self.conv_formants_amplitude = ln_c.Conv1d(32,n_formants,1,1,0 ,causal=causal,anticausal = anticausal)
        self.conv_formants_freqs_noise = ln_c.Conv1d(32,n_formants_noise,1,1,0 ,causal=causal,anticausal = anticausal)
        self.conv_formants_bandwidth_noise = ln_c.Conv1d(32,n_formants_noise,1,1,0 ,causal=causal,anticausal = anticausal)
        self.conv_formants_amplitude_noise = ln_c.Conv1d(32,n_formants_noise,1,1,0 ,causal=causal,anticausal = anticausal)
    def forward(self, x_common):
        if self.compute_db_loudness:
            loudness = F.sigmoid(self.conv_loudness(x_common)) #0-1
            loudness = loudness*200-100 #-100 ~ 100 db
            loudness = 10**(loudness/10.) #amplitude
        else:
            loudness = F.softplus(self.conv_loudness(x_common))
        logits = self.conv_amplitudes(x_common)
        amplitudes_logsoftmax = F.log_softmax(logits,dim=1)
        amplitudes_h = F.softmax(self.conv_amplitudes_h(x_common),dim=1)
        x_fundementals = self.f0_drop(F.leaky_relu(self.norm_fundementals(self.conv_fundementals(x_common)),0.2))
        f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 332 + 88 # 88hz < f0 < 420 hz
        f0 = torch.clamp(mel_scale(self.n_mels,f0_hz)/(self.n_mels*1.0),min=0.0001)

        x_formants = F.leaky_relu(self.norm_formants(self.conv_formants(x_common)),0.2)
        formants_freqs = torch.sigmoid(self.conv_formants_freqs(x_formants))
        formants_freqs_hz = formants_freqs*(self.formant_freq_limits_abs[:,:self.n_formants]-self.formant_freq_limits_abs_low[:,:self.n_formants])+self.formant_freq_limits_abs_low[:,:self.n_formants]
        formants_freqs = torch.clamp(mel_scale(self.n_mels,formants_freqs_hz)/(self.n_mels*1.0),min=0)
        formants_bandwidth_hz = 0.65*(0.00625*torch.relu(formants_freqs_hz)+375)
        formants_bandwidth = bandwidth_mel(formants_freqs_hz,formants_bandwidth_hz,self.n_mels)
        formants_amplitude_logit = self.conv_formants_amplitude(x_formants)
        formants_freqs_noise = torch.sigmoid(self.conv_formants_freqs_noise(x_formants))
        formants_freqs_hz_noise = formants_freqs_noise*(self.formant_freq_limits_abs_noise[:,:self.n_formants_noise]-self.formant_freq_limits_abs_noise_low[:,:self.n_formants_noise])+self.formant_freq_limits_abs_noise_low[:,:self.n_formants_noise]
        formants_freqs_hz_noise = torch.cat([formants_freqs_hz,formants_freqs_hz_noise],dim=1)
        formants_freqs_noise = torch.clamp(mel_scale(self.n_mels,formants_freqs_hz_noise)/(self.n_mels*1.0),min=0)
        formants_bandwidth_hz_noise = self.conv_formants_bandwidth_noise(x_formants)
        formants_bandwidth_hz_noise_1 = F.softplus(formants_bandwidth_hz_noise[:,:1]) * 2344 + 586 #2000-10000
        formants_bandwidth_hz_noise_2 = torch.sigmoid(formants_bandwidth_hz_noise[:,1:]) * 586 #0-2000
        formants_bandwidth_hz_noise = torch.cat([formants_bandwidth_hz_noise_1,formants_bandwidth_hz_noise_2],dim=1)
        formants_bandwidth_hz_noise = torch.cat([formants_bandwidth_hz,formants_bandwidth_hz_noise],dim=1)
        formants_bandwidth_noise = bandwidth_mel(formants_freqs_hz_noise,formants_bandwidth_hz_noise,self.n_mels)
        formants_amplitude_noise_logit = self.conv_formants_amplitude_noise(x_formants)
        formants_amplitude_noise_logit = torch.cat([formants_amplitude_logit,formants_amplitude_noise_logit],dim=1)
        if self.network_db:
            formants_amplitude = torch.sigmoid(formants_amplitude_logit )    
            formants_amplitude_noise = torch.sigmoid(formants_amplitude_noise_logit )
            amplitudes = torch.sigmoid(logits ) 
            amplitudes = db_to_amp(amplitudes)
            formants_amplitude_noise = db_to_amp(formants_amplitude_noise)
            formants_amplitude = db_to_amp(formants_amplitude)
            amplitudes = amplitudes/torch.sum(amplitudes,dim=1,keepdim=True)
            formants_amplitude_noise = formants_amplitude_noise/torch.sum(formants_amplitude_noise,dim=1,keepdim=True)
            formants_amplitude = formants_amplitude/torch.sum(formants_amplitude,dim=1,keepdim=True)
        else:
            formants_amplitude = F.softmax(formants_amplitude_logit,dim=1)    
            formants_amplitude_noise = F.softmax(formants_amplitude_noise_logit,dim=1)
            amplitudes = F.softmax(logits,dim=1) 
            
        components = { 'f0':f0 ,
                    'f0_hz':f0_hz ,
                    'loudness':loudness ,
                    'amplitudes':amplitudes ,
                    'amplitudes_logsoftmax':amplitudes_logsoftmax ,
                    'amplitudes_h':amplitudes_h ,
                    'freq_formants_hamon':formants_freqs,
                    'bandwidth_formants_hamon':formants_bandwidth ,
                    'freq_formants_hamon_hz':formants_freqs_hz ,
                    'bandwidth_formants_hamon_hz':formants_bandwidth_hz ,
                    'amplitude_formants_hamon':formants_amplitude ,
                    'freq_formants_noise':formants_freqs_noise ,
                    'bandwidth_formants_noise':formants_bandwidth_noise ,
                    'freq_formants_noise_hz':formants_freqs_hz_noise ,
                    'bandwidth_formants_noise_hz':formants_bandwidth_hz_noise ,
                    'amplitude_formants_noise':formants_amplitude_noise ,
        }
        return components
        

class BasicRNN(torch.nn.Module):
    """vanilla rnn"""

    def __init__(
        self,
        n_layers=2,
        n_rnn_units=256,
        max_sequence_length=500,
        bidirectional=False,
        n_input_features=256,
        n_output_features=256,
        batch_first=True,
        dropout=0.5,
        use_final_linear_layer=False,
    ):
        super(BasicRNN, self).__init__()

        # load params
        self.n_layers = n_layers
        self.n_rnn_units = n_rnn_units
        self.max_sequence_length = max_sequence_length
        self.n_directions = 2 if bidirectional else 1
        self.n_input_features = n_input_features
        self.n_output_features = n_output_features
        self.use_final_linear_layer = use_final_linear_layer
        self.batch_first = batch_first

        # define architecture
        self.transform_initial_state = torch.nn.Linear(
            self.n_input_features,
            self.n_rnn_units * self.n_directions * self.n_layers,
            bias=False,
        )
        self.rnn = torch.nn.LSTM(
            self.n_input_features,
            self.n_rnn_units,
            self.n_layers,
            batch_first=self.batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        if self.use_final_linear_layer:
            self.output_layer = torch.nn.Linear(
                n_rnn_units * self.n_directions, self.n_output_features
            )

    def forward(
        self,
        inputs,
        initial_state=None,
        seq_lengths=None,
        transform_state=False,
        **kwargs,
    ):

        # transform states if needed
        if initial_state is not None and transform_state:
            transformed_state = self.transform_initial_state(initial_state)
            transformed_state = transformed_state.view(
                -1, self.n_layers * self.n_directions, self.n_rnn_units
            )
        else:
            transformed_state = initial_state

        # run rnn
        if seq_lengths is not None:
            inputs = torch.nn.utils.rnn.pack_padded_sequence(
                inputs, seq_lengths, batch_first=True, enforce_sorted=False
            )
        # import pdb; pdb.set_trace()
        output, state = self.rnn(inputs, transformed_state)
        if seq_lengths is not None:
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=self.batch_first, total_length=self.max_length
            )

        # optional linear layer and return
        if self.use_final_linear_layer:
            output = self.output_layer(output)
        # print ('basic rnn output',output.shape)
        return output


@ECOG_ENCODER.register("ECoGMapping_ResNet")
class ECoGMapping_Bottleneck_ran(nn.Module):
    def __init__(
        self,
        n_mels,
        n_formants,
        n_formants_noise=1,
        compute_db_loudness=False,
        causal=False,
        anticausal=False,
        pre_articulate=False,
        upsample="ConvTranspose",
        GR=False,
        network_db=False,
        latent_channel=32,
    ):
        super(ECoGMapping_Bottleneck_ran, self).__init__()
        self.causal = causal
        self.anticausal = anticausal
        causal_ = [1, 0, 0] if causal else False
        self.pre_articulate = pre_articulate
        norm = GroupNormXDim
        base = 32 if self.pre_articulate else 16
        conv1_in_channel = base
        self.from_ecog = FromECoG_causal(
            base, in_channel=1, residual=True, causal=causal_, anticausal=anticausal
        )
        self.conv1 = ECoGMappingBlock_causal(
            conv1_in_channel,
            base * 2,
            [5, 1, 1],
            residual=True,
            resample=[2, 1, 1],
            pool="MAX",
            causal=causal_,
            anticausal=anticausal,
        )
        self.conv2 = ECoGMappingBlock_causal(
            base * 2,
            base * 4,
            [3, 1, 1],
            residual=True,
            resample=[(1 if self.pre_articulate else 2), 1, 1],
            pool="MAX",
            causal=causal_,
            anticausal=anticausal,
        )
        self.conv3 = ECoGMappingBlock_causal(
            base * 4,
            base * 8,
            [3, 3, 3],
            residual=True,
            resample=[2, 2, 2],
            pool="MAX",
            causal=causal_,
            anticausal=anticausal,
        )
        self.conv4 = ECoGMappingBlock_causal(
            base * 8,
            base * 16,
            [3, 3, 3],
            residual=True,
            resample=[(1 if self.pre_articulate else 2), 2, 2],
            pool="MAX",
            causal=causal_,
            anticausal=anticausal,
        )
        self.norm_mask = norm(32, base * 4)
        self.mask = ln_c.Conv3d(
            base * 4, 1, [3, 1, 1], 1, [1, 0, 0], causal=causal_, anticausal=anticausal
        )
        self.norm = norm(32, base * 16)
        self.conv5 = ln_c.Conv1d(
            base * 16, base * 16, 3, 1, 1, causal=causal, anticausal=anticausal
        )
        self.norm2 = norm(32, base * 16)
        if self.pre_articulate:
            self.lin1 = ln_c.Linear(base * 16, 2048)  # 2048 = 256*8
        self.conv6 = Upsample_Block(
            256, 128, bilinear=True if upsample == "bilinear" else False
        )
        self.norm3 = norm(32, 128)
        self.conv7 = Upsample_Block(
            128, 64, bilinear=True if upsample == "bilinear" else False
        )
        self.norm4 = norm(32, 64)
        self.conv8 = Upsample_Block(
            64, 32, bilinear=True if upsample == "bilinear" else False
        )
        self.norm5 = norm(32, 32)
        self.conv9 = Upsample_Block(
            32, latent_channel, bilinear=True if upsample == "bilinear" else False
        )
        self.norm6 = norm(32, latent_channel)
        self.dropout = nn.Dropout(p=0.2)
        self.GR = GR
        if GR:  # gradient reversal
            self.Speaker_Classifier_GR(
                Channels=[32], previous_Channels=32, Num_Speakers=2
            )
        self.prediction_head = Speech_Para_Prediction(
            causal=causal,
            anticausal=anticausal,
            compute_db_loudness=compute_db_loudness,
            n_formants=n_formants,
            n_formants_noise=n_formants_noise,
            n_mels=n_mels,
            network_db=network_db,
            input_channel=latent_channel,
        )

    def forward(self, ecog, ):
        elec_length = int(ecog[0].shape[-1] ** 0.5)
        x = ecog
        x = x.reshape([-1, 1, x.shape[1], elec_length, elec_length])
        x = self.from_ecog(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x[:, :, 4:]
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.dropout(x)
        #C*8*2*2
        x = x.max(-1)[0].max(-1)[0]
        #C*8*1*1
        x = self.conv5(F.leaky_relu(self.norm(x), 0.2))
        x = self.norm2(x)
        x = self.conv6(F.leaky_relu(x, 0.2))
        x = self.conv7(F.leaky_relu(self.norm3(x), 0.2))
        x = self.conv8(F.leaky_relu(self.norm4(x), 0.2))
        x = self.conv9(F.leaky_relu(self.norm5(x), 0.2))
        #C'*128
        x_common = F.leaky_relu(self.norm6(x), 0.2)
        # print (x_common.shape)
        if self.GR:
            classified_Speakers = self.Speaker_Classifier_GR(x_common)
        components = self.prediction_head(x_common)
        return components if not self.GR else classified_Speakers


@ECOG_ENCODER.register("ECoGMapping_RNN")
class ECoGMappingRNN_ran(torch.nn.Module):
    """Model used to predict motion traces and speech features from ECoG data using Transformer"""

    def __init__(
        self,
        n_electrodes=64,
        n_input_features=64,
        n_classes=64,  # latent feature dimension
        n_layers=3,
        n_rnn_units=128,
        max_sequence_length=500,
        bidirectional=False,
        n_output_features=256,
        batch_first=True,
        dropout=0.5,
        use_final_linear_layer=False,
        normalize_inputs=False,
        out_channels=None,
        n_mels=32,
        network_db=False,
        causal=False,
        n_formants=5,
        n_formants_noise=1,
        anticausal=False,
        compute_db_loudness=False,
        pre_articulate=False,
        GR=False,
    ):
        super(ECoGMappingRNN_ran, self).__init__()

        # params
        self.n_classes = n_classes
        self.num_electrodes = n_electrodes
        self.normalize_inputs = normalize_inputs

        # rnn model
        self.base_model = BasicRNN(
            n_layers=n_layers,
            n_rnn_units=n_rnn_units,
            max_sequence_length=max_sequence_length,
            bidirectional=False if causal else True,
            n_input_features=n_input_features,
            n_output_features=n_output_features,
            batch_first=batch_first,
            dropout=dropout,
            use_final_linear_layer=use_final_linear_layer,
        )

        self.relu = torch.nn.ReLU()
        latent_channel = self.n_classes
        self.motion_projection = torch.nn.Linear(
            n_rnn_units * 2 if not causal else n_rnn_units, latent_channel
        )
        norm = GroupNormXDim
        self.dropout = nn.Dropout(p=0.2)
        self.GR = GR
        if GR:  # gradient reversal
            self.Speaker_Classifier_GR(
                Channels=[self.n_classes],
                previous_Channels=self.n_classes,
                Num_Speakers=2,
            )
        self.prediction_head = Speech_Para_Prediction(
            causal=causal,
            anticausal=anticausal,
            compute_db_loudness=compute_db_loudness,
            n_formants=n_formants,
            n_formants_noise=n_formants_noise,
            n_mels=n_mels,
            network_db=network_db,
            input_channel=latent_channel,
        )

    def forward(
        self,
        inputs,
        gender="Female",
        **kwargs,
    ):

        dim = inputs.shape[2]
        n_features = dim // self.num_electrodes
        if self.normalize_inputs:
            for n in range(n_features):
                normalized_inputs = torch.nn.functional.normalize(
                    inputs[
                        :, :, n * self.num_electrodes : (n + 1) * self.num_electrodes
                    ],
                    dim=-1,
                )
                inputs[
                    :, :, n * self.num_electrodes : (n + 1) * self.num_electrodes
                ] = normalized_inputs
        latent_representation = self.base_model.forward(inputs)
        print("latent_representation", latent_representation.shape)
        x_common = self.motion_projection(self.relu(latent_representation))[
            :, 8:-8
        ].permute(0, 2, 1)
        print("x_common", x_common.shape)
        if self.GR:
            classified_Speakers = self.Speaker_Classifier_GR(x_common)
        components = self.prediction_head(x_common)
        return components if not self.GR else classified_Speakers


@ECOG_ENCODER.register("ECoGMapping_3D_SWIN")
class ECoGMapping_3D_SWIN(nn.Module):
    def __init__(
        self,
        n_mels,
        n_formants,
        n_formants_noise=1,
        compute_db_loudness=False,
        causal=False,
        anticausal=False,
        pre_articulate=False,
        hidden_dim=256,
        upsample="bilinear",
        select_ind=0,
        network_db=False,
        GR=0,
    ):
        super(ECoGMapping_3D_SWIN, self).__init__()
        self.causal = causal
        self.anticausal = anticausal
        causal_ = [1, 0, 0] if causal else False
        self.n_formants = n_formants
        self.n_mels = n_mels
        self.n_formants_noise = n_formants_noise
        self.compute_db_loudness = compute_db_loudness
        self.pre_articulate = pre_articulate

        self.norm6 = nn.GroupNorm(hidden_dim, hidden_dim)
        self.conv_fundementals = ln_c.Conv1d(
            hidden_dim, 32, 3, 1, 1, causal=causal, anticausal=anticausal
        )
        self.norm_fundementals = nn.GroupNorm(32, 32)
        self.f0_drop = nn.Dropout()
        self.conv_f0 = ln_c.Conv1d(32, 1, 1, 1, 0, causal=causal, anticausal=anticausal)
        self.conv_amplitudes = ln_c.Conv1d(
            hidden_dim, 2, 1, 1, 0, causal=causal, anticausal=anticausal
        )
        self.conv_amplitudes_h = ln_c.Conv1d(
            hidden_dim, 2, 1, 1, 0, causal=causal, anticausal=anticausal
        )
        if compute_db_loudness:
            self.conv_loudness = ln_c.Conv1d(
                hidden_dim, 1, 1, 1, 0, causal=causal, anticausal=anticausal
            )
        else:
            self.conv_loudness = ln_c.Conv1d(
                hidden_dim,
                1,
                1,
                1,
                0,
                bias_initial=-9.0,
                causal=causal,
                anticausal=anticausal,
            )

        self.conv_formants = ln_c.Conv1d(
            hidden_dim, 32, 3, 1, 1, causal=causal, anticausal=anticausal
        )
        self.norm_formants = nn.GroupNorm(32, 32)
        self.conv_formants_freqs = ln_c.Conv1d(
            32, n_formants, 1, 1, 0, causal=causal, anticausal=anticausal
        )
        self.conv_formants_bandwidth = ln_c.Conv1d(
            32, n_formants, 1, 1, 0, causal=causal, anticausal=anticausal
        )
        self.conv_formants_amplitude = ln_c.Conv1d(
            32, n_formants, 1, 1, 0, causal=causal, anticausal=anticausal
        )

        self.conv_formants_freqs_noise = ln_c.Conv1d(
            32, n_formants_noise, 1, 1, 0, causal=causal, anticausal=anticausal
        )
        self.conv_formants_bandwidth_noise = ln_c.Conv1d(
            32, n_formants_noise, 1, 1, 0, causal=causal, anticausal=anticausal
        )
        self.conv_formants_amplitude_noise = ln_c.Conv1d(
            32, n_formants_noise, 1, 1, 0, causal=causal, anticausal=anticausal
        )

        depthss = [[2, 2, 6, 2], [2, 2, 6, 2]]
        patch_size_Ts = [2, 2]
        patch_size_Es = [(1, 1), (2, 2)]
        window_size_Ts = [4, 8, 16]
        window_size_Es = [(4, 4), (4, 4), (4, 4)]
        numclassess = [384, 384, 384]  # according to final temporal length

        if causal:
            norm = GroupNormXDim
        else:
            norm = GroupNormXDim
        base_swin = 48
        self.swin_transformer = SwinTransformer3D1(
            patch_size=(2, 2, 2),
            in_chans=1,
            embed_dim=48,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(window_size_Ts[select_ind], 2, 2),
            mlp_ratio=4.0,
            causal=causal,
            anticausal=anticausal,
        )
        self.swin_down_times = 4
        self.conv6 = ln.ConvTranspose1d(
            numclassess[select_ind], 256, 3, 2, 1, transform_kernel=True
        ) 

        self.upconvs1 = Upsample_Block(
            256, hidden_dim, bilinear=True if upsample == "bilinear" else False
        )
        self.norms1 = norm(32, hidden_dim)
        self.upconvs2 = Upsample_Block(
            hidden_dim, hidden_dim, bilinear=True if upsample == "bilinear" else False
        )
        self.norms2 = norm(32, hidden_dim)
        self.upconvs3 = Upsample_Block(
            hidden_dim, hidden_dim, bilinear=True if upsample == "bilinear" else False
        )
        self.norms3 = norm(32, hidden_dim)
        self.upconvs4 = Upsample_Block(
            hidden_dim, hidden_dim, bilinear=True if upsample == "bilinear" else False
        )
        self.norms4 = norm(32, hidden_dim)
        self.upconv_layers = [
            self.upconvs1,
            self.upconvs2,
            self.upconvs3,
            self.upconvs4,
        ]
        self.norm_layers = [self.norms1, self.norms2, self.norms3, self.norms4]
        self.GR = GR
        if GR: 
            self.Speaker_Classifier_GR(
                Channels=[self.n_classes],
                previous_Channels=self.n_classes,
                Num_Speakers=2,
            )

        self.prediction_head = Speech_Para_Prediction(
            causal=causal,
            anticausal=anticausal,
            compute_db_loudness=compute_db_loudness,
            n_formants=n_formants,
            n_formants_noise=n_formants_noise,
            n_mels=n_mels,
            network_db=network_db,
            input_channel=hidden_dim,
        )

    def forward(self, ecog):
        elec_length = int(ecog[0].shape[-1] ** 0.5)
        x = ecog[:, 16:]
        x = x.reshape([-1, x.shape[1], elec_length, elec_length, 1])
        xs = F.pad(
            input=x, pad=(0, 0, 1, 0, 1, 0, 0, 0), mode="constant", value=0
        ).permute(
            0, 4, 1, 2, 3
        )
        bs_per = x.shape[0]
        xs = (
            self.swin_transformer(xs, region_index=None)
            .squeeze(-1)
            .squeeze(-1)
            .transpose(1, 2)
        )
        x = self.conv6(F.leaky_relu(xs.transpose(1, 2), 0.2))

        for i in range(self.swin_down_times - 1):
            x = self.upconv_layers[i](F.leaky_relu(self.norm_layers[i](x), 0.2))
        x_common = F.leaky_relu(self.norm6(x), 0.2)
        print("x_common", x_common.shape)
        if self.GR:
            classified_Speakers = self.Speaker_Classifier_GR(x_common)
        components = self.prediction_head(x_common)
        return components if not self.GR else classified_Speakers


# gradient reversal
class Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight):
        ctx.save_for_backward(input_)
        ctx.weight = weight
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -ctx.weight * grad_output
        return grad_input, None


class GRL(torch.nn.Module):
    def __init__(self, weight=1.0):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super(GRL, self).__init__()
        self.weight = weight

    def forward(self, input_):
        return Func.apply(
            input_, torch.FloatTensor([self.weight]).to(device=input_.device)
        )


class Speaker_Classifier_GR(torch.nn.Module):
    def __init__(
        self,
        Adversarial_Speaker_Weight=0.0005,
        Channels=[256],
        previous_Channels=256,
        Num_Speakers=2,
    ):
        super(Speaker_Classifier_GR, self).__init__()

        self.layer = torch.nn.Sequential()
        self.layer.add_module("GRL", GRL(weight=Adversarial_Speaker_Weight))

        for index, channels in enumerate(Channels):
            self.layer.add_module(
                "Hidden_{}".format(index),
                Conv1d(
                    in_channels=previous_Channels,
                    out_channels=channels,
                    kernel_size=1,
                    bias=True,
                    w_init_gain="relu",
                ),
            )
            self.layer.add_module("ReLU_{}".format(index), torch.nn.ReLU())
            previous_Channels = channels

        self.layer.add_module(
            "Output_{}".format(index),
            Conv1d(
                in_channels=previous_Channels,
                out_channels=Num_Speakers,
                kernel_size=1,
                bias=True,
                w_init_gain="linear",
            ),
        )

    def forward(self, x):
        return self.layer(x.unsqueeze(2)).squeeze(2)
