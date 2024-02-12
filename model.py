# Copyright 2020-2023 Xupeng Chen, Ran Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""audio to audio and ecog to audio pipeline with loss functions"""

from networks import *
import numpy as np
from torch.nn import functional as F
from metrics.torch_stoi import NegSTOILoss
cumsum = torch.cumsum


def compdiff(comp):
    '''smoothness loss'''
    return ((comp[:, :, 1:] - comp[:, :, :-1]).abs()).mean()
def compdiffd2(comp):
    diff = comp[:, :, 1:] - comp[:, :, :-1]
    return ((diff[:, :, 1:] - diff[:, :, :-1]).abs()).mean()
def diff_dim(data, axis=1):
    if axis == 1:
        data = F.pad(data, (0, 0, 1, 0))
        return data[:, 1:] - data[:, :-1]
    elif axis == 2:
        data = F.pad(data, (1, 0, 0, 0))
        return data[:, :, 1:] - data[:, :, :-1]

def safe_divide(numerator, denominator, eps=1e-7):
    """Avoid dividing by zero by adding a small epsilon."""
    safe_denominator = torch.where(
        denominator == 0.0, eps, denominator.double()
    ).float()
    return numerator / safe_denominator


def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    # print ('type',type(x),x)
    safe_x = torch.where(x <= eps, eps, x.double())
    return torch.log(safe_x).float()

def safe_log_(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    print("type", type(x))
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)

def logb(x, base=2.0, safe=True):
    """Logarithm with base as an argument."""
    if safe:
        return safe_divide(safe_log(x), safe_log(torch.tensor([base])))
    else:
        return torch.log(x) / torch.log(base)

def hz_to_midi(frequencies):
    """Torch-compatible hz_to_midi function."""
    notes = 12.0 * (logb(frequencies, 2.0) - logb(torch.tensor([440.0]), 2.0)) + 69.0
    # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
    notes = torch.where(torch.less_equal(frequencies, 0.0), 0.0, notes.double())
    return notes.float()

def piecewise_linear(epoch, start_decay=20, end_decay=40):
    if epoch < start_decay:
        return 1
    elif start_decay <= epoch < end_decay:
        return 1 / (start_decay - end_decay) * epoch + 2
    else:
        return 0

def minmaxscale(data, quantile=0.9):
    # for frequency scaling
    if quantile is not None:
        datamax = torch.quantile(data, quantile)
        data = torch.clip(data, -10e10, datamax)
    minv = data.min()
    maxv = data.max()
    if minv == maxv:
        return data
    else:
        return (data - minv) / (maxv - minv)

def minmaxscale_ref(data, data_select, quantile=0.9):
    # for noise scale
    # data_select: a reference data, eg only true noise part
    if quantile is not None:
        datamax = torch.quantile(data_select, quantile)
        data_select = torch.clip(data_select, -10e10, datamax)
    minv = data_select.min()
    maxv = data_select.max()
    data = torch.clip(data, -10e10, datamax)
    # print (minv,maxv,maxv - minv)
    if minv == maxv:
        return data
    else:
        return (data - minv) / (maxv - minv)

def df_norm_torch(amp):
    amp_db = torchaudio.transforms.AmplitudeToDB()(amp)
    amp_db_norm = (amp_db.clamp(min=-70) + 70) / 50
    return amp_db, amp_db_norm

class GHMR(nn.Module):
    def __init__(self, mu=0.02, bins=30, momentum=0, loss_weight=1.0):
        super(GHMR, self).__init__()
        """
        explained in Gradient Harmonized Single-stage Detector: https://arxiv.org/pdf/1811.05181.pdf
        balance the training process in object detection and other tasks
        especially for datasets with imbalanced distributions of object classes and difficulties.
        """
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None, reweight=1):
        """Args:
        pred [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:
            The target regression values with the same size of pred.
        label_weight [batch_num, 4 (* class_num)]:
            The weight of each sample, 0 if ignored.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum
        # ASL1 loss
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu
        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)
        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = (loss * reweight).sum() / tot
        return loss * self.loss_weight


class STOI_Loss(nn.Module):
    def __init__(self, extended=False, plus=True, FFT_size=256):
        super(STOI_Loss, self).__init__()
        """
        Differentiable Short-Time Objective Intelligibility (STOI) Loss
        https://ieeexplore.ieee.org/document/5495701
        Detailed in metrics.torch_stoi 
        """
        self.loss_func = NegSTOILoss(
            sample_rate=16000, extended=extended, plus=plus, FFT_size=FFT_size
        )

    def forward(self, rec_amp, spec_amp, on_stage, suffix="stoi", tracker=None):
        stoi_loss = self.loss_func(rec_amp, spec_amp, on_stage).mean()
        tracker.update(dict({"Lae_" + suffix: stoi_loss}))
        return stoi_loss

def MTF_pytorch(S):
    """
    Compute the Modulation Transfer Function (MTF) of a spectrogram. 
    Then we could use it to compute the MTF loss between original and reconstructed spectrogram.
    """
    # S is linear spectrogram
    F = torch.fft.fftshift(
        torch.log(torch.abs(torch.fft.fft2(torch.log(torch.abs(S)))))
    )
    # 128, 256
    F_tmp = F[
        :,
        :,
        F.shape[2] // 2 - 15 : F.shape[2] // 2 + 15,
        5 : F.shape[3] // 2,
    ]
    F_horizontal = F_tmp[:, :, :, -10:] * 2
    F_vertical = F_tmp[:, :, F_tmp.shape[2] // 2 - 5 : F_tmp.shape[2] // 2 + 5] * 2
    return F_tmp, F_horizontal, F_vertical

class Model(nn.Module):
    def __init__(
        self,
        generator="",
        encoder="",
        ecog_decoder_name="",
        spec_chans=128,
        n_formants=2,
        n_formants_noise=2,
        n_formants_ecog=2,
        n_fft=256,
        noise_db=-50,
        max_db=22.5,
        wavebased=False,
        with_ecog=False,
        with_encoder2=False,
        ghm_loss=True,
        power_synth=True,
        apply_flooding=False,
        normed_mask=False,
        dummy_formant=False,
        Visualize=False,
        key=None,
        index=None,
        A2A=False,
        do_mel_guide=True,
        noise_from_data=False,
        specsup=True,
        causal=False,
        anticausal=False,
        pre_articulate=False,
        alpha_sup=False,
        ld_loss_weight=True,
        alpha_loss_weight=True,
        consonant_loss_weight=True,
        component_regression=False,
        amp_formant_loss_weight=True,
        freq_single_formant_loss_weight=True,
        amp_minmax=False,
        distill=False,
        amp_energy=False,
        network_db=False,
        f0_midi=False,
        alpha_db=False,
        delta_time=False,
        delta_freq=False,
        cumsum=False,
        learned_mask=False,
        n_filter_samples=40,
        dynamic_filter_shape=False,
        learnedbandwidth=False,
        patient="NY742",
        rdropout=0,  #https://arxiv.org/abs/2106.14448
        return_filtershape=False,
        spec_fr=125,
        gender_patient="Female",
        reverse_order=True,
        larger_capacity=False,
        unified=False,
        use_stoi=False
    ):
        super(Model, self).__init__()
        self.component_regression = component_regression
        self.use_stoi = use_stoi
        self.amp_minmax = amp_minmax
        self.amp_energy = amp_energy
        self.f0_midi = f0_midi
        self.alpha_db = alpha_db
        self.ld_loss_weight = ld_loss_weight
        self.freq_single_formant_loss_weight = freq_single_formant_loss_weight
        self.amp_formant_loss_weight = amp_formant_loss_weight
        self.alpha_loss_weight = alpha_loss_weight
        self.consonant_loss_weight = consonant_loss_weight
        self.spec_chans = spec_chans
        self.with_ecog = with_ecog
        self.with_encoder2 = with_encoder2
        self.ecog_decoder_name = ecog_decoder_name
        self.n_formants_ecog = n_formants_ecog
        self.n_formants = n_formants
        self.wavebased = wavebased
        self.n_fft = n_fft
        self.n_mels = spec_chans
        self.do_mel_guide = do_mel_guide
        self.noise_db = noise_db
        self.spec_sup = specsup
        self.max_db = max_db
        self.n_formants_noise = n_formants_noise
        self.power_synth = power_synth
        self.apply_flooding = apply_flooding
        self.Visualize = Visualize
        self.key = key
        self.index = index
        self.A2A = A2A
        self.return_value = True
        self.alpha_sup = alpha_sup
        self.return_guide = False
        self.delta_time = delta_time
        self.delta_freq = delta_freq
        self.cumsum = cumsum
        self.distill = distill
        self.patient = patient
        self.rdropout = rdropout
        self.return_cc = False
        self.cc_method = "None"
        self.noise_from_data = noise_from_data
        self.weighted_vis = True
        self.return_filtershape = return_filtershape
        self.formant_freq_limits_abs = torch.tensor(
            [950.0, 3400.0, 3800.0, 5000.0, 6000.0, 7000.0]
        ).reshape(
            [1, 6, 1]
        )  # freq difference
        self.formant_freq_limits_abs_low = torch.tensor(
            [300.0, 700.0, 1800.0, 3400, 5000.0, 6000.0]
        ).reshape(
            [1, 6, 1]
        )  # freq difference
        print("patient in model", patient)
        self.decoder = GENERATORS[generator](
            n_mels=spec_chans,
            k=40,
            wavebased=wavebased,
            n_fft=n_fft,
            noise_db=noise_db,
            max_db=max_db,
            noise_from_data=noise_from_data,
            return_wave=False,
            power_synth=power_synth,
            n_formants=n_formants,
            normed_mask=normed_mask,
            dummy_formant=dummy_formant,
            learned_mask=learned_mask,
            n_filter_samples=n_filter_samples,
            dynamic_filter_shape=dynamic_filter_shape,
            learnedbandwidth=learnedbandwidth,
            return_filtershape=return_filtershape,
            spec_fr=spec_fr,
            reverse_order=reverse_order
        )
        if do_mel_guide:
            self.decoder_mel = GENERATORS[generator](
                n_mels=spec_chans,
                k=40,
                wavebased=False,
                n_fft=n_fft,
                noise_db=noise_db,
                max_db=max_db,
                add_bgnoise=False,
            )
        self.encoder = ENCODERS[encoder](
            n_mels=spec_chans,
            n_formants=n_formants,
            n_formants_noise=n_formants_noise,
            wavebased=wavebased,
            hop_length=128,
            n_fft=n_fft,
            noise_db=noise_db,
            max_db=max_db,
            power_synth=power_synth,
            patient=patient,
            gender_patient=gender_patient,
            larger_capacity=larger_capacity,
            unified=unified,
        )

        if with_ecog:
            self.ecog_decoder = ECOG_DECODER[ecog_decoder_name](
                    n_mels=spec_chans,
                    n_formants=n_formants_ecog,
                    network_db=network_db,
                    causal=causal,
                    anticausal=anticausal,
                    pre_articulate=pre_articulate,
                )
        self.ghm_loss = ghm_loss
        self.stoi_loss_female = STOI_Loss(extended=False, plus=True, FFT_size=256)
        self.stoi_loss_male = STOI_Loss(extended=False, plus=True, FFT_size=512)

    def noise_dist_init(self, dist):
        with torch.no_grad():
            self.decoder.noise_dist = dist.reshape([1, 1, 1, dist.shape[0]])

    def generate_fromecog(
        self,
        ecog=None,
        return_components=False,
        onstage=None,
    ):
        '''
        Use the ECoG decoder to generate speech parameters
        and then generate spectrogram using speech synthesizer
        '''
        components = self.ecog_decoder(ecog)
        rec = self.decoder.forward(components, onstage)
        if return_components:
            return rec, components
        else:
            return rec

    def encode(
        self,
        spec,
        x_denoise=None,
        duomask=False,
        noise_level=None,
        x_amp=None,
        gender="Female",
    ):
        '''encode the spectrogram to components using the speech encoder'''
        components = self.encoder(
            spec,
            x_denoise=x_denoise,
            duomask=duomask,
            noise_level=noise_level if self.noise_from_data else None,
            x_amp=x_amp,
            gender=gender,
        )
        return components

    def lae(
        self,
        spec,
        rec,
        db=True,
        amp=True,
        tracker=None,
        GHM=False,
        suffix="",
        MTF=False,
        reweight = 1
    ):
        """
        given rec, spec as reconstructed and original spectrogram, compute the difference loss including:
        1. L2 GHMR loss in dB scale
        2. L2 GHMR loss in amp scale
        3. delta loss in time domain to smooth neighboring frames
        4. delta loss in freq domain to smooth neighboring freqs
        5. cumsum loss to ensure more attention to low freqs
        6. L2 GHMR loss between MTF of original and reconstructed spectrogram
        """
        if amp:
            spec_amp = amplitude(spec, noise_db=self.noise_db, max_db=self.max_db)
            rec_amp = amplitude(rec, noise_db=self.noise_db, max_db=self.max_db)
            spec_amp_ = spec_amp
            rec_amp_ = rec_amp
            if GHM:
                Lae_a = self.ghm_loss(
                    rec_amp_, spec_amp_, torch.ones(spec_amp_), reweight=reweight
                )
                Lae_a_l2 = torch.tensor([0.0])
            else:
                Lae_a = (spec_amp_ - rec_amp_).abs().mean()
                Lae_a_l2 = torch.sqrt((spec_amp_ - rec_amp_) ** 2 + 1e-6).mean()
        else:
            Lae_a = torch.tensor(0.0)
            Lae_a_l2 = torch.tensor(0.0)
        if tracker is not None:
            tracker.update(
                dict({"Lae_a" + suffix: Lae_a, "Lae_a_l2" + suffix: Lae_a_l2})
            )
        if db:
            if GHM:
                Lae_db = self.ghm_loss(rec, spec, torch.ones(spec), reweight=reweight)
                Lae_db_l2 = torch.tensor([0.0])
            else:
                Lae_db = (spec - rec).abs().mean()
                Lae_db_l2 = torch.sqrt((spec - rec) ** 2 + 1e-6).mean()
        else:
            Lae_db = torch.tensor(0.0)
            Lae_db_l2 = torch.tensor(0.0)
        if MTF:
            spec_amp = amplitude(spec, noise_db=self.noise_db, max_db=self.max_db)
            rec_amp = amplitude(rec, noise_db=self.noise_db, max_db=self.max_db)
            F_tmp, F_horizontal, F_vertical = MTF_pytorch(spec_amp)
            F_tmp_rec, F_horizontal_rec, F_vertical_rec = MTF_pytorch(rec_amp)
            Lae_mtf = (
                (F_tmp - F_tmp_rec).abs().mean()
                + (F_horizontal - F_horizontal_rec).abs().mean()
                + (F_vertical - F_vertical_rec).abs().mean()
            )
        else:
            Lae_mtf = torch.tensor(0.0)

        if self.delta_time:
            loss_delta_time = (
                (diff_dim(rec_amp, axis=2) - diff_dim(spec_amp, axis=2)).abs().mean()
            )
            if tracker is not None:
                tracker.update(dict({"Lae_delta_time" + suffix: Lae_delta_time}))
        else:
            loss_delta_time = torch.tensor(0.0)
        if self.delta_freq:
            loss_delta_freq = (
                (diff_dim(rec_amp, axis=1) - diff_dim(spec_amp, axis=1)).abs().mean()
            )
            if tracker is not None:
                tracker.update(dict({"Lae_delta_time" + suffix: loss_delta_freq}))
        else:
            loss_delta_freq = torch.tensor(0.0)
        if self.cumsum:
            loss_cumsum = (
                (cumsum(rec_amp, axis=1) - cumsum(spec_amp, axis=1)).abs().mean()
            )
            if tracker is not None:
                tracker.update(dict({"Lae_delta_time" + suffix: loss_cumsum}))
        else:
            loss_cumsum = torch.tensor(0.0)
        
        if tracker is not None:
            tracker.update(
                dict({"Lae_db" + suffix: Lae_db, "Lae_db_l2" + suffix: Lae_db_l2})
            )
        return (
            Lae_a
            + Lae_db / 2.0
            + Lae_mtf
            + loss_delta_time
            + loss_delta_freq
            + loss_cumsum
        )

    def flooding(self, loss, beta):
        '''flooding loss function https://proceedings.mlr.press/v119/ishida20a/ishida20a.pdf'''
        if self.apply_flooding:
            return (loss - beta).abs() + beta
        else:
            return loss
    
    def run_components_loss(
        self,
        rec,
        spec,
        tracker,
        encoder_guide,
        components_ecog,
        components_guide,
        alpha,
        betas,
        on_stage_wider,
        on_stage,
    ):
        """
        the core function for ECoG to Speech decoding (step2)
        calculate component loss with spectrogram encoded components and ECoG decoded components
        """
        if self.spec_sup:
            '''
            original and reconstructed linear spectrogram loss using function `lae`
            '''
            Lrec = 80 * self.lae(
                    rec, spec, tracker=tracker, amp=False, suffix="1", MTF=False
                )
        else:
            Lrec = torch.tensor([0.0])

        #original and reconstructed mel spectrogram loss using function `lae`
        spec_amp = amplitude(spec, self.noise_db, self.max_db).transpose(-2, -1)
        rec_amp = amplitude(rec, self.noise_db, self.max_db).transpose(-2, -1)
        spec_mel = to_db(
            torchaudio.transforms.MelScale(f_max=8000, n_stft=self.n_fft)(
                spec_amp
            ).transpose(-2, -1),
            self.noise_db,
            self.max_db,
        )
        rec_mel = to_db(
            torchaudio.transforms.MelScale(f_max=8000, n_stft=self.n_fft)(
                rec_amp
            ).transpose(-2, -1),
            self.noise_db,
            self.max_db,
        )
        Lrec += 80 * self.lae(rec_mel, spec_mel, tracker=tracker, amp=False, suffix="2")
        
        #stoi loss
        if self.use_stoi:
            if spec_amp.shape[-2] == 256:
                stoi_loss = (
                    self.stoi_loss_female(
                        rec_amp, spec_amp, on_stage, suffix="stoi", tracker=tracker
                    )
                    * 10
                )
            else:
                stoi_loss = (
                    self.stoi_loss_male(
                        rec_amp, spec_amp, on_stage, suffix="stoi", tracker=tracker
                    )
                    * 10
                )
            Lrec += stoi_loss
        tracker.update(dict(Lrec=Lrec))
        
        '''
        loss function for ECoG to speech paramater decoding
        computed between speech to speech encoded guidance speecch parameters
        and ECoG decoded speech parameters
        '''
        Lcomp = 0 
        
        # define a bunch of weights for different speech parameters
        consonant_weight = 1
        if self.power_synth:
            loudness_db = torchaudio.transforms.AmplitudeToDB()(
                components_guide["loudness"]
            )
            loudness_db_norm = (loudness_db.clamp(min=-70) + 70) / 50
            if self.amp_minmax:
                hamon_amp_minmax_weight = (
                    torch.tensor([2, 5, 4, 0.5, 0.5, 0.5])
                    .view(1, -1, 1)
                    .expand_as(components_guide["amplitude_formants_hamon"])
                )
                amplitude_formants_hamon_db_norm = torch.zeros_like(
                    components_guide["amplitude_formants_hamon"]
                )
                for freqs in range(amplitude_formants_hamon_db_norm.shape[1]):
                    amplitude_formants_hamon_db_norm[:, freqs] = minmaxscale(
                        components_guide["amplitude_formants_hamon"][:, freqs]
                    )
                amplitude_formants_hamon_db_norm = (
                    amplitude_formants_hamon_db_norm * hamon_amp_minmax_weight
                )

                noise_amp_minmax_weight = (
                    torch.tensor([2, 0.5, 0.5, 0.5, 0.5, 0.5, 10])
                    .view(1, -1, 1)
                    .expand_as(components_guide["amplitude_formants_noise"])
                )
                amplitude_formants_noise_db_norm = torch.zeros_like(
                    components_guide["amplitude_formants_noise"]
                )
                for freqs in range(amplitude_formants_noise_db_norm.shape[1]):
                    amplitude_formants_noise_db_norm[:, freqs] = minmaxscale_ref(
                        components_guide["amplitude_formants_noise"][:, freqs],
                        data_select=components_guide["amplitude_formants_noise"][
                            :, freqs
                        ][
                            torch.where(
                                components_guide["amplitudes"][:, 0:1].squeeze(1)
                                < 0.5
                            )
                        ],
                    )
                amplitude_formants_noise_db_norm = (
                    amplitude_formants_noise_db_norm * noise_amp_minmax_weight
                )

            else:
                amplitude_formants_hamon_db = torchaudio.transforms.AmplitudeToDB()(
                    components_guide["amplitude_formants_hamon"]
                )
                amplitude_formants_hamon_db_norm = (
                    amplitude_formants_hamon_db.clamp(min=-70) + 70
                ) / 50
                amplitude_formants_noise_db = torchaudio.transforms.AmplitudeToDB()(
                    components_guide["amplitude_formants_noise"]
                )
                amplitude_formants_noise_db_norm = (
                    amplitude_formants_noise_db.clamp(min=-70) + 70
                ) / 50
        else:
            loudness_db = torchaudio.transforms.AmplitudeToDB()(
                components_guide["loudness"]
            )
            loudness_db_norm = (loudness_db.clamp(min=-70) + 70) / 50

            if self.amp_minmax:
                hamon_amp_minmax_weight = (
                    torch.tensor([2, 5, 4, 0.5, 0.5, 0.5])
                    .view(1, -1, 1)
                    .expand_as(components_guide["amplitude_formants_hamon"])
                )
                amplitude_formants_hamon_db_norm = torch.zeros_like(
                    components_guide["amplitude_formants_hamon"]
                )
                for freqs in range(amplitude_formants_hamon_db_norm.shape[1]):
                    amplitude_formants_hamon_db_norm[:, freqs] = minmaxscale(
                        components_guide["amplitude_formants_hamon"][:, freqs]
                    )
                amplitude_formants_hamon_db_norm = (
                    amplitude_formants_hamon_db_norm * hamon_amp_minmax_weight
                )

                noise_amp_minmax_weight = (
                    torch.tensor([2, 0.5, 0.5, 0.5, 0.5, 0.5, 10])
                    .view(1, -1, 1)
                    .expand_as(components_guide["amplitude_formants_noise"])
                )
                amplitude_formants_noise_db_norm = torch.zeros_like(
                    components_guide["amplitude_formants_noise"]
                )
                for freqs in range(amplitude_formants_noise_db_norm.shape[1]):
                    amplitude_formants_noise_db_norm[:, freqs] = minmaxscale_ref(
                        components_guide["amplitude_formants_noise"][:, freqs],
                        data_select=components_guide["amplitude_formants_noise"][
                            :, freqs
                        ][
                            torch.where(
                                components_guide["amplitudes"][:, 0:1].squeeze(1)
                                < 0.5
                            )
                        ],
                    )
                amplitude_formants_noise_db_norm = (
                    amplitude_formants_noise_db_norm * noise_amp_minmax_weight
                )

            else:
                amplitude_formants_hamon_db = torchaudio.transforms.AmplitudeToDB()(
                    components_guide["amplitude_formants_hamon"]
                )
                amplitude_formants_hamon_db_norm = (
                    amplitude_formants_hamon_db.clamp(min=-70) + 70
                ) / 50
                amplitude_formants_noise_db = torchaudio.transforms.AmplitudeToDB()(
                    components_guide["amplitude_formants_noise"]
                )
                amplitude_formants_noise_db_norm = (
                    amplitude_formants_noise_db.clamp(min=-70) + 70
                ) / 50
        loudness_db_norm_weight = loudness_db_norm if self.ld_loss_weight else 1
        if self.alpha_loss_weight:
            alpha_formant_weight = components_guide["amplitudes"][:, 0:1]
            alpha_noise_weight = components_guide["amplitudes"][:, 1:2]
        else:
            alpha_formant_weight = 1
            alpha_noise_weight = 1
        if self.amp_formant_loss_weight:
            amplitude_formants_hamon_db_norm_weight = (
                amplitude_formants_hamon_db_norm
            )
            amplitude_formants_noise_db_norm_weight = (
                amplitude_formants_noise_db_norm
            )
        else:
            amplitude_formants_hamon_db_norm_weight = 1
            amplitude_formants_noise_db_norm_weight = 1
        if self.freq_single_formant_loss_weight:
            freq_single_formant_weight = (
                torch.tensor([6, 3, 2, 1, 1, 1])
                .view(1, -1, 1)
                .expand_as(
                    components_guide["amplitude_formants_hamon"][
                        :, : self.n_formants_ecog
                    ]
                )
            )
        else:
            freq_single_formant_weight = 1

        if self.consonant_loss_weight:
            consonant_weight = 100 * (
                torch.sign(components_guide["amplitudes"][:, 1:] - 0.5) * 0.5 + 0.5
            )
        else:
            consonant_weight = 1
        
        #we calculate the loss for each speech parameter separately
        for key in [
            "loudness",
            "f0_hz",
            "amplitudes",
            "amplitude_formants_hamon",
            "freq_formants_hamon",
            "amplitude_formants_noise",
            "freq_formants_noise",
            "bandwidth_formants_noise_hz",
        ]:
            if key == "loudness":
                if self.power_synth:
                    loudness_db_norm_ecog = (
                        torchaudio.transforms.AmplitudeToDB()(components_ecog[key])
                        + 70
                    ) / 50
                else:
                    loudness_db_norm_ecog = (
                        torchaudio.transforms.AmplitudeToDB()(components_ecog[key])
                        + 70
                    ) / 50
                
                diff = (
                    alpha["loudness"]
                    * 150
                    * torch.mean(
                        (loudness_db_norm - loudness_db_norm_ecog) ** 2
                    )
                )
                tracker.update(
                    {
                        "loudness_metric": torch.mean(
                            (loudness_db_norm - loudness_db_norm_ecog) ** 2
                            * on_stage_wider
                        )
                    }
                )

            if key == "f0_hz":
                if self.f0_midi:
                    difftmp = (
                        hz_to_midi(components_guide[key]) / 4
                        - hz_to_midi(components_ecog[key]) / 4
                    )
                else:
                    difftmp = (
                        components_guide[key] / 40
                        - components_ecog[key] / 40
                    )
                diff = (
                    alpha["f0_hz"]
                    * 0.3
                    * torch.mean(difftmp**2 * on_stage_wider * loudness_db_norm)
                )
                diff = self.flooding(diff, alpha["f0_hz"] * betas["f0_hz"])
                tracker.update(
                    {
                        "f0_metric": torch.mean(
                            (
                                components_guide["f0_hz"] / 40
                                - components_ecog["f0_hz"] / 40
                            )
                            ** 2
                            * on_stage_wider
                            * loudness_db_norm
                        ) } )

            if key in ["amplitudes"]:
                weight = on_stage_wider * loudness_db_norm_weight
                tmp_target = components_guide[key]
                tmp_ecog = components_ecog[key]
                if self.alpha_db:
                    tmp_target = df_norm_torch(tmp_target)[1]
                    tmp_ecog = df_norm_torch(tmp_ecog)[1]

                if self.ghm_loss:
                    diff = (
                        alpha["amplitudes"]
                        * 540 * self.lae(tmp_target, tmp_ecog, reweight=weight)
                    )
                else:
                    diff = (
                        alpha["amplitudes"]
                        * 180 * torch.mean((tmp_target - tmp_ecog) ** 2 * weight)
                    )
                diff = self.flooding(
                    diff, alpha["amplitudes"] * betas["amplitudes"]
                )
                tracker.update(
                    {
                        "amplitudes_metric": torch.mean(
                            (
                                components_guide["amplitudes"]
                                - components_ecog["amplitudes"]
                            ) ** 2 * weight
                        ) } )

            if key in ["amplitude_formants_hamon"]:
                weight = (
                    alpha_formant_weight
                    * on_stage_wider
                    * consonant_weight
                    * loudness_db_norm_weight
                )
                
                
                if self.amp_energy == 1:
                    tmp_diff = (
                        df_norm_torch(
                            (
                                components_guide["loudness"].expand_as(
                                    components_guide[key][
                                        :, : self.n_formants_ecog
                                    ]
                                )
                                * components_guide[key][
                                    :, : self.n_formants_ecog
                                ]
                            )
                        )[1]
                        - df_norm_torch(
                            (
                                components_guide["loudness"].expand_as(
                                    components_ecog[key][
                                        :, : self.n_formants_ecog
                                    ]
                                )
                                * components_ecog[key][
                                    :, : self.n_formants_ecog
                                ]
                            )
                        )[1]
                    ) ** 2 * freq_single_formant_weight
                elif self.amp_energy == 2:
                    tmp_diff = (
                        df_norm_torch(
                            components_guide[key][:, : self.n_formants_ecog]
                        )[1]
                        - df_norm_torch(
                            components_ecog[key][:, : self.n_formants_ecog]
                        )[1]
                    ) ** 2 * freq_single_formant_weight
                elif self.amp_energy == 3:
                    tmp_diff = (
                        df_norm_torch(
                            components_guide[key][:, : self.n_formants_ecog]
                        )[1]
                        - df_norm_torch(
                            components_ecog[key][:, : self.n_formants_ecog]
                        )[1]
                    ) ** 2 * freq_single_formant_weight + (
                        (
                            components_guide[key][:, : self.n_formants_ecog]
                            - components_ecog[key][:, : self.n_formants_ecog]
                        )
                        ** 2
                        * freq_single_formant_weight
                    )
                else:
                    tmp_diff = (
                        components_guide[key][:, : self.n_formants_ecog]
                        - components_ecog[key][:, : self.n_formants_ecog]
                    ) ** 2 * freq_single_formant_weight
                diff = (
                    alpha["amplitude_formants_hamon"]
                    * 400
                    * torch.mean(tmp_diff * weight)
                )
                diff = self.flooding(
                    diff,
                    alpha["amplitude_formants_hamon"]
                    * betas["amplitude_formants_hamon"],
                )
                tracker.update(
                    {
                        "amplitude_formants_hamon_metric": torch.mean(
                            (
                                components_guide["amplitude_formants_hamon"][
                                    :, : self.n_formants_ecog
                                ]
                                - components_ecog["amplitude_formants_hamon"]
                            )
                            ** 2
                            * weight
                        )
                    }
                )
            if key in ["freq_formants_hamon"]:
                weight = (
                    alpha_formant_weight
                    * on_stage_wider
                    * consonant_weight
                    * loudness_db_norm_weight
                )

                tmp_diff = (
                    components_guide[key][:, : self.n_formants_ecog]
                    - components_ecog[key][:, : self.n_formants_ecog]
                ) ** 2 * freq_single_formant_weight
                diff = (
                    alpha["freq_formants_hamon"]
                    * 300
                    * torch.mean(
                        tmp_diff
                        * amplitude_formants_hamon_db_norm_weight
                        * weight
                    )
                )
                diff = self.flooding(
                    diff,
                    alpha["freq_formants_hamon"] * betas["freq_formants_hamon"],
                )
                tracker.update(
                    {
                        "freq_formants_hamon_hz_metric_2": torch.mean(
                            (
                                components_guide["freq_formants_hamon_hz"][:, :2]
                                / 400
                                - components_ecog["freq_formants_hamon_hz"][:, :2]
                                / 400
                            )
                            ** 2
                            * weight
                        )
                    }
                )
                tracker.update(
                    {
                        "freq_formants_hamon_hz_metric_"
                        + str(self.n_formants_ecog): torch.mean(
                            (
                                components_guide["freq_formants_hamon_hz"][
                                    :, : self.n_formants_ecog
                                ]
                                / 400
                                - components_ecog["freq_formants_hamon_hz"][
                                    :, : self.n_formants_ecog
                                ]
                                / 400
                            )
                            ** 2
                            * weight
                        )
                    }
                )
            if key in ["amplitude_formants_noise"]:
                weight = (
                    alpha_noise_weight
                    * on_stage_wider
                    * consonant_weight
                    * loudness_db_norm_weight
                )

                tmp_target = torch.cat(
                    [
                        components_guide[key][:, : self.n_formants_ecog],
                        components_guide[key][:, -self.n_formants_noise :],
                    ],
                    dim=1,
                )
                tmp_ecog = components_ecog[key]
                if self.amp_energy == 1:
                    tmp_diff = (
                        df_norm_torch(
                            (
                                components_guide["loudness"].expand_as(
                                    tmp_target
                                )
                                * tmp_target
                            )
                        )[1]
                        - df_norm_torch(
                            (
                                components_guide["loudness"].expand_as(tmp_ecog)
                                * tmp_ecog
                            )
                        )[1]
                    ) ** 2 * freq_single_formant_weight
                elif self.amp_energy == 2:
                    tmp_diff = (
                        df_norm_torch(tmp_target)[1]
                        - df_norm_torch(tmp_ecog)[1]
                    ) ** 2 * freq_single_formant_weight
                elif self.amp_energy == 3:
                    tmp_diff = (
                        df_norm_torch(tmp_target)[1]
                        - df_norm_torch(tmp_ecog)[1]
                    ) ** 2 * freq_single_formant_weight + (
                        tmp_target - tmp_ecog
                    ) ** 2 * freq_single_formant_weight
                else:
                    tmp_diff = (
                        tmp_target - tmp_ecog
                    ) ** 2 * freq_single_formant_weight
                diff = (
                    alpha["amplitude_formants_noise"]
                    * 400
                    * torch.mean(tmp_diff * weight)
                )
                diff = self.flooding(
                    diff,
                    alpha["amplitude_formants_noise"]
                    * betas["amplitude_formants_noise"],
                )
                tracker.update(
                    {
                        "amplitude_formants_noise_metric": torch.mean(
                            (
                                torch.cat(
                                    [
                                        components_guide[key][
                                            :, : self.n_formants_ecog
                                        ],
                                        components_guide[key][
                                            :, -self.n_formants_noise :
                                        ],
                                    ],
                                    dim=1,
                                )
                                - components_ecog[key]
                            )
                            ** 2
                            * weight
                        )
                    }
                )

            if key in ["freq_formants_noise"]:
                weight = (
                    alpha_noise_weight
                    * on_stage_wider
                    * consonant_weight
                    * loudness_db_norm_weight
                )

                diff = (
                    alpha["freq_formants_noise"]
                    * 12000
                    * torch.mean(
                        (
                            components_guide[key][:, -self.n_formants_noise :]
                            - components_ecog[key][:, -self.n_formants_noise :]
                        )
                        ** 2
                        * weight
                        * amplitude_formants_noise_db_norm_weight
                    )
                )
                diff = self.flooding(
                    diff,
                    alpha["freq_formants_noise"] * betas["freq_formants_noise"],
                )
                tracker.update(
                    {
                        "freq_formants_noise_metric": torch.mean(
                            (
                                components_guide["freq_formants_noise_hz"][
                                    :, -self.n_formants_noise :
                                ]
                                / 400
                                - components_ecog["freq_formants_noise_hz"][
                                    :, -self.n_formants_noise :
                                ]
                                / 400
                            ) ** 2 * weight
                        )
                    }
                )

            if key in ["bandwidth_formants_noise_hz"]:
                weight = (
                    alpha_noise_weight
                    * on_stage_wider
                    * consonant_weight
                    * loudness_db_norm_weight
                )
                diff = (
                        alpha["bandwidth_formants_noise_hz"]
                        * 3
                        * torch.mean(
                            (
                                components_guide[key][:, -self.n_formants_noise :]
                                / 400
                                - components_ecog[key][:, -self.n_formants_noise :]
                                / 400
                            ) ** 2 * weight
                        )
                    )
                diff = self.flooding(
                    diff,
                    alpha["bandwidth_formants_noise_hz"]
                    * betas["bandwidth_formants_noise_hz"],
                )
                tracker.update(
                    {
                        "bandwidth_formants_noise_hz_metric": torch.mean(
                            (
                                components_guide[key][:, -self.n_formants_noise :]
                                / 400
                                - components_ecog[key][:, -self.n_formants_noise :]
                                / 400
                            ) ** 2 * weight
                        )
                    }
                )

            tracker.update({key: diff})
            Lcomp += diff

        if self.component_regression:
            # we only regress to speech parameters without spectrogram decoding loss
            Loss = Lcomp
        else:
            Loss = Lrec + Lcomp

        hamonic_components_diff = (
            compdiffd2(components_ecog["freq_formants_hamon_hz"] * 1.5)
            + compdiffd2(components_ecog["f0_hz"] * 2)
            + compdiff(
                components_ecog["bandwidth_formants_noise_hz"][
                    :, components_ecog["freq_formants_hamon_hz"].shape[1] :
                ] / 5
            )
            + compdiff(
                components_ecog["freq_formants_noise_hz"][
                    :, components_ecog["freq_formants_hamon_hz"].shape[1] :
                ] / 5
            )
            + compdiff(components_ecog["amplitudes"]) * 750.0
        )
        Ldiff = torch.mean(hamonic_components_diff) / 2000.0
        tracker.update(dict(Ldiff=Ldiff))
        Loss += Ldiff

        freq_linear_reweighting = 1
        thres = (
            int(hz2ind(4000, self.n_fft))
            if self.wavebased
            else mel_scale(self.spec_chans, 4000, pt=False).astype(np.int32)
        )
        explosive = (
            torch.sign(
                torch.mean((spec * freq_linear_reweighting)[..., thres:], dim=-1)
                - torch.mean((spec * freq_linear_reweighting)[..., :thres], dim=-1)
            )
            * 0.5
            + 0.5
        )
        Lexp = (
            torch.mean(
                (
                    components_ecog["amplitudes"][:, 0:1]
                    - components_ecog["amplitudes"][:, 1:2]
                )
                * explosive
            )
            * 10
        )
        tracker.update(dict(Lexp=Lexp))
        Loss += Lexp

        Lfreqorder = torch.mean(
            F.relu(
                components_ecog["freq_formants_hamon_hz"][:, :-1]
                - components_ecog["freq_formants_hamon_hz"][:, 1:]
            )
        )
        Loss += Lfreqorder
        return Loss, tracker

    
    def run_a2a_loss(self, rec,
        spec,x_amp_from_denoise,x_denoise,x_mel,x_amp,
        components, tracker,
        voice,unvoice,semivoice,plosive,fricative,
        duomask,hamonic_bias,pitch_aug,epoch_current,
        gender,pitch_label,formant_label,
        on_stage_wider,
        on_stage):
        """
        the core function for Speech to Speech decoding (step1)
        calculate unsupervised and supervised losses within a2a
        """
        freq_cord2 = torch.arange(self.spec_chans + 1).reshape(
            [1, 1, 1, self.spec_chans + 1]
        ) / (1.0 * self.spec_chans)
        freq_linear_reweighting = (
            1
            if self.wavebased
            else (
                inverse_mel_scale(freq_cord2[..., 1:])
                - inverse_mel_scale(freq_cord2[..., :-1])
            )
            / 440
            * 7
        )
        Lae = 8 * self.lae(
            rec * freq_linear_reweighting,
            spec * freq_linear_reweighting,
            tracker=tracker,
            suffix="1",
        )
        if self.wavebased:
            spec_amp = amplitude(spec, self.noise_db, self.max_db).transpose(
                -2, -1
            )
            rec_amp = amplitude(rec, self.noise_db, self.max_db).transpose(
                -2, -1
            )
            #STOI loss
            if self.use_stoi:
                if spec_amp.shape[-2] == 256:
                    stoi_loss = (
                        self.stoi_loss_female(
                            rec_amp,
                            spec_amp,
                            on_stage,
                            suffix="stoi",
                            tracker=tracker,
                        ) * 10 )
                else:
                    stoi_loss = (
                        self.stoi_loss_male(
                            rec_amp,
                            spec_amp,
                            on_stage,
                            suffix="stoi",
                            tracker=tracker,
                        ) * 10 )
                Lae += stoi_loss

            freq_cord2 = torch.arange(128 + 1).reshape([1, 1, 1, 128 + 1]) /  128
            freq_linear_reweighting2 = (
                (
                    inverse_mel_scale(freq_cord2[..., 1:])
                    - inverse_mel_scale(freq_cord2[..., :-1])
                )
                / 440 * 7 )
            spec_mel = to_db(
                torchaudio.transforms.MelScale(f_max=8000, n_stft=self.n_fft)(
                    spec_amp
                ).transpose(-2, -1),
                self.noise_db,
                self.max_db,
            )
            rec_mel = to_db(
                torchaudio.transforms.MelScale(f_max=8000, n_stft=self.n_fft)(
                    rec_amp
                ).transpose(-2, -1),
                self.noise_db,
                self.max_db,
            )
            #spectrogram reconstruction loss (mel-scale)
            Lae += 8 * self.lae(
                rec_mel * freq_linear_reweighting2,
                spec_mel * freq_linear_reweighting2,
                tracker=tracker,
                amp=False,
                suffix="2",
            )

        #if we have alpha additional supervision loss (needs careful labelling of phonemes which is not always available)
        if self.alpha_sup and voice is not None and unvoice is not None:
            Lsilence = 10 * (
                (components["amplitudes"][:, 0:1][on_stage == 0] - 0.0) ** 2
            )
            Lsilence = (
                Lsilence.mean() if len(Lsilence) != 0 else torch.tensor(0.0)
            )
            Lvoice = 10 * (
                (components["amplitudes"][:, 0:1][voice == 1] - 1.0) ** 2
            )
            Lvoice = Lvoice.mean() if len(Lvoice) != 0 else torch.tensor(0.0)
            Lunvoice = 10 * (
                (components["amplitudes"][:, 1:2][unvoice == 1] - 1.0) ** 2
            )
            Lunvoice = (
                Lunvoice.mean() if len(Lunvoice) != 0 else torch.tensor(0.0)
            )
            Lsemivoice = 10 * (
                F.relu(0.3 - components["amplitudes"][:, 1:2][semivoice == 1])
                ** 2
            )
        
            Lsemivoice = (
                Lsemivoice.mean() if len(Lsemivoice) != 0 else torch.tensor(0.0)
            )
            Lplosive = 4 * (
                (
                    components["freq_formants_noise_hz"][:, -1:][plosive == 1]
                    / 4000 - 4000.0 / 4000
                )
                ** 2
            )
            Lplosive = (
                Lplosive.mean() if len(Lplosive) != 0 else torch.tensor(0.0)
            )
            Lplosiveband = 4 * (
                (
                    components["bandwidth_formants_noise_hz"][:, -1:][
                        plosive == 1
                    ]
                    / 8000 - 8000.0 / 8000
                ) ** 2
            )
            Lplosiveband = (
                Lplosiveband.mean()
                if len(Lplosiveband) != 0
                else torch.tensor(0.0)
            )
            Lfricativeband = 4 * (
                F.relu(
                    components["bandwidth_formants_noise_hz"][:, -1:][
                        fricative == 1
                    ] / 1500.0 - 1500.0 / 1500
                )
                ** 2
            ) 
            Lfricativeband = (
                Lfricativeband.mean()
                if len(Lfricativeband) != 0
                else torch.tensor(0.0)
            )
            Lfricative = torch.tensor(0.0)

            Lae += (
                Lvoice
                + Lunvoice
                + Lsemivoice
                + Lplosive
                + Lfricative
                + Lsilence
                + Lplosiveband
                + Lfricativeband
            )
            tracker.update(
                dict(
                    Lvoice=Lvoice,
                    Lunvoice=Lunvoice,
                    Lsemivoice=Lsemivoice,
                    Lplosive=Lplosive,
                    Lfricative=Lfricative,
                    Lplosiveband=Lplosiveband,
                    Lfricativeband=Lfricativeband,
                )
            )

        #loudness loss which discourages silent parts loudness values be too high
        Lloudness = (
            10**6 * (components["loudness"] * (1 - on_stage_wider)).mean()
        )
        tracker.update(dict(Lloudness=Lloudness))
        Lae += Lloudness

        if self.wavebased and x_denoise is not None:
            thres = (
                int(hz2ind(4000, self.n_fft))
                if self.wavebased
                else mel_scale(self.spec_chans, 4000, pt=False).astype(np.int32)
            )
            explosive = (
                (
                    torch.mean(
                        (spec * freq_linear_reweighting)[..., thres:], dim=-1
                    )
                    > torch.mean(
                        (spec * freq_linear_reweighting)[..., :thres], dim=-1
                    )
                )
                .to(torch.float32)
                .unsqueeze(-1)
            )
            rec_denoise = self.decoder.forward(
                components,
                enable_hamon_excitation=True,
                enable_noise_excitation=True,
                enable_bgnoise=False,
                onstage=on_stage,
            )
            Lae_denoise = 20 * self.lae(
                rec_denoise * freq_linear_reweighting * explosive,
                x_denoise * freq_linear_reweighting * explosive,
            )
            tracker.update(dict(Lae_denoise=Lae_denoise))
            Lae += Lae_denoise
        
        freq_limit = self.encoder.formant_freq_limits_abs.squeeze()
        freq_limit = (
            hz2ind(freq_limit, self.n_fft).long()
            if self.wavebased
            else mel_scale(self.spec_chans, freq_limit).long()
        )

        Lamp = 30 * torch.mean(
            F.relu(
                -components["amplitude_formants_hamon"][:, 0:3]
                + components["amplitude_formants_hamon"][:, 1:4]
            )
            * (
                components["amplitudes"][:, 0:1]
                > components["amplitudes"][:, 1:2]
            ).float()
        )
        tracker.update(dict(Lamp=Lamp))
        Lae += Lamp
        tracker.update(dict(Lae=Lae))
        
        thres = (
            int(hz2ind(4000, self.n_fft))
            if self.wavebased
            else mel_scale(self.spec_chans, 4000, pt=False).astype(np.int32)
        )
        explosive = (
            torch.sign(
                torch.mean(
                    (spec * freq_linear_reweighting)[..., thres:], dim=-1
                )
                - torch.mean(
                    (spec * freq_linear_reweighting)[..., :thres], dim=-1
                ) ) * 0.5 + 0.5
        )
        Lexp = (
            torch.mean(
                (
                    components["amplitudes"][:, 0:1]
                    - components["amplitudes"][:, 1:2]
                )
                * explosive
            ) * 100
        )
        tracker.update(dict(Lexp=Lexp))
        Lae += Lexp
        
        if hamonic_bias:
            hamonic_loss = 1000 * torch.mean(
                (1 - components["amplitudes"][:, 0]) * on_stage
            )
            Lae += hamonic_loss
        if pitch_aug:
            pitch_shift = 2 ** (
                -1.5 + 3
                * torch.rand([components["f0_hz"].shape[0]]).to(torch.float32)
            ).reshape(
                [components["f0_hz"].shape[0], 1, 1]
            )
            components["f0_hz"] = (components["f0_hz"] * pitch_shift).clamp(
                min=88, max=300
            )
            rec_shift = self.decoder.forward(components, onstage=on_stage)
            components_enc = self.encoder(
                rec_shift,
                duomask=duomask,
                x_denoise=x_denoise,
                noise_level=None,
                x_amp=x_amp,
                gender=gender,
            )
            Lf0 = torch.mean(
                (components_enc["f0_hz"] / 200 - components["f0_hz"] / 200) ** 2
            )
            rec_cycle = self.decoder.forward(components_enc, onstage=on_stage)
            Lae += self.lae(
                rec_shift * freq_linear_reweighting,
                rec_cycle * freq_linear_reweighting,
                tracker=tracker,
            )
        else:
            Lf0 = torch.tensor([0.0])
        tracker.update(dict(Lf0=Lf0))
        
        spec = spec.squeeze(dim=1).permute(0, 2, 1)  # B * f * T
        if self.wavebased:
            if self.power_synth:
                hamonic_components_diff = (
                    compdiffd2(components["freq_formants_hamon_hz"] * 1.5)
                    + compdiffd2(components["f0_hz"] * 2)
                    + compdiff(
                        components["bandwidth_formants_noise_hz"][
                            :, components["freq_formants_hamon_hz"].shape[1] :
                        ]
                        / 5
                    )
                    + compdiff(
                        components["freq_formants_noise_hz"][
                            :, components["freq_formants_hamon_hz"].shape[1] :
                        ]
                        / 5
                    )
                    + compdiff(components["amplitudes"]) * 750.0
                    + compdiffd2(components["amplitude_formants_hamon"])
                    * 1500.0
                    + compdiffd2(components["amplitude_formants_noise"])
                    * 1500.0
                )
            else:
                hamonic_components_diff = (
                    compdiffd2(components["freq_formants_hamon_hz"] * 1.5)
                    + compdiffd2(components["f0_hz"] * 2)
                    + compdiff(
                        components["bandwidth_formants_noise_hz"][
                            :, components["freq_formants_hamon_hz"].shape[1] :
                        ]
                        / 5
                    )
                    + compdiff(
                        components["freq_formants_noise_hz"][
                            :, components["freq_formants_hamon_hz"].shape[1] :
                        ]
                        / 5
                    )
                    + compdiff(components["amplitudes"]) * 750.0
                    + compdiffd2(components["amplitude_formants_hamon"])
                    * 1500.0
                    + compdiffd2(components["amplitude_formants_noise"])
                    * 1500.0
                )
        else:
            hamonic_components_diff = (
                compdiff(
                    components["freq_formants_hamon_hz"] * (1 - on_stage_wider)
                )
                + compdiff(components["f0_hz"] * (1 - on_stage_wider))
                + compdiff(components["amplitude_formants_hamon"]) * 750.0
                + compdiff(components["amplitude_formants_noise"]) * 750.0
            )
        Ldiff = torch.mean(hamonic_components_diff) / 2000.0
        tracker.update(dict(Ldiff=Ldiff))
        Lae += Ldiff
        Lfreqorder = torch.mean(
            F.relu(
                components["freq_formants_hamon_hz"][:, :-1]
                - components["freq_formants_hamon_hz"][:, 1:]
            )
        )
        if formant_label is not None:
            formant_label = formant_label[:, 0].permute(0, 2, 1)
            
            Lformant = torch.mean(
                (
                    (
                        components["freq_formants_hamon_hz"][:, :-2]
                        - formant_label[:, :-2]
                    )
                    * on_stage.expand_as(formant_label[:, :-2])
                )
                ** 2
            )
            Lformant += (
                torch.mean(
                    (
                        (
                            components["freq_formants_hamon_hz"][:, 0:1]
                            - formant_label[:, 0:1]
                        )
                        * on_stage.expand_as(formant_label[:, 0:1])
                    ) ** 2 ) * 6
            )
            Lformant += (
                torch.mean(
                    (
                        (
                            components["freq_formants_hamon_hz"][:, 1:2]
                            - formant_label[:, 1:2]
                        )
                        * on_stage.expand_as(formant_label[:, 1:2])
                    )
                    ** 2
                )
                * 3
            )
            Lformant += (
                torch.mean(
                    (
                        (
                            components["freq_formants_hamon_hz"][:, 2:3]
                            - formant_label[:, 2:3]
                        )
                        * on_stage.expand_as(formant_label[:, 1:2])
                    )
                    ** 2
                )
                * 1.5
            )

            weight_decay_formant = piecewise_linear(
                epoch_current, start_decay=20, end_decay=40
            )
            Lformant *= weight_decay_formant * 0.000003
            tracker.update(dict(Lformant=Lformant))
            tracker.update(
                dict(
                    weight_decay_formant=torch.FloatTensor(
                        [weight_decay_formant]
                    )
                )
            )
        else:
            Lformant = torch.tensor([0.0])
        if pitch_label is not None:
            Lpitch = torch.mean(
                (
                    (components["f0_hz"] - pitch_label)
                    * on_stage.expand_as(pitch_label)
                )
                ** 2
            )
            weight_decay_pitch = piecewise_linear(
                epoch_current, start_decay=20, end_decay=40
            )
            Lpitch *= weight_decay_pitch * 0.0004
            tracker.update(dict(Lpitch=Lpitch))
        else:
            Lpitch = torch.tensor([0.0])
        return Lae + Lf0 + Lfreqorder + Lformant + Lloudness + Lpitch, tracker

       
    def forward(
        self,
        spec,
        ecog,
        on_stage,
        on_stage_wider,
        ae=True,
        tracker=None,
        encoder_guide=True,
        x_mel=None,
        x_denoise=None,
        pitch_aug=False,
        duomask=False,
        x_amp=None,
        hamonic_bias=False,
        x_amp_from_denoise=False,
        gender="Female",
        voice=None,
        unvoice=None,
        semivoice=None,
        plosive=None,
        fricative=None,
        formant_label=None,
        pitch_label=None,
        epoch_current=0,
        n_iter=0,
        save_path="",
    ):
        if not self.Visualize:
            if ae: # if auto encoding: do speech to speech auto-encoding (step1)
                self.encoder.requires_grad_(True)
                components = self.encoder(
                    spec,
                    x_denoise=x_denoise,
                    duomask=duomask,
                    noise_level=None,
                    x_amp=x_amp,
                    gender=gender,
                )
                rec = self.decoder.forward(
                    components, on_stage, n_iter=n_iter, save_path=save_path
                )
                Loss, tracker = self.run_a2a_loss(rec,
                                            spec,x_amp_from_denoise,x_denoise,x_mel,x_amp,
                                            components,
                                            tracker,
                                            voice,unvoice,semivoice,plosive,fricative,
                                            duomask,hamonic_bias,pitch_aug,epoch_current,
                                            gender,pitch_label,formant_label,
                                            on_stage_wider,
                                            on_stage)
                return Loss, tracker
            else: # if ECoG to speech decoding: (step2)
                components_guide = self.encode(
                    spec,
                    x_denoise=x_denoise,
                    duomask=duomask,
                    noise_level=None,
                    x_amp=x_amp,
                    gender=gender,
                )
                self.encoder.requires_grad_(False)

                rec, components_ecog = self.generate_fromecog(
                    ecog,
                    return_components=True,
                    gender=gender,
                    onstage=on_stage,
                )
                if self.rdropout != 0:
                    rec1, components_ecog1 = self.generate_fromecog(
                        ecog,
                        return_components=True,
                        gender=gender,
                        onstage=on_stage,
                    )

                betas = {
                    "loudness": 0.01,
                    "freq_formants_hamon": 0.0025,
                    "formants_ratio": 0.0025,
                    "f0_hz": 0.0,
                    "amplitudes": 0.0,
                    "amplitude_formants_hamon": 0.0,
                    "amplitude_formants_noise": 0.0,
                    "freq_formants_noise": 0.05,
                    "bandwidth_formants_noise_hz": 0.01,
                }
                alpha = {
                    "loudness": 1.0,
                    "freq_formants_hamon": 4.0,
                    "formants_ratio": 4.0,
                    "f0_hz": 1.0,
                    "amplitudes": 1.0,
                    "amplitude_formants_hamon": 1.0,
                    "amplitude_formants_noise": 1.0,
                    "freq_formants_noise": 1.0,
                    "bandwidth_formants_noise_hz": 1.0,
                }

                Loss, tracker = self.run_components_loss(
                    rec,
                    spec,
                    tracker,
                    encoder_guide,
                    components_ecog,
                    components_guide,
                    alpha,
                    betas,
                    on_stage_wider,
                    on_stage,
                )

                if self.rdropout != 0:
                    MSELoss, _ = self.run_components_loss(
                        rec1,
                        rec,
                        tracker,
                        encoder_guide,
                        components_ecog,
                        components_ecog1,
                        alpha,
                        betas,
                        on_stage_wider,
                        on_stage,
                    )
                    Loss1, _ = self.run_components_loss(
                        rec1,
                        spec,
                        tracker,
                        encoder_guide,
                        components_ecog1,
                        components_guide,
                        alpha,
                        betas,
                        on_stage_wider,
                        on_stage,
                    )
                    Loss = 0.5 * (Loss + Loss1) + self.rdropout * MSELoss

                return Loss, tracker

    def lerp(self, other, betta, w_classifier=False):
        if hasattr(other, "module"):
            other = other.module
        with torch.no_grad():
            params = (
                list(self.decoder.parameters())
                + list(self.encoder.parameters())
                + (
                    list(self.ecog_decoder.parameters())
                    if self.with_ecog else []
                )
                + (list(self.decoder_mel.parameters()) if self.do_mel_guide else [])
                + (list(self.encoder2.parameters()) if self.with_encoder2 else [])
            )
            other_param = (
                list(other.decoder.parameters())
                + list(other.encoder.parameters())
                + (
                    list(other.ecog_decoder.parameters())
                    if self.with_ecog else []
                )
                + (list(other.decoder_mel.parameters()) if self.do_mel_guide else [])
                + (list(other.encoder2.parameters()) if self.with_encoder2 else [])
            )
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)