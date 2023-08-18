from networks import *
import numpy as np
from torch.nn import functional as F
from losses import CosineLoss
from metrics.torch_stoi import NegSTOILoss


def compdiff(comp):
    return ((comp[:, :, 1:] - comp[:, :, :-1]).abs()).mean()


def compdiffd2(comp):
    diff = comp[:, :, 1:] - comp[:, :, :-1]
    return ((diff[:, :, 1:] - diff[:, :, :-1]).abs()).mean()


cumsum = torch.cumsum


def diff_dim(data, axis=1):
    if axis == 1:
        data = F.pad(data, (0, 0, 1, 0))
        return data[:, 1:] - data[:, :-1]
    elif axis == 2:
        data = F.pad(data, (1, 0, 0, 0))
        return data[:, :, 1:] - data[:, :, :-1]


# cumsum(torch.tensor(result_dict['org']), axis=1)
# diff_dim(torch.tensor(result_dict['org']), axis=1)
# diff_dim(torch.tensor(result_dict['org']), axis=2)


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels
    )
    return bin_labels, bin_label_weights


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
    # frequencies = frequencies.float()
    notes = 12.0 * (logb(frequencies, 2.0) - logb(torch.tensor([440.0]), 2.0)) + 69.0
    # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
    notes = torch.where(torch.less_equal(frequencies, 0.0), 0.0, notes.double())
    return notes.float()


# def minmaxscale(data):
#   minv = data.min()
#   maxv = data.max()
#   return (data-minv)/(maxv - minv)
def piecewise_linear(epoch, start_decay=20, end_decay=40):
    if epoch < start_decay:
        return 1
    elif start_decay <= epoch < end_decay:
        return 1 / (start_decay - end_decay) * epoch + 2
    else:
        return 0


def minmaxscale(data, quantile=0.9):
    # for harmonic scale
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


class GHMC(nn.Module):
    def __init__(self, bins=30, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        label_weight [batch_num, class_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(
                target, label_weight, pred.size(-1)
            )
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, weights, reduction="sum")
            / tot
        )
        return loss * self.loss_weight


class GHMR(nn.Module):
    def __init__(self, mu=0.02, bins=30, momentum=0, loss_weight=1.0):
        super(GHMR, self).__init__()
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
        self.loss_func = NegSTOILoss(
            sample_rate=16000, extended=extended, plus=plus, FFT_size=FFT_size
        )

    def forward(self, rec_amp, spec_amp, on_stage, suffix="stoi", tracker=None):
        stoi_loss = self.loss_func(rec_amp, spec_amp, on_stage).mean()
        tracker.update(dict({"Lae_" + suffix: stoi_loss}))
        return stoi_loss


class LAE(nn.Module):
    def __init__(
        self,
        mu=0.02,
        bins=30,
        momentum=0.75,
        loss_weight=1.0,
        db=True,
        amp=True,
        noise_db=-50,
        max_db=22.5,
        cumsum=False,
        delta_freq=False,
        delta_time=False,
    ):
        super(LAE, self).__init__()
        self.db = db
        self.amp = amp
        self.noise_db = noise_db
        self.max_db = max_db
        self.delta_time = delta_time
        self.delta_freq = delta_freq
        self.cumsum = cumsum
        if db:
            self.ghm_db = GHMR(mu, bins, momentum, loss_weight)
        if amp:
            self.ghm_amp = GHMR(mu, bins, momentum, loss_weight)

    def forward(self, rec, spec, tracker=None, reweight=1):
        if self.db:
            loss_db = self.ghm_db(rec, spec, torch.ones(spec.shape), reweight=reweight)
            if tracker is not None:
                tracker.update(dict(Lae_db=loss_db))
        else:
            loss_db = torch.tensor(0.0)
        spec_amp = amplitude(spec, noise_db=self.noise_db, max_db=self.max_db)
        rec_amp = amplitude(rec, noise_db=self.noise_db, max_db=self.max_db)
        if self.amp:
            loss_a = self.ghm_amp(
                rec_amp, spec_amp, torch.ones(spec_amp.shape), reweight=reweight
            )
            if tracker is not None:
                tracker.update(dict(Lae_a=loss_a))
        else:
            loss_a = torch.tensor(0.0)
        if self.delta_time:
            loss_delta_time = self.ghm_amp(
                diff_dim(rec_amp, axis=2),
                diff_dim(spec_amp, axis=2),
                torch.ones(spec_amp.shape),
                reweight=reweight,
            )
            if tracker is not None:
                tracker.update(dict(Lae_delta_time=loss_delta_time))
        else:
            loss_delta_time = torch.tensor(0.0)
        if self.delta_freq:
            loss_delta_freq = self.ghm_amp(
                diff_dim(rec_amp, axis=1),
                diff_dim(spec_amp, axis=1),
                torch.ones(spec_amp.shape),
                reweight=reweight,
            )
            if tracker is not None:
                tracker.update(dict(Lae_delta_time=loss_delta_freq))
        else:
            loss_delta_freq = torch.tensor(0.0)
        if self.cumsum:
            loss_cumsum = self.ghm_amp(
                cumsum(rec_amp, axis=1),
                cumsum(spec_amp, axis=1),
                torch.ones(spec_amp.shape),
                reweight=reweight,
            )
            if tracker is not None:
                tracker.update(dict(Lae_delta_time=loss_cumsum))
        else:
            loss_cumsum = torch.tensor(0.0)
        return loss_db + loss_a + loss_delta_time + loss_delta_freq + loss_cumsum


def MTF_pytorch(S):
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
        ecog_encoder_name="",
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
        consistency_loss=False,
        delta_time=False,
        delta_freq=False,
        cumsum=False,
        learned_mask=False,
        n_filter_samples=40,
        dynamic_filter_shape=False,
        learnedbandwidth=False,
        auto_regressive=False,
        patient="NY742",
        batch_size=8,
        rdropout=0,
        tmpsavepath="",
        return_filtershape=False,
        spec_fr=125,
        gender_patient="Female",
        reverse_order=True,
        larger_capacity=False,
        unified=False,
        use_stoi=False,quantfilename=None,
    ):
        super(Model, self).__init__()
        self.component_regression = component_regression
        self.tmpsavepath = tmpsavepath
        print("component_regression", component_regression)
        self.use_stoi = use_stoi
        self.amp_minmax = amp_minmax
        self.consistency_loss = consistency_loss
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
        self.ecog_encoder_name = ecog_encoder_name
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
        self.auto_regressive = auto_regressive
        self.patient = patient
        self.rdropout = rdropout
        self.return_cc = False
        self.cc_method = "None"
        self.noise_from_data = noise_from_data
        self.auto_regressive = False
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
            # dynamic_bgnoise = dynamic_bgnoise,  dealed by noise_from_data
            dynamic_filter_shape=dynamic_filter_shape,
            learnedbandwidth=learnedbandwidth,
            return_filtershape=return_filtershape,
            spec_fr=spec_fr,
            reverse_order=reverse_order,quantfilename=quantfilename,
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
        # print ('*'*100,'within model class n fft',n_fft)
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
        # if Visualize and A2A:
        if A2A:
            self.encoder2 = ENCODERS["EncoderFormantVis2D"](
                n_mels=spec_chans,
                n_formants=n_formants,
                n_formants_noise=n_formants_noise,
                wavebased=wavebased,
                hop_length=128,
                n_fft=n_fft,
                noise_db=noise_db,
                max_db=max_db,
                broud=False,
                power_synth=power_synth,
            )
        if with_ecog and not A2A:
            self.ecog_encoder = ECOG_ENCODER[ecog_encoder_name](
                    n_mels=spec_chans,
                    n_formants=n_formants_ecog,
                    network_db=network_db,
                    causal=causal,
                    anticausal=anticausal,
                    pre_articulate=pre_articulate,
                )
        self.ghm_loss = ghm_loss
        self.lae1 = LAE(noise_db=self.noise_db, max_db=self.max_db)
        self.lae2 = LAE(amp=False)
        self.lae3 = LAE(amp=False)
        self.lae4 = LAE(amp=False)
        self.lae5 = LAE(amp=False)
        self.lae6 = LAE(amp=False)
        self.lae7 = LAE(amp=False)
        self.lae8 = LAE(amp=False)
        self.stoi_loss_female = STOI_Loss(extended=False, plus=True, FFT_size=256)
        self.stoi_loss_male = STOI_Loss(extended=False, plus=True, FFT_size=512)

    def noise_dist_init(self, dist):
        with torch.no_grad():
            self.decoder.noise_dist = dist.reshape([1, 1, 1, dist.shape[0]])

    def generate_fromecog(
        self,
        ecog=None,
        return_components=False,
        gt_comp=None,
        gt_spec=None,
        return_also_encodings=False,
        gender=None,
        onstage=None,
    ):
        if self.auto_regressive:
            # print ('use auto regressive')
            # print ('gt_comp first come in function',gt_comp['f0_hz'].shape)
            components = self.ecog_encoder(
                ecog,
                gt_comp,
                dec_return_also_encodings=return_also_encodings,
                gender=gender,
            )
            if return_also_encodings:
                encodings, target_comp, components = components
            else:
                target_comp, components = components
            target_spec = gt_spec[:, :, 1:]
        else:
            # print ('not use auto regressive')
            components = self.ecog_encoder(ecog)
        rec = self.decoder.forward(components, onstage)
        if return_components:
            if self.auto_regressive:
                return rec, components, target_spec, target_comp
            else:
                return rec, components
        else:
            if self.auto_regressive:
                return rec, target_spec
            else:
                return rec

    def generate_fromspec(
        self,
        spec,
        return_components=False,
        x_denoise=None,
        duomask=False,
        gender="Female",
        onstage=None,
    ):
        components = self.encoder(
            spec, x_denoise=x_denoise, duomask=duomask, gender=gender
        )
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
    ):

        if amp:
            spec_amp = amplitude(spec, noise_db=self.noise_db, max_db=self.max_db)
            rec_amp = amplitude(rec, noise_db=self.noise_db, max_db=self.max_db)
            spec_amp_ = spec_amp
            rec_amp_ = rec_amp
            if GHM:
                Lae_a = self.ghm_loss(
                    rec_amp_, spec_amp_, torch.ones(spec_amp_)
                )  # *150
                Lae_a_l2 = torch.tensor([0.0])
            else:
                Lae_a = (spec_amp_ - rec_amp_).abs().mean()  # *150
                Lae_a_l2 = torch.sqrt((spec_amp_ - rec_amp_) ** 2 + 1e-6).mean()  # *150
        else:
            Lae_a = torch.tensor(0.0)
            Lae_a_l2 = torch.tensor(0.0)
        if tracker is not None:
            tracker.update(
                dict({"Lae_a" + suffix: Lae_a, "Lae_a_l2" + suffix: Lae_a_l2})
            )
        if db:
            if GHM:
                Lae_db = self.ghm_loss(rec, spec, torch.ones(spec))  # *150
                Lae_db_l2 = torch.tensor([0.0])
            else:  # we use this branch!!!
                # print ('loudness loss!!')
                Lae_db = (spec - rec).abs().mean()
                Lae_db_l2 = torch.sqrt((spec - rec) ** 2 + 1e-6).mean()
        else:
            Lae_db = torch.tensor(0.0)
            Lae_db_l2 = torch.tensor(0.0)
        if MTF:
            spec_amp = amplitude(spec, noise_db=self.noise_db, max_db=self.max_db)
            rec_amp = amplitude(rec, noise_db=self.noise_db, max_db=self.max_db)
            # print ('loudness MTF loss!!')
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
        # return loss_db+loss_a+loss_delta_time+loss_delta_freq+loss_cumsum

        if tracker is not None:
            tracker.update(
                dict({"Lae_db" + suffix: Lae_db, "Lae_db_l2" + suffix: Lae_db_l2})
            )
        # return (Lae_a + Lae_a_l2)/2. + (Lae_db+Lae_db_l2)/2.
        return (
            Lae_a
            + Lae_db / 2.0
            + Lae_mtf
            + loss_delta_time
            + loss_delta_freq
            + loss_cumsum
        )

    def flooding(self, loss, beta):
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
        if self.spec_sup and not self.A2A:
            if False:  # self.ghm_loss:
                Lrec = 0.3 * self.lae1(rec, spec, tracker=tracker)
            else:
                Lrec = 80 * self.lae(
                    rec, spec, tracker=tracker, amp=False, suffix="1", MTF=False
                )  # torch.mean((rec - spec)**2)
        else:
            Lrec = torch.tensor([0.0])  #

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
        Lcomp = 0
        if encoder_guide:
            consonant_weight = (
                1 
            )
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
            if self.ld_loss_weight:
                loudness_db_norm_weight = loudness_db_norm
            else:
                loudness_db_norm_weight = 1
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
                consonant_weight = 1  # 100*(torch.sign(components_guide['amplitudes'][:,1:]-0.5)*0.5+0.5)
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
                    if False:  # self.ghm_loss:
                        diff = self.lae2(loudness_db_norm, loudness_db_norm_ecog)
                    else:
                        diff = (
                            alpha["loudness"]
                            * 150
                            * torch.mean(
                                (loudness_db_norm - loudness_db_norm_ecog) ** 2
                            )
                        )  # + torch.mean((components_guide[key] - components_ecog[key])**2 * on_stage * consonant_weight)
                    # diff = self.flooding(diff * 0.2 ,alpha['loudness']*betas['loudness']) #0519 modify loudness loss!
                    # 0918 modify loudness loss, remove flooding to encourage a more rigourous learning
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
                            components_guide[key] / 200 * 5
                            - components_ecog[key] / 200 * 5
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
                                    components_guide["f0_hz"] / 200 * 5
                                    - components_ecog["f0_hz"] / 200 * 5
                                )
                                ** 2
                                * on_stage_wider
                                * loudness_db_norm
                            )
                        }
                    )

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
                            * 540
                            * self.lae3(tmp_target, tmp_ecog, reweight=weight)
                        )
                    else:
                        diff = (
                            alpha["amplitudes"]
                            * 180
                            * torch.mean((tmp_target - tmp_ecog) ** 2 * weight)
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
                                )
                                ** 2
                                * weight
                            )
                        }
                    )

                if key in ["amplitude_formants_hamon"]:
                    weight = (
                        alpha_formant_weight
                        * on_stage_wider
                        * consonant_weight
                        * loudness_db_norm_weight
                    )
                    if False:  # self.ghm_loss:
                        diff = 40 * self.lae4(
                            components_guide[key][:, : self.n_formants_ecog],
                            components_ecog[key],
                            reweight=weight,
                        )
                    else:
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
                            # just to db norm
                        elif self.amp_energy == 3:
                            # half linear, half db
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
                    if False:  # self.ghm_loss:
                        diff = self.lae6(
                            components_guide[key], components_ecog[key], reweight=weight
                        )
                    else:
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
                            # just to db norm
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

                    if self.consistency_loss:
                        diff = (
                            kde_loss.call(
                                components_guide["amplitude_formants_noise"],
                                components_guide["freq_formants_noise_hz"],
                                components_ecog["amplitude_formants_noise"],
                                components_ecog["freq_formants_noise_hz"],
                                freq_single_formant_weight,
                                amplitude_formants_noise_db_norm_weight,
                                weight,
                            )
                            * 5000
                        )
                    else:
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
                                    / 2000
                                    * 5
                                    - components_ecog["freq_formants_noise_hz"][
                                        :, -self.n_formants_noise :
                                    ]
                                    / 2000
                                    * 5
                                )
                                ** 2
                                * weight
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
                    if False:  # self.ghm_loss:
                        diff = 3 * self.lae8(
                            components_guide[key][:, -self.n_formants_noise :]
                            / 2000
                            * 5,
                            components_ecog[key][:, -self.n_formants_noise :]
                            / 2000
                            * 5,
                            reweight=weight,
                        )
                    else:
                        # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
                        diff = (
                            alpha["bandwidth_formants_noise_hz"]
                            * 3
                            * torch.mean(
                                (
                                    components_guide[key][:, -self.n_formants_noise :]
                                    / 2000
                                    * 5
                                    - components_ecog[key][:, -self.n_formants_noise :]
                                    / 2000
                                    * 5
                                )
                                ** 2
                                * weight
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
                                    / 2000
                                    * 5
                                    - components_ecog[key][:, -self.n_formants_noise :]
                                    / 2000
                                    * 5
                                )
                                ** 2
                                * weight
                            )
                        }
                    )

                tracker.update({key: diff})
                Lcomp += diff

            if "formant_ratio" in components_ecog.keys():
                print("loss add formant ratio!")
                weight = on_stage_wider * loudness_db_norm_weight
                formants_freqs_ratio_guide = (
                    components_guide["freq_formants_hamon_hz"][
                        :, : self.n_formants_ecog
                    ]
                    - self.formant_freq_limits_abs_low[:, : self.n_formants_ecog]
                ) / (
                    (
                        self.formant_freq_limits_abs[:, : self.n_formants_ecog]
                        - self.formant_freq_limits_abs_low[:, : self.n_formants_ecog]
                    )
                )
                formants_freqs_ratio_guide = torch.log(
                    formants_freqs_ratio_guide[:, 1 : self.n_formants_ecog]
                    / formants_freqs_ratio_guide[:, :1].expand(
                        formants_freqs_ratio_guide[:, 1 : self.n_formants_ecog].size()
                    )
                )
                formants_freqs_ratio_ecog = components_ecog["formant_ratio"][:, 1:]

                tmp_diff = (
                    formants_freqs_ratio_guide - formants_freqs_ratio_ecog
                ) ** 2 * freq_single_formant_weight
                diff = (
                    alpha["formants_ratio"]
                    * 300
                    * torch.mean(
                        tmp_diff * amplitude_formants_hamon_db_norm_weight * weight
                    )
                )
                diff = self.flooding(
                    diff, alpha["formants_ratio"] * betas["formants_ratio"]
                )
                tracker.update({"formant_ratio_" + str(self.n_formants_ecog): tmp_diff})
            if "formant_ratio2" in components_ecog.keys():
                weight = on_stage_wider * loudness_db_norm_weight

                # get weight factor
                formants_freqs_ratio_guide = (
                    components_guide["freq_formants_hamon_hz"][
                        :, : self.n_formants_ecog
                    ]
                    - self.formant_freq_limits_abs_low[:, : self.n_formants_ecog]
                ) / (
                    (
                        self.formant_freq_limits_abs[:, : self.n_formants_ecog]
                        - self.formant_freq_limits_abs_low[:, : self.n_formants_ecog]
                    )
                )
                formants_freqs_ratio_guide = formants_freqs_ratio_guide[:, :1].expand(
                    formants_freqs_ratio_guide[:, 1 : self.n_formants_ecog].size()
                ) / (formants_freqs_ratio_guide[:, 1 : self.n_formants_ecog] + 0.000001)
                formants_freqs_ratio_ecog = components_ecog["formant_ratio2"][:, 1:]

                tmp_diff = (
                    formants_freqs_ratio_guide - formants_freqs_ratio_ecog
                ) ** 2 * freq_single_formant_weight
                diff = (
                    alpha["formants_ratio"]
                    * 300
                    * torch.mean(
                        tmp_diff * amplitude_formants_hamon_db_norm_weight * weight
                    )
                )
                diff = self.flooding(
                    diff, alpha["formants_ratio"] * betas["formants_ratio"]
                )
                tracker.update(
                    {"formant_ratio2_" + str(self.n_formants_ecog): tmp_diff}
                )

        if self.component_regression:
            print("component regression!")
            Loss = Lcomp
        else:
            Loss = Lrec + Lcomp

        if self.distill:
            weight = (
                alpha_formant_weight
                * on_stage_wider
                * consonant_weight
                * loudness_db_norm_weight
            )
            tmp_diff = (components_guide["x_common"] - components_ecog["x_common"]) ** 2
            diff_x_common = 400 * torch.mean(tmp_diff * weight)
            tracker.update({"x_common": torch.mean(tmp_diff * weight)})

            weight = (
                alpha_formant_weight
                * on_stage_wider
                * consonant_weight
                * loudness_db_norm_weight
            )
            tmp_diff = (
                components_guide["x_formants"] - components_ecog["x_formants"]
            ) ** 2
            diff_x_formants = 400 * torch.mean(tmp_diff * weight)
            tracker.update({"x_formants": torch.mean(tmp_diff * weight)})
            Ldistill = diff_x_common + diff_x_formants
            Loss += Ldistill

        hamonic_components_diff = (
            compdiffd2(components_ecog["freq_formants_hamon_hz"] * 1.5)
            + compdiffd2(components_ecog["f0_hz"] * 2)
            + compdiff(
                components_ecog["bandwidth_formants_noise_hz"][
                    :, components_ecog["freq_formants_hamon_hz"].shape[1] :
                ]
                / 5
            )
            + compdiff(
                components_ecog["freq_formants_noise_hz"][
                    :, components_ecog["freq_formants_hamon_hz"].shape[1] :
                ]
                / 5
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

    def forward(
        self,
        spec,
        ecog,
        on_stage,
        on_stage_wider,
        gt_comp=None,
        ae=True,
        tracker=None,
        encoder_guide=True,
        x_mel=None,
        x_denoise=None,
        pitch_aug=False,
        duomask=False,
        debug=False,
        x_amp=None,
        hamonic_bias=False,
        x_amp_from_denoise=False,
        gender="Female",
        voice=None,
        unvoice=None,
        semivoice=None,
        plosive=None,
        fricative=None,
        t1=None,
        t2=None,
        formant_label=None,
        pitch_label=None,
        intensity_label=None,
        epoch_record=0,
        epoch_current=0,
        n_iter=0,
        save_path="",
    ):
        # print ('self.Visualize',self.Visualize)

        if not self.Visualize:
            if ae:  # audio to audio
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
                    if self.use_stoi:
                        if spec_amp.shape[-2] == 256:
                            stoi_loss = (
                                self.stoi_loss_female(
                                    rec_amp,
                                    spec_amp,
                                    on_stage,
                                    suffix="stoi",
                                    tracker=tracker,
                                )
                                * 10
                            )
                        else:
                            stoi_loss = (
                                self.stoi_loss_male(
                                    rec_amp,
                                    spec_amp,
                                    on_stage,
                                    suffix="stoi",
                                    tracker=tracker,
                                )
                                * 10
                            )
                        Lae += stoi_loss

                    freq_cord2 = torch.arange(128 + 1).reshape([1, 1, 1, 128 + 1]) / (
                        1.0 * 128
                    )
                    freq_linear_reweighting2 = (
                        (
                            inverse_mel_scale(freq_cord2[..., 1:])
                            - inverse_mel_scale(freq_cord2[..., :-1])
                        )
                        / 440
                        * 7
                    )
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
                    Lae += 8 * self.lae(
                        rec_mel * freq_linear_reweighting2,
                        spec_mel * freq_linear_reweighting2,
                        tracker=tracker,
                        amp=False,
                        suffix="2",
                    )

                if self.do_mel_guide:
                    rec_mel = self.decoder_mel.forward(components)
                    freq_linear_reweighting_mel = (
                        (
                            inverse_mel_scale(freq_cord2[..., 1:])
                            - inverse_mel_scale(freq_cord2[..., :-1])
                        )
                        / 440
                        * 7
                    )
                    Lae_mel = 4 * self.lae(
                        rec_mel * freq_linear_reweighting_mel,
                        x_mel * freq_linear_reweighting_mel,
                        tracker=None,
                    )
                    tracker.update(dict(Lae_mel=Lae_mel))
                    Lae += Lae_mel
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
                            / 4000
                            - 4000.0 / 4000
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
                            / 8000
                            - 8000.0 / 8000
                        )
                        ** 2
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
                            ]
                            / 1500.0
                            - 1500.0 / 1500
                        )
                        ** 2
                    )
                    # Lfricativeband = 4*(F.relu(components['bandwidth_formants_noise_hz'][:,-1:][fricative==1]/4500-4500./4500)**2)
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

                if x_amp_from_denoise:
                    if self.wavebased:
                        if self.power_synth:
                            Lloudness = (
                                10**6
                                * (components["loudness"] * (1 - on_stage_wider)).mean()
                            )
                        else:
                            Lloudness = (
                                10**6
                                * (components["loudness"] * (1 - on_stage_wider)).mean()
                            )
                        tracker.update(dict(Lloudness=Lloudness))
                        Lae += Lloudness
                else:
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
                # import pdb;pdb.set_trace()
                # if components['freq_formants_hamon'].shape[1] > 2:
                freq_limit = self.encoder.formant_freq_limits_abs.squeeze()

                freq_limit = (
                    hz2ind(freq_limit, self.n_fft).long()
                    if self.wavebased
                    else mel_scale(self.spec_chans, freq_limit).long()
                )
                if debug:
                    import pdb

                    pdb.set_trace()

                # if True:
                if not self.wavebased:
                    n_formant_noise = (
                        components["freq_formants_noise"].shape[1]
                        - components["freq_formants_hamon"].shape[1]
                    )
                    for formant in range(
                        components["freq_formants_hamon"].shape[1] - 1, 1, -1
                    ):
                        components_copy = {i: j.clone() for i, j in components.items()}
                        components_copy["freq_formants_hamon"] = components_copy[
                            "freq_formants_hamon"
                        ][:, :formant]
                        components_copy["freq_formants_hamon_hz"] = components_copy[
                            "freq_formants_hamon_hz"
                        ][:, :formant]
                        components_copy["bandwidth_formants_hamon"] = components_copy[
                            "bandwidth_formants_hamon"
                        ][:, :formant]
                        components_copy[
                            "bandwidth_formants_hamon_hz"
                        ] = components_copy["bandwidth_formants_hamon_hz"][:, :formant]
                        components_copy["amplitude_formants_hamon"] = components_copy[
                            "amplitude_formants_hamon"
                        ][:, :formant]

                        if duomask:
                            # components_copy['freq_formants_noise'] = components_copy['freq_formants_noise'][:,:formant]
                            # components_copy['freq_formants_noise_hz'] = components_copy['freq_formants_noise_hz'][:,:formant]
                            # components_copy['bandwidth_formants_noise'] = components_copy['bandwidth_formants_noise'][:,:formant]
                            # components_copy['bandwidth_formants_noise_hz'] = components_copy['bandwidth_formants_noise_hz'][:,:formant]
                            # components_copy['amplitude_formants_noise'] = components_copy['amplitude_formants_noise'][:,:formant]
                            components_copy["freq_formants_noise"] = torch.cat(
                                [
                                    components_copy["freq_formants_noise"][:, :formant],
                                    components_copy["freq_formants_noise"][
                                        :, -n_formant_noise:
                                    ],
                                ],
                                dim=1,
                            )
                            components_copy["freq_formants_noise_hz"] = torch.cat(
                                [
                                    components_copy["freq_formants_noise_hz"][
                                        :, :formant
                                    ],
                                    components_copy["freq_formants_noise_hz"][
                                        :, -n_formant_noise:
                                    ],
                                ],
                                dim=1,
                            )
                            components_copy["bandwidth_formants_noise"] = torch.cat(
                                [
                                    components_copy["bandwidth_formants_noise"][
                                        :, :formant
                                    ],
                                    components_copy["bandwidth_formants_noise"][
                                        :, -n_formant_noise:
                                    ],
                                ],
                                dim=1,
                            )
                            components_copy["bandwidth_formants_noise_hz"] = torch.cat(
                                [
                                    components_copy["bandwidth_formants_noise_hz"][
                                        :, :formant
                                    ],
                                    components_copy["bandwidth_formants_noise_hz"][
                                        :, -n_formant_noise:
                                    ],
                                ],
                                dim=1,
                            )
                            components_copy["amplitude_formants_noise"] = torch.cat(
                                [
                                    components_copy["amplitude_formants_noise"][
                                        :, :formant
                                    ],
                                    components_copy["amplitude_formants_noise"][
                                        :, -n_formant_noise:
                                    ],
                                ],
                                dim=1,
                            )

                        # Lae += self.lae(rec,spec,tracker=tracker)#torch.mean((rec - spec).abs())
                        rec = self.decoder.forward(
                            components_copy,
                            enable_noise_excitation=True if self.wavebased else True,
                            onstage=on_stage,
                        )
                        Lae += 1 * self.lae(
                            (rec * freq_linear_reweighting),
                            (spec * freq_linear_reweighting),
                            tracker=tracker,
                        )  # torch.mean(((rec - spec).abs()*freq_linear_reweighting)[...,:freq_limit[formant-1]])
                        #   Lae += self.lae((rec*freq_linear_reweighting)[...,:freq_limit[formant-1]],(spec*freq_linear_reweighting)[...,:freq_limit[formant-1]],tracker=tracker)#torch.mean(((rec - spec).abs()*freq_linear_reweighting)[...,:freq_limit[formant-1]])
                        # Lamp = 1*torch.mean(F.relu(-components['amplitude_formants_hamon'][:,0:3]+components['amplitude_formants_hamon'][:,1:4])*(components['amplitudes'][:,0:1]>components['amplitudes'][:,1:2]).float())
                        # tracker.update(dict(Lamp=Lamp))
                        # Lae+=Lamp
                else:
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
                if debug:
                    import pdb

                    pdb.set_trace()

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
                        )
                    )
                    * 0.5
                    + 0.5
                )
                Lexp = (
                    torch.mean(
                        (
                            components["amplitudes"][:, 0:1]
                            - components["amplitudes"][:, 1:2]
                        )
                        * explosive
                    )
                    * 100
                )
                tracker.update(dict(Lexp=Lexp))
                Lae += Lexp

                if hamonic_bias:
                    hamonic_loss = 1000 * torch.mean(
                        (1 - components["amplitudes"][:, 0]) * on_stage
                    )
                    Lae += hamonic_loss

                # alphaloss=(F.relu(0.5-(components['amplitudes']-0.5).abs())*100).mean()
                # Lae+=alphaloss

                if pitch_aug:
                    pitch_shift = 2 ** (
                        -1.5
                        + 3
                        * torch.rand([components["f0_hz"].shape[0]]).to(torch.float32)
                    ).reshape(
                        [components["f0_hz"].shape[0], 1, 1]
                    )  # +- 1 octave
                    # pitch_shift = (2**(torch.randint(-1,2,[components['f0_hz'].shape[0]]).to(torch.float32)).reshape([components['f0_hz'].shape[0],1,1])).clamp(min=88,max=616) # +- 1 octave
                    components["f0_hz"] = (components["f0_hz"] * pitch_shift).clamp(
                        min=88, max=300
                    )
                    # components['f0'] = mel_scale(self.spec_chans,components['f0'])/self.spec_chans
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
                    )  # torch.mean((rec_shift-rec_cycle).abs()*freq_linear_reweighting)
                    # import pdb;pdb.set_trace()
                else:
                    # Lf0 = torch.mean((F.relu(160 - components['f0_hz']) + F.relu(components['f0_hz']-420))/10)
                    Lf0 = torch.tensor([0.0])
                # Lf0 = torch.tensor([0.])
                tracker.update(dict(Lf0=Lf0))

                spec = spec.squeeze(dim=1).permute(0, 2, 1)  # B * f * T
                loudness = torch.mean(spec * 0.5 + 0.5, dim=1, keepdim=True)
                # import pdb;pdb.set_trace()
                if self.wavebased:
                    #  hamonic_components_diff = compdiffd2(components['f0_hz']*2) + compdiff(components['amplitudes'])*750.# + compdiff(components['amplitude_formants_hamon'])*750.  + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50 + compdiff(components['amplitude_formants_noise'])*750.
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
                        )  #   + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50
                    else:
                        # hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*1.5) + compdiffd2(components['f0_hz']*2)  + compdiff(components['amplitudes'])*750. + compdiffd2(components['amplitude_formants_hamon'])*1500.+ compdiffd2(components['amplitude_formants_noise'])*1500.#   + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50
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
                        )  #   + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50

                    # hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*1.5) + compdiffd2(components['f0_hz']*2)   + compdiff(components['bandwidth_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components['freq_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components['amplitudes'])*750.# + compdiff(components['amplitude_formants_hamon'])*750.  + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50 + compdiff(components['amplitude_formants_noise'])*750.
                    #  hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*1.5) + compdiffd2(components['f0_hz']*8)   + compdiff(components['bandwidth_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components['freq_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components['amplitudes'])*750.# + compdiff(components['amplitude_formants_hamon'])*750.  + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50 + compdiff(components['amplitude_formants_noise'])*750.
                    # hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*2) + compdiffd2(components['f0_hz']/10) + compdiff(components['amplitude_formants_hamon'])*750. + compdiff(components['amplitude_formants_noise'])*750. + compdiffd2(components['freq_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/10) + compdiff(components['bandwidth_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/10)
                    # hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz'])+100*compdiffd2(components['f0_hz']*3) + compdiff(components['amplitude_formants_hamon'])*750. + compdiff(components['amplitude_formants_noise'])*750. #+ compdiff(components['freq_formants_noise_hz']*(1-on_stage_wider))
                else:
                    hamonic_components_diff = (
                        compdiff(
                            components["freq_formants_hamon_hz"] * (1 - on_stage_wider)
                        )
                        + compdiff(components["f0_hz"] * (1 - on_stage_wider))
                        + compdiff(components["amplitude_formants_hamon"]) * 750.0
                        + compdiff(components["amplitude_formants_noise"]) * 750.0
                    )  # + compdiff(components['freq_formants_noise_hz']*(1-on_stage_wider))
                # hamonic_components_diff = compdiff(components['freq_formants_hamon_hz'])+compdiff(components['f0_hz']) + compdiff(components['amplitude_formants_hamon']*(1-on_stage_wider))*1500. + compdiff(components['amplitude_formants_noise']*(1-on_stage_wider))*1500. + compdiff(components['freq_formants_noise_hz'])
                Ldiff = torch.mean(hamonic_components_diff) / 2000.0
                # Ldiff = torch.mean(components['freq_formants_hamon'].var()+components['freq_formants_noise'].var())*10
                tracker.update(dict(Ldiff=Ldiff))
                Lae += Ldiff
                Lfreqorder = torch.mean(
                    F.relu(
                        components["freq_formants_hamon_hz"][:, :-1]
                        - components["freq_formants_hamon_hz"][:, 1:]
                    )
                )  # + (torch.mean(F.relu(components['freq_formants_noise_hz'][:,:-1]-components['freq_formants_noise_hz'][:,1:])) if components['freq_formants_noise_hz'].shape[1]>1 else 0)

                # TODO: add formant and pitch supervision!!
                if formant_label is not None:
                    formant_label = formant_label[:, 0].permute(0, 2, 1)
                    # print ('formant_label is used!')
                    # print (components['freq_formants_hamon_hz'].shape,formant_label.shape,on_stage.shape)
                    # print (components['freq_formants_hamon_hz'] ,formant_label ,on_stage )
                    debug_save_path = (
                        self.tmpsavepath
                    )  # '/scratch/xc1490/projects/ecog/ALAE_1023/output/a2a_04121200_a2a_sub_NY710_density_LD_formantsup_1_wavebased_1_bgnoisefromdata_1_load_0_ft_1_learnfilter_0/'
                    # if epoch_record <= epoch_current:
                    """
                    try:
                    #print ([int(i.split('.')[0].split('_')[-2]) for i in os.listdir(debug_save_path) if i.endswith('savelabel.npy')])
                        #trial = int(np.max([int(i.split('.')[0].split('_')[-3]) for i in os.listdir(debug_save_path) if i.endswith(epoch_current+'_savelabel.npy')]))
                        trial = int(np.max([int(i.split('.')[0].split('_')[-2]) for i in os.listdir(debug_save_path) if i.endswith('_savelabel.npy')]))
                    except:
                        trial = 0
                    if trial <=200:
                        trial += 1
                        np.save(debug_save_path+'/formant_label_{}_savelabel.npy'.format( trial ), formant_label.detach().cpu().numpy())
                        np.save(debug_save_path+'/formant_freq_{}_savelabel.npy'.format(trial ), components['freq_formants_hamon_hz'].detach().cpu().numpy())
                        np.save(debug_save_path+'/on_stage_{}_savelabel.npy'.format(trial ), on_stage.detach().cpu().numpy())
                        np.save(debug_save_path+'/spec_{}_savelabel.npy'.format(trial ), spec.detach().cpu().numpy())
                        np.save(debug_save_path+'/rec_{}_savelabel.npy'.format(trial ), rec.detach().cpu().numpy())
                        np.save(debug_save_path+'/formantonstage1_{}_savelabel.npy'.format(trial ), ((components['freq_formants_hamon_hz'][:,:-2] - formant_label[:,:-2]) * on_stage.expand_as(formant_label[:,:-2])).detach().cpu().numpy())
                        np.save(debug_save_path+'/formantonstage2_{}_savelabel.npy'.format(trial ), ((components['freq_formants_hamon_hz'][:,0:1] - formant_label[:,0:1]) * on_stage.expand_as(formant_label[:,0:1])).detach().cpu().numpy())
                        np.save(debug_save_path+'/formantonstage3_{}_savelabel.npy'.format(trial ), ((components['freq_formants_hamon_hz'][:,0:1] - formant_label[:,0:1]) * on_stage.expand_as(formant_label[:,0:1])).detach().cpu().numpy())
                    #epoch_record += 1
                    """
                    # print (on_stage.expand_as(formant_label).shape)
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
                    # give more weight to f1 and f2
                    Lformant += (
                        torch.mean(
                            (
                                (
                                    components["freq_formants_hamon_hz"][:, 0:1]
                                    - formant_label[:, 0:1]
                                )
                                * on_stage.expand_as(formant_label[:, 0:1])
                            )
                            ** 2
                        )
                        * 6
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
                    # np.exp(0.1 * -np.arange(60))
                    # weight_decay_formant = np.exp(- 0.1 *  epoch_current)
                    # picewise linear, first 20 epochs, full formant supervision, then decay supervision to 0 at 40 epochs, then unsupervised
                    # Lformant *= 0.005
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
                    # print ('weight decay', weight_decay_formant)
                else:
                    Lformant = torch.tensor([0.0])  # 0
                    # print ('formant_label is not used!')
                if pitch_label is not None:
                    # print (pitch_label.shape, components['f0_hz'].shape,on_stage.shape)
                    pitch_label = pitch_label  # [:,0]#.permute(0,2,1)
                    debug_save_path = (
                        self.tmpsavepath
                    )  # '/scratch/xc1490/projects/ecog/ALAE_1023/output/a2a_04121200_a2a_sub_NY710_density_LD_formantsup_1_wavebased_1_bgnoisefromdata_1_load_0_ft_1_learnfilter_0/'
                    # print (on_stage.expand_as(formant_label).shape)
                    Lpitch = torch.mean(
                        (
                            (components["f0_hz"] - pitch_label)
                            * on_stage.expand_as(pitch_label)
                        )
                        ** 2
                    )
                    # give more weight to f1 and f2
                    # np.exp(0.1 * -np.arange(60))
                    # weight_decay_formant = np.exp(- 0.1 *  epoch_current)
                    # picewise linear, first 20 epochs, full formant supervision, then decay supervision to 0 at 40 epochs, then unsupervised
                    # Lformant *= 0.005
                    weight_decay_pitch = piecewise_linear(
                        epoch_current, start_decay=20, end_decay=40
                    )
                    Lpitch *= weight_decay_pitch * 0.0004
                    tracker.update(dict(Lpitch=Lpitch))
                    # tracker.update(dict(  weight_decay_formant = torch.FloatTensor([weight_decay_formant])))
                    # print ('weight decay', weight_decay_formant)
                    # print ((components['f0_hz']  - pitch_label).abs().mean(), components['f0_hz'] , pitch_label)
                else:
                    Lpitch = torch.tensor([0.0])  # 0
                    # print ('formant_label is not used!')

                return Lae + Lf0 + Lfreqorder + Lformant + Lloudness + Lpitch, tracker
                # return  Lformant #debug 0413
            else:  # ecog to audio
                components_guide = self.encode(
                    spec,
                    x_denoise=x_denoise,
                    duomask=duomask,
                    noise_level=None,
                    x_amp=x_amp,
                    gender=gender,
                )
                self.encoder.requires_grad_(False)
                if self.A2A:
                    components_ecog = self.encoder2(
                        spec2,
                        x_denoise=x_denoise,
                        duomask=duomask,
                        noise_level=None,
                        x_amp=x_amp,
                        gender=gender,
                    )
                else:
                    if (
                        self.auto_regressive
                    ):  # the target / guidance is one sample ahead of time
                        # print (' spec before function', spec.shape)
                        (
                            rec,
                            components_ecog,
                            spec,
                            components_guide,
                        ) = self.generate_fromecog(
                            ecog,
                            return_components=True,
                            gt_comp=components_guide,
                            gt_spec=spec,
                            gender=gender,
                            onstage=on_stage,
                        )
                        # print ('rec, spec after function',rec.shape,spec.shape)
                        on_stage = on_stage[:, :, 1:]
                        on_stage_wider = on_stage_wider[:, :, 1:]
                        voice = voice[:, :, 1:]
                        unvoice = unvoice[:, :, 1:]
                        semivoice = semivoice[:, :, 1:]
                        plosive = plosive[:, :, 1:]
                        fricative = fricative[:, :, 1:]
                    else:
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
                    # for both rec and components_ecog
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
                    list(self.ecog_encoder.parameters())
                    if self.with_ecog and not self.A2A
                    else []
                )
                + (list(self.decoder_mel.parameters()) if self.do_mel_guide else [])
                + (list(self.encoder2.parameters()) if self.with_encoder2 else [])
            )
            other_param = (
                list(other.decoder.parameters())
                + list(other.encoder.parameters())
                + (
                    list(other.ecog_encoder.parameters())
                    if self.with_ecog and not self.A2A
                    else []
                )
                + (list(other.decoder_mel.parameters()) if self.do_mel_guide else [])
                + (list(other.encoder2.parameters()) if self.with_encoder2 else [])
            )
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)


#
