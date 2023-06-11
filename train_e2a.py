# Copyright 2020-2023 Ran Wang, Xupeng Chen
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

LOAD = 1

import torch
from torch import optim as optim
import torch.utils.data
from tqdm import tqdm as tqdm
import numpy as np
import argparse, os, json, yaml
from networks import *
from model import Model
from dataset import *
from tracker import LossTracker
from utils.custom_adam import LREQAdam
from utils.checkpointer import Checkpointer
from utils.launcher import run
from utils.defaults import get_cfg_defaults
from utils.save import save_sample
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="ecog formant model")
parser.add_argument(
    "-c",
    "--config-file",
    default="configs/e2a_production.yaml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
parser.add_argument(
    "--DENSITY",
    type=str,
    default="LD",
    help="Data density, LD for low density, HB for hybrid density",
)
parser.add_argument("--wavebased", 
                    type=int, default=1, 
                    help="If True, use spectrogram generated from STFT of waveform")
parser.add_argument(
    "--bgnoise_fromdata",
    type=int,
    default=1,
    help="bgnoise_fromdata or not, if false the bg noise will be learned from spec",
)
parser.add_argument(
    "--ignore_loading",
    type=int,
    default=0,
    help="ignore_loading true: from scratch, false: load pretrainde model and finetune",
)
parser.add_argument(
    "--finetune", type=int, default=1, help="finetune could influence load checkpoint"
)
parser.add_argument(
    "--learnedmask",
    type=int,
    default=1,
    help="finetune could influence load checkpoint",
)
parser.add_argument(
    "--dynamicfiltershape",
    type=int,
    default=0,
    help="finetune could influence load checkpoint",
)
parser.add_argument(
    "--formant_supervision", type=int, default=0, help="formant_supervision"
)
parser.add_argument(
    "--pitch_supervision", type=int, default=0, help="pitch_supervision"
)
parser.add_argument(
    "--intensity_supervision", type=int, default=0, help="intensity_supervision"
)
parser.add_argument(
    "--n_filter_samples", type=int, default=80, help="n_filter_samples"
)
parser.add_argument(
    "--n_fft",
    type=int,
    default=512,
    help="deliberately set a wrong default to make sure feed a correct n fft ",
)
parser.add_argument(
    "--reverse_order",
    type=int,
    default=1,
    help="reverse order of learn filter shape from spec, which is actually not appropriate",
)
parser.add_argument(
    "--lar_cap", type=int, default=0, help="larger capacity for male encoder"
)
parser.add_argument(
    "--intensity_thres",
    type=float,
    default=-1,
    help="used to determine onstage, 0 means we use the default setting in Dataset.json",
)
parser.add_argument(
    "--RNN_COMPUTE_DB_LOUDNESS", type=int, default=1, help="RNN_COMPUTE_DB_LOUDNESS"
)
parser.add_argument("--BIDIRECTION", type=int, default=1, help="BIDIRECTION")
parser.add_argument(
    "--MAPPING_FROM_ECOG",
    type=str,
    default="ECoGMapping_fasttrans_3Dconv_downsample1_posemb_flatten_phoneclassifier",
    help="MAPPING_FROM_ECOG",
)
parser.add_argument(
    "--OUTPUT_DIR", type=str, default="output/resnet", help="OUTPUT_DIR"
)
parser.add_argument("--COMPONENTKEY", type=str, default="", help="COMPONENTKEY")
parser.add_argument(
    "--trainsubject",
    type=str,
    default="",
    help="if None, will use subject info, if specified, the training subjects might be different from subject ",
)
parser.add_argument(
    "--testsubject",
    type=str,
    default="",
    help="if None, will use subject info, if specified, the test subjects might be different from subject ",
)
parser.add_argument(
    "--reshape", type=int, default=-1, help="-1 None, 0 no reshape, 1 reshape"
)
parser.add_argument(
    "--ld_loss_weight", type=int, default=1, help="ld_loss_weight use or not"
)
parser.add_argument(
    "--alpha_loss_weight", type=int, default=1, help="alpha_loss_weight use or not"
)
parser.add_argument(
    "--consonant_loss_weight",
    type=int,
    default=0,
    help="consonant_loss_weight use or not",
)
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument(
    "--param_file",
    type=str,
    default="configs/e2a_production.yaml",
    help="param_file",
)
parser.add_argument(
    "--pretrained_model_dir", type=str, default="output/a2a/NY869", help="pretrained_model_dir"
)
parser.add_argument("--causal", type=int, default=0, help="causal")
parser.add_argument("--anticausal", type=int, default=0, help="anticausal")
parser.add_argument("--rdropout", type=float, default=0, help="rdropout")
parser.add_argument("--epoch_num", type=int, default=100, help="epoch num")
parser.add_argument("--use_stoi", type=int, default=0, help="Use STOI+ loss or not")
parser.add_argument(
    "--use_denoise", type=int, default=0, help="Use denoise audio or not"
)
parser.add_argument('--noise_db',  type=float, default=-50,
                help='distill use or not ')
args_ = parser.parse_args()

with open("configs/AllSubjectInfo.json", "r") as rfile:
    allsubj_param = json.load(rfile)
with open(args_.param_file, 'r') as stream:
    param = yaml.safe_load(stream)


(
    ecog_all,
    wave_orig_all,
    x_orig_all,
    x_orig_amp_all,
    labels_all,
    gender_train_all,
    on_stage_all,
    on_stage_wider_all
) = ({}, {}, {}, {}, {}, {}, {}, {})
hann_win = torch.hann_window(21, periodic=False).reshape([1, 1, 21, 1])
hann_win = hann_win / hann_win.sum()

def get_train_data(
    ecog_all,
    wave_orig_all,
    x_orig_all,
    x_orig_amp_all,
    labels_all,
    gender_train_all,
    on_stage_all,
    on_stage_wider_all,
    sample_dict_train=None,
    subject=None,
):
    wave_orig_all[subject] = (
        sample_dict_train["wave_re_batch_all"].to(device).float()
    )
    gender_train_all[subject] =sample_dict_train['gender_all'].to(device).float()
    if cfg.MODEL.WAVE_BASED:
        x_orig_all[subject] = (
            sample_dict_train["wave_spec_re_batch_all"].to(device).float()
        )
        x_orig_amp_all[subject] = (
            sample_dict_train["wave_spec_re_amp_batch_all"].to(device).float()
        )
    on_stage_all[subject] = (
        sample_dict_train["on_stage_re_batch_all"].to(device).float()
    )
    on_stage_wider_all[subject] = (
        sample_dict_train["on_stage_wider_re_batch_all"].to(device).float()
    )
    labels_all[subject] = sample_dict_train["label_batch_all"]
    ecog_all[subject] = sample_dict_train["ecog_re_batch_all"].to(device).float()

    return (
        ecog_all,
        wave_orig_all,
        x_orig_all,
        x_orig_amp_all,
        labels_all,
        gender_train_all,
        on_stage_all,
        on_stage_wider_all,
        )

def load_model_checkpoint(
    logger,
    local_rank,
    distributed,
    tracker=None,
    tracker_test=None,
    dataset_all=None,
    subject="NY742",
    load_dir="",
    single_patient_mapping=0,param=None
):
    if args_.trainsubject != "":
        train_subject_info = args_.trainsubject.split(",")
    else:
        train_subject_info = cfg.DATASET.SUBJECT
    model = Model(
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        ecog_encoder_name=cfg.MODEL.MAPPING_FROM_ECOG,
        spec_chans=cfg.DATASET.SPEC_CHANS,
        n_formants=cfg.MODEL.N_FORMANTS,
        n_formants_noise=cfg.MODEL.N_FORMANTS_NOISE,
        n_formants_ecog=cfg.MODEL.N_FORMANTS_ECOG,
        wavebased=cfg.MODEL.WAVE_BASED,
        n_fft=cfg.MODEL.N_FFT,
        noise_db=cfg.MODEL.NOISE_DB,
        max_db=cfg.MODEL.MAX_DB,
        with_ecog=cfg.MODEL.ECOG,
        do_mel_guide=cfg.MODEL.DO_MEL_GUIDE,
        noise_from_data=cfg.MODEL.BGNOISE_FROMDATA and cfg.DATASET.PROD,
        specsup=cfg.FINETUNE.SPECSUP,
        power_synth=cfg.MODEL.POWER_SYNTH,
        apply_flooding=cfg.FINETUNE.APPLY_FLOODING,
        normed_mask=cfg.MODEL.NORMED_MASK,
        dummy_formant=cfg.MODEL.DUMMY_FORMANT,
        A2A=cfg.VISUAL.A2A,
        causal=cfg.MODEL.CAUSAL,
        anticausal=cfg.MODEL.ANTICAUSAL,
        pre_articulate=cfg.DATASET.PRE_ARTICULATE,
        alpha_sup=param["Subj"][subject][
            "AlphaSup"
        ],
        ld_loss_weight=cfg.MODEL.ld_loss_weight,
        alpha_loss_weight=cfg.MODEL.alpha_loss_weight,
        consonant_loss_weight=cfg.MODEL.consonant_loss_weight,
        component_regression=cfg.MODEL.component_regression,
        amp_formant_loss_weight=cfg.MODEL.amp_formant_loss_weight,
        freq_single_formant_loss_weight=cfg.MODEL.freq_single_formant_loss_weight,
        amp_minmax=cfg.MODEL.amp_minmax,
        amp_energy=cfg.MODEL.amp_energy,
        f0_midi=cfg.MODEL.f0_midi,
        alpha_db=cfg.MODEL.alpha_db,
        network_db=cfg.MODEL.network_db,
        consistency_loss=cfg.MODEL.consistency_loss,
        delta_time=cfg.MODEL.delta_time,
        delta_freq=cfg.MODEL.delta_freq,
        cumsum=cfg.MODEL.cumsum,
        distill=cfg.MODEL.distill,
        learned_mask=cfg.MODEL.LEARNED_MASK,
        n_filter_samples=cfg.MODEL.N_FILTER_SAMPLES,
        patient=subject,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        rdropout=cfg.MODEL.rdropout,
        dynamic_filter_shape=cfg.MODEL.DYNAMIC_FILTER_SHAPE,
        learnedbandwidth=cfg.MODEL.LEARNEDBANDWIDTH,
        gender_patient=allsubj_param["Subj"][train_subject_info[0]]["Gender"],
        reverse_order=args_.reverse_order,
        larger_capacity=args_.lar_cap,
        use_stoi=args_.use_stoi,
    )

    if torch.cuda.is_available():
        model.cuda(local_rank)
    model.train()

    model_s = Model(
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        ecog_encoder_name=cfg.MODEL.MAPPING_FROM_ECOG,
        spec_chans=cfg.DATASET.SPEC_CHANS,
        n_formants=cfg.MODEL.N_FORMANTS,
        n_formants_noise=cfg.MODEL.N_FORMANTS_NOISE,
        n_formants_ecog=cfg.MODEL.N_FORMANTS_ECOG,
        wavebased=cfg.MODEL.WAVE_BASED,
        n_fft=cfg.MODEL.N_FFT,
        noise_db=cfg.MODEL.NOISE_DB,
        max_db=cfg.MODEL.MAX_DB,
        with_ecog=cfg.MODEL.ECOG,
        do_mel_guide=cfg.MODEL.DO_MEL_GUIDE,
        noise_from_data=cfg.MODEL.BGNOISE_FROMDATA and cfg.DATASET.PROD,
        specsup=cfg.FINETUNE.SPECSUP,
        power_synth=cfg.MODEL.POWER_SYNTH,
        apply_flooding=cfg.FINETUNE.APPLY_FLOODING,
        normed_mask=cfg.MODEL.NORMED_MASK,
        dummy_formant=cfg.MODEL.DUMMY_FORMANT,
        A2A=cfg.VISUAL.A2A,
        causal=cfg.MODEL.CAUSAL,
        anticausal=cfg.MODEL.ANTICAUSAL,
        pre_articulate=cfg.DATASET.PRE_ARTICULATE,
        alpha_sup=param["Subj"][subject][
            "AlphaSup"
        ],
        ld_loss_weight=cfg.MODEL.ld_loss_weight,
        alpha_loss_weight=cfg.MODEL.alpha_loss_weight,
        consonant_loss_weight=cfg.MODEL.consonant_loss_weight,
        component_regression=cfg.MODEL.component_regression,
        amp_formant_loss_weight=cfg.MODEL.amp_formant_loss_weight,
        freq_single_formant_loss_weight=cfg.MODEL.freq_single_formant_loss_weight,
        amp_minmax=cfg.MODEL.amp_minmax,
        amp_energy=cfg.MODEL.amp_energy,
        f0_midi=cfg.MODEL.f0_midi,
        alpha_db=cfg.MODEL.alpha_db,
        network_db=cfg.MODEL.network_db,
        consistency_loss=cfg.MODEL.consistency_loss,
        delta_time=cfg.MODEL.delta_time,
        delta_freq=cfg.MODEL.delta_freq,
        cumsum=cfg.MODEL.cumsum,
        distill=cfg.MODEL.distill,
        learned_mask=cfg.MODEL.LEARNED_MASK,
        n_filter_samples=cfg.MODEL.N_FILTER_SAMPLES,
        patient=subject,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        rdropout=cfg.MODEL.rdropout,
        dynamic_filter_shape=cfg.MODEL.DYNAMIC_FILTER_SHAPE,
        learnedbandwidth=cfg.MODEL.LEARNEDBANDWIDTH,
        gender_patient=allsubj_param["Subj"][train_subject_info[0]]["Gender"],
        reverse_order=args_.reverse_order,
        larger_capacity=args_.lar_cap,
        use_stoi=args_.use_stoi,
    )
    if torch.cuda.is_available():
        model_s.cuda(local_rank)
    model_s.eval()
    model_s.requires_grad_(False)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True,
        )
        model.device_ids = None
        decoder = model.module.decoder
        encoder = model.module.encoder
        if hasattr(model.module, "ecog_encoder"):
            ecog_encoder = model.module.ecog_encoder
            if torch.cuda.is_available():
                ecog_encoder = ecog_encoder.cuda(local_rank)
            # ecog_encoder.performer.cuda(local_rank)
        if hasattr(model.module, "decoder_mel"):
            decoder_mel = model.module.decoder_mel
    else:
        decoder = model.decoder
        encoder = model.encoder
        if hasattr(model, "ecog_encoder"):
            ecog_encoder = model.ecog_encoder
            if torch.cuda.is_available():
                ecog_encoder = ecog_encoder.cuda(local_rank)
        if hasattr(model, "decoder_mel"):
            decoder_mel = model.decoder_mel
    logger.info("Trainable parameters generator:")
    logger.info("Trainable parameters discriminator:")
    arguments = dict()
    arguments["iteration"] = 0

    if cfg.MODEL.ECOG:
        if cfg.MODEL.SUPLOSS_ON_ECOGF:
            optimizer = LREQAdam(
                [{"params": ecog_encoder.parameters()}],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
        else:
            optimizer = LREQAdam(
                [
                    {"params": ecog_encoder.parameters()},
                    {"params": decoder.parameters()},
                ],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )

    else:
        if cfg.MODEL.DO_MEL_GUIDE:
            optimizer = LREQAdam(
                [
                    {"params": encoder.parameters()},
                    {"params": decoder.parameters()},
                    {"params": decoder_mel.parameters()},
                ],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
        else:
            optimizer = LREQAdam(
                [{"params": encoder.parameters()}, {"params": decoder.parameters()}],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
    model_dict = {
        "encoder": encoder,
        "generator": decoder,
    }
    if hasattr(model, "ecog_encoder"):
        model_dict["ecog_encoder"] = ecog_encoder
    if hasattr(model, "decoder_mel"):
        model_dict["decoder_mel"] = decoder_mel
    if local_rank == 0:
        model_dict["encoder_s"] = model_s.encoder.to(device)
        model_dict["generator_s"] = model_s.decoder.to(device)
        if hasattr(model_s, "ecog_encoder"):
            model_dict["ecog_encoder_s"] = model_s.ecog_encoder.to(device)
        if hasattr(model_s, "decoder_mel"):
            model_dict["decoder_mel_s"] = model_s.decoder_mel
    noise_dist = torch.from_numpy(dataset_all[subject].noise_dist).to(device).float()
    if cfg.MODEL.BGNOISE_FROMDATA:
        model_s.noise_dist_init(noise_dist)
        model.noise_dist_init(noise_dist)
    if cfg.MODEL.ECOG:
        if cfg.MODEL.SUPLOSS_ON_ECOGF:
            optimizer = LREQAdam(
                [{"params": ecog_encoder.parameters()}],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
        else:
            optimizer = LREQAdam(
                [
                    {"params": ecog_encoder.parameters()},
                    {"params": decoder.parameters()},
                ],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
    else:
        if cfg.MODEL.DO_MEL_GUIDE:
            optimizer = LREQAdam(
                [
                    {"params": encoder.parameters()},
                    {"params": decoder.parameters()},
                    {"params": decoder_mel.parameters()},
                ],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
        else:
            optimizer = LREQAdam(
                [
                    {"params": encoder.parameters()},
                    {"params": decoder.parameters()},
                ],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
    tracker = LossTracker(cfg.OUTPUT_DIR)
    tracker_test = LossTracker(cfg.OUTPUT_DIR, test=True)
    auxiliary = {
        "optimizer": optimizer,
        "tracker": tracker,
        "tracker_test": tracker_test,
    }
    checkpointer = Checkpointer(
        cfg, model_dict, auxiliary, logger=logger, save=local_rank == 0
    )
    if LOAD:
        extra_checkpoint_data = checkpointer.load(
            ignore_last_checkpoint=True if not LOAD else False,
            ignore_auxiliary=True,
            file_name=load_dir,
        )
        arguments.update(extra_checkpoint_data)
    return (
        checkpointer,
        model,
        model_s,
        encoder,
        decoder,
        ecog_encoder,
        optimizer,
        tracker,
        tracker_test,
    )

def train(cfg, logger, local_rank, world_size, distributed):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    with open('configs/train_param_production.json', 'r') as stream:
        param = json.load(stream)
    dataset_all, dataset_test_all = {}, {}
    train_subject_info = args_.trainsubject.split(",") if args_.trainsubject != "" else cfg.DATASET.SUBJECT 
    test_subject_info = args_.testsubject.split(",") if args_.testsubject != "" else cfg.DATASET.SUBJECT

    for subject in np.union1d(train_subject_info, test_subject_info):
        dataset_all[subject] = TFRecordsDataset(
            cfg,
            logger,
            rank=local_rank,
            world_size=world_size,
            SUBJECT=[subject],
            buffer_size_mb=1024,
            channels=cfg.MODEL.CHANNELS,
            param=param,
            allsubj_param=allsubj_param,
            ReshapeAsGrid=1,
            rearrange_elec=0,
            low_density=cfg.DATASET.DENSITY == "LD",
            process_ecog=True,
        )

    for subject in test_subject_info:
        dataset_test_all[subject] = TFRecordsDataset(
            cfg,
            logger,
            rank=local_rank,
            world_size=world_size,
            SUBJECT=[subject],
            buffer_size_mb=1024,
            channels=cfg.MODEL.CHANNELS,
            train=False,
            param=param,
            allsubj_param=allsubj_param,
            ReshapeAsGrid=1,
            rearrange_elec=0,
            low_density=cfg.DATASET.DENSITY == "LD",
            process_ecog=True,
        )
    tracker = LossTracker(cfg.OUTPUT_DIR)
    tracker_test = LossTracker(cfg.OUTPUT_DIR, test=True)

    (checkpointer_all,
        model_all,
        model_s_all,
        encoder_all,
        decoder_all,
        ecog_encoder_all,
        optimizer_all,
    ) = ({}, {}, {}, {}, {}, {}, {})

    for single_patient_mapping, subject in enumerate(
        np.union1d(train_subject_info, test_subject_info)
    ): 
        if args_.pretrained_model_dir != "":
            load_sub_dir = args_.pretrained_model_dir
            max_epoch = (
                np.array(
                    [
                        i.split('epoch')[1].split('.pth')[0]
                        for i in os.listdir(load_sub_dir)
                        if i.endswith("pth")
                    ]
                )
                .astype("int")
                .max()
            )
            load_sub_name = [i for i in load_sub_dir.split("/") if "NY" in i][0]
            print("subject, load_sub_name", subject, load_sub_name)
            load_sub_dir = load_sub_dir + "/{}_a2a_model_epoch{}.pth".format(
                   load_sub_name, max_epoch  
                )
            print("pretrained load dir", load_sub_dir)
        else:
            load_sub_dir = ''
            print ('No pretrainde a2a model provided!')
            #raise Exception("Please Provide pretrained_model_dir")
        (
            checkpointer_all[subject],
            model_all[subject],
            model_s_all[subject],
            encoder_all[subject],
            decoder_all[subject],
            ecog_encoder_all[subject],
            optimizer_all[subject],
            tracker,
            tracker_test,
        ) = load_model_checkpoint(
            logger,
            local_rank,
            distributed,
            tracker=tracker,
            tracker_test=tracker_test,
            dataset_all=dataset_all,
            subject=subject,
            load_dir=load_sub_dir,
            single_patient_mapping=single_patient_mapping,param=param
        )
    loadsub = train_subject_info[0]
    ecog_encoder_shared = ecog_encoder_all[loadsub]

    for single_patient_mapping, subject in enumerate(
        np.union1d(train_subject_info, test_subject_info)
    ):
        model_all[
            subject
        ].ecog_encoder = ecog_encoder_shared
        model_s_all[
            subject
        ].ecog_encoder = ecog_encoder_shared
    (   
        ecog_test_all,
        sample_wave_test_all,
        sample_spec_test_all,
        sample_spec_amp_test_all,
        sample_label_test_all,
        gender_test_all,
        on_stage_test_all,
        on_stage_wider_test_all,
    ) = (
        {},{},{},{},{},{},{},{})

    hann_win = torch.hann_window(21, periodic=False).reshape([1, 1, 21, 1])
    hann_win = hann_win / hann_win.sum()
    x_amp_from_denoise = False

    for subject in test_subject_info:
        dataset_test_all[subject].reset(
            cfg.DATASET.MAX_RESOLUTION_LEVEL, len(dataset_test_all[subject].dataset)
        )
        sample_dict_test = next(iter(dataset_test_all[subject].iterator))
        gender_test_all[subject] = sample_dict_test['gender_all'].to(device).float()
        if cfg.DATASET.PROD:
            sample_wave_test_all[subject] = (
                sample_dict_test["wave_re_batch_all"].to(device).float()
            )
            if cfg.MODEL.WAVE_BASED:
                sample_spec_test_all[subject] = (
                    sample_dict_test["wave_spec_re_batch_all"].to(device).float()
                )
                sample_spec_amp_test_all[subject] = (
                    sample_dict_test["wave_spec_re_amp_batch_all"]
                    .to(device)
                    .float()
                )
            sample_label_test_all[subject] = sample_dict_test["label_batch_all"]
            if cfg.MODEL.ECOG:
                ecog_test_all[subject] = sample_dict_test["ecog_re_batch_all"].to(device).float()
            on_stage_test_all[subject] = (
                sample_dict_test["on_stage_re_batch_all"].to(device).float()
            )
            on_stage_wider_test_all[subject] = (
                sample_dict_test["on_stage_wider_re_batch_all"].to(device).float()
            )
    duomask = True
    x_amp_from_denoise = False
    n_iter = 0

    (
        ecog_all,
        wave_orig_all,
        x_orig_all,
        x_orig_amp_all,
        labels_all,
        gender_train_all,
        on_stage_all,
        on_stage_wider_all
    ) = (
        {},{},{},{},{},{},{},{}
    )
    for epoch in tqdm(range(cfg.TRAIN.TRAIN_EPOCHS)):
        
        
        #train
        for subject in train_subject_info:
            model_all[subject].train()
        i = 0
        dataset_iterator_all = {}
        if len(train_subject_info) <= 1:
            dataset_iterator_all[train_subject_info[0]] = iter(
                dataset_all[train_subject_info[0]].iterator
            )
            sample_dict_train_all = {}
            for sample_dict_train_all[train_subject_info[0]] in tqdm(
                dataset_iterator_all[train_subject_info[0]]
            ):
                n_iter += 1
                i += 1
                for subject in train_subject_info:
                    if n_iter % 200 == 0:
                        print(tracker.register_means(n_iter))
                    (
                        ecog_all,
                        wave_orig_all,
                        x_orig_all,
                        x_orig_amp_all,
                        labels_all,
                        gender_train_all,
                        on_stage_all,
                        on_stage_wider_all
                    ) = get_train_data(
                        ecog_all,
                        wave_orig_all,
                        x_orig_all,
                        x_orig_amp_all,
                        labels_all,
                        gender_train_all,
                        on_stage_all,
                        on_stage_wider_all,
                        sample_dict_train_all[train_subject_info[0]],
                        subject=subject,
                    )
                    initial = None
                    
                    optimizer_all[subject].zero_grad()
                    Lrec, tracker = model_all[subject](
                        x_orig_all[subject],
                        ecog=ecog_all[subject],
                        on_stage=on_stage_all[subject],
                        on_stage_wider=on_stage_all[subject],
                        ae=False,
                        tracker=tracker,
                        encoder_guide=cfg.MODEL.W_SUP,
                        duomask=duomask,
                        x_amp=x_orig_amp_all[subject],
                        x_amp_from_denoise=x_amp_from_denoise,
                        gender=gender_train_all[subject],
                    )
                    (Lrec).backward()
                    optimizer_all[subject].step()

                    betta = 0.5 ** (cfg.TRAIN.BATCH_SIZE / (10 * 1000.0))
                    model_s_all[subject].lerp(
                        model_all[subject],
                        betta,
                        w_classifier=cfg.MODEL.W_CLASSIFIER,
                    )

        #test
        for subject in test_subject_info:
            print(
                2
                ** (
                    torch.tanh(
                        model_all[subject].encoder.formant_bandwitdh_slop
                    )
                )
            )
            print("save test result!")

            model_all[subject].eval()
            Lrec = model_all[subject](
                sample_spec_test_all[subject],
                x_denoise=None,
                x_mel=None,
                ecog=ecog_test_all[subject] if cfg.MODEL.ECOG else None,
                on_stage=on_stage_test_all[subject],
                ae=not cfg.MODEL.ECOG,
                tracker=tracker_test,
                encoder_guide=cfg.MODEL.W_SUP,
                pitch_aug=False,
                duomask=duomask,
                debug=False,
                x_amp=sample_spec_amp_test_all[subject],
                hamonic_bias=False,
                gender=gender_test_all[subject],
                on_stage_wider=on_stage_test_all[subject],
            )

            initial = None

            if epoch % 10 == 0:
                checkpointer_all[subject].save(
                    "model_epoch{}_{}".format(epoch, subject)
                )
                save_sample(
                    cfg,
                    sample_spec_test_all[subject],
                    ecog_test_all[subject],
                    encoder_all[subject],
                    decoder_all[subject],
                    ecog_encoder_shared
                    if hasattr(model_all[subject], "ecog_encoder")
                    else None,
                    encoder2
                    if hasattr(model_all[subject], "encoder2")
                    else None,
                    x_denoise=None,
                    decoder_mel=decoder_mel if cfg.MODEL.DO_MEL_GUIDE else None,
                    epoch=epoch,
                    label=sample_label_test_all[subject],
                    mode="test",
                    path=cfg.OUTPUT_DIR,
                    tracker=tracker_test,
                    linear=cfg.MODEL.WAVE_BASED,
                    n_fft=cfg.MODEL.N_FFT,
                    duomask=duomask,
                    x_amp=sample_spec_amp_test_all[subject],
                    gender=gender_test_all[subject],
                    sample_wave=sample_wave_test_all[subject],
                    sample_wave_denoise=None,
                    on_stage_wider=on_stage_test_all[subject],
                    auto_regressive=False,
                    seq_out_start=initial,
                    suffix=subject,
                )

        

if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    cfg = get_cfg_defaults()
    if args_.trainsubject != "":
        train_subject_info = args_.trainsubject.split(",")
    else:
        train_subject_info = cfg.DATASET.SUBJECT
    if args_.testsubject != "":
        test_subject_info = args_.testsubject.split(",")
    else:
        test_subject_info = cfg.DATASET.SUBJECT
    with open("configs/AllSubjectInfo.json", "r") as rfile:
        allsubj_param = json.load(rfile)
    subj_param = allsubj_param["Subj"][args_.trainsubject.split(",")[0]]
    Gender = subj_param["Gender"] if cfg.DATASET.PROD else "Female"
    config_file = args_.param_file
    cfg.merge_from_file(config_file)
    args_.config_file = config_file

    run(
    train,
    cfg,
    description="ECoG to Audio",
    default_config=config_file,
    world_size=gpu_count,
    args_=args_,
    )