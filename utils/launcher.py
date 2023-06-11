# Copyright 2019 Stanislav Pidhorskyi
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

import os
import sys
import argparse
import logging
import torch
import torch.multiprocessing as mp
from torch import distributed
import inspect


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    distributed.destroy_process_group()


def _run(rank, world_size, fn, defaults, write_log, no_cuda, args_):
    if world_size > 1:
        setup(rank, world_size)
    if not no_cuda and torch.cuda.is_available():
        torch.cuda.set_device(rank)
    print ('rank in _run', rank)
    cfg = defaults
    config_file = args_.config_file
    if len(os.path.splitext(config_file)[1]) == 0:
        config_file += '.yaml'
    if not os.path.exists(config_file) and os.path.exists(os.path.join('configs', config_file)):
        config_file = os.path.join('configs', config_file)
    cfg.merge_from_file(config_file)
    #cfg.merge_from_list(args_.opts)
    if cfg.FINETUNE.FINETUNE:
        cfg.MODEL.ECOG = True
        cfg.MODEL.SUPLOSS_ON_ECOGF = cfg.FINETUNE.FIX_GEN
        cfg.MODEL.W_SUP = cfg.FINETUNE.ENCODER_GUIDE
    if not cfg.MODEL.POWER_SYNTH:
        cfg.MODEL.NOISE_DB = cfg.MODEL.NOISE_DB_AMP
        cfg.MODEL.MAX_DB = cfg.MODEL.MAX_DB_AMP
    cfg.TRAIN.LOD_2_BATCH_1GPU = [bs//len(cfg.DATASET.SUBJECT) for bs in cfg.TRAIN.LOD_2_BATCH_1GPU]
    cfg.TRAIN.LOD_2_BATCH_2GPU = [bs//len(cfg.DATASET.SUBJECT) for bs in cfg.TRAIN.LOD_2_BATCH_2GPU]
    cfg.TRAIN.LOD_2_BATCH_4GPU = [bs//len(cfg.DATASET.SUBJECT) for bs in cfg.TRAIN.LOD_2_BATCH_4GPU]
    cfg.TRAIN.LOD_2_BATCH_8GPU = [bs//len(cfg.DATASET.SUBJECT) for bs in cfg.TRAIN.LOD_2_BATCH_8GPU]
    cfg.TRAIN.TRAIN_EPOCHS = args_.epoch_num
    cfg.MODEL.RNN_COMPUTE_DB_LOUDNESS  = True if args_.RNN_COMPUTE_DB_LOUDNESS==1 else False
    cfg.MODEL.BIDIRECTION  = True if args_.BIDIRECTION==1 else False
    cfg.OUTPUT_DIR = args_.OUTPUT_DIR
    cfg.MODEL.MAPPING_FROM_ECOG = args_.MAPPING_FROM_ECOG
    cfg.MODEL.EXPERIMENT_KEY = args_.COMPONENTKEY if args_.COMPONENTKEY!='' else 0
    cfg.MODEL.TRANSFORMER.FASTATTENTYPE = 'full'
    cfg.MODEL.PHONEMEWEIGHT = 0
    cfg.DATASET.SUBJECT = args_.trainsubject.split(',')#[args_.subject]
    cfg.DATASET.DENSITY = args_.DENSITY
    cfg.MODEL.ld_loss_weight = args_.ld_loss_weight
    cfg.MODEL.alpha_loss_weight = args_.alpha_loss_weight
    cfg.MODEL.consonant_loss_weight = args_.consonant_loss_weight
    cfg.MODEL.component_regression = False
    cfg.MODEL.CAUSAL = args_.causal
    cfg.MODEL.ANTICAUSAL = args_.anticausal
    cfg.MODEL.freq_single_formant_loss_weight = False
    cfg.MODEL.amp_minmax = False
    cfg.MODEL.mapping_layers = 0
    cfg.MODEL.region_index = 0
    cfg.MODEL.multiscale= 0
    cfg.MODEL.rdropout= 0
    cfg.MODEL.LEARNED_MASK = bool(args_.learnedmask)
    cfg.MODEL.DYNAMIC_FILTER_SHAPE = bool(args_.dynamicfiltershape)
    cfg.IGNORE_LOADING= args_.ignore_loading
    cfg.FINETUNE.FINETUNE= args_.finetune
    cfg.MODEL.single_patient_mapping = -1
    cfg.MODEL.amp_formant_loss_weight = 0
    cfg.MODEL.amp_energy = 0
    cfg.MODEL.f0_midi = 0
    cfg.MODEL.alpha_db = 0
    cfg.MODEL.network_db = 0
    cfg.MODEL.consistency_loss = 0
    cfg.MODEL.delta_time = 0
    cfg.MODEL.delta_freq = 0
    cfg.MODEL.cumsum = 0
    cfg.MODEL.distill = 0
    
    cfg.MODEL.classic_pe = 0
    cfg.MODEL.temporal_down_before = 0
    cfg.MODEL.conv_method = 'both'
    cfg.MODEL.classic_attention = True
    cfg.MODEL.WAVE_BASED = bool(args_.wavebased)
    cfg.TRAIN.BATCH_SIZE = args_.batch_size
    cfg.MODEL.BGNOISE_FROMDATA = bool(args_.bgnoise_fromdata)
    cfg.MODEL.NOISE_DB = args_.noise_db
    cfg.MODEL.RETURNFILTERSHAPE = 0
    cfg.MODEL.N_FILTER_SAMPLES = args_.n_filter_samples
    cfg.MODEL.N_FFT = args_.n_fft
    
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    

    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if write_log:
            filepath = os.path.join(output_dir, 'log.txt')
            if isinstance(write_log, str):
                filepath = write_log
            fh = logging.FileHandler(filepath)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    logger.info(args_)

    logger.info("World size: {}".format(world_size))

    logger.info("Loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        #logger.info(config_str)
    #logger.info("Running with config:\n{}".format(cfg))

    if not no_cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.cuda.current_device()
        print("Running on ", torch.cuda.get_device_name(device))

    args_.distributed = world_size > 1
    args__to_pass = dict(cfg=cfg, logger=logger, local_rank=rank, world_size=world_size, distributed=args_.distributed)
    signature = inspect.signature(fn)
    matching_args_ = {}
    for key in args__to_pass.keys():
        if key in signature.parameters.keys():
            matching_args_[key] = args__to_pass[key]
    fn(**matching_args_)

    if world_size > 1:
        cleanup()


def run(fn, defaults, description='', default_config='configs/experiment.yaml', world_size=1, write_log=False, no_cuda=False,args_=None):
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    world_size = max(world_size, 1) #avoid cpu running issue
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(cpu_count / world_size)))
    del multiprocessing

    if world_size > 1:
        mp.spawn(_run, args=(world_size, fn, defaults, write_log, no_cuda, args_), nprocs=world_size, join=True)
    else:
        _run(0, world_size, fn, defaults, write_log, no_cuda, args_)
