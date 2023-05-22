import json
import pdb
import torch
import os
import numpy as np
import scipy.io
from scipy import signal
import h5py
import mat73
import random
import pandas
from torch.utils.data import Dataset
from defaults import get_cfg_defaults
from net_formant_ran import wave2spec
from scipy.interpolate import LinearNDInterpolator
from scipy.io import savemat
import augmentor
import pandas as pd




def what_phonemes_ind(phonemes,which_phoneme):
    index = np.zeros(len(which_phoneme))
    for v in range(index.shape[0]):
        index[v] = phonemes.index(which_phoneme[v])
    return index

class ECoGDataset(Dataset):
    """docstring for ECoGDataset"""
    def zscore(self,ecog,badelec,axis=None):
        if badelec.shape[0] != 0:
            statics_ecog = np.delete(ecog,badelec,axis=1).mean(axis=axis, keepdims=True)+1e-10,np.delete(ecog,badelec,axis=1).std(axis=axis, keepdims=True)+1e-10
        else:
            statics_ecog = ecog.mean(axis=axis, keepdims=True)+1e-10,ecog.std(axis=axis, keepdims=True)+1e-10
        # statics_ecog = ecog.mean(axis=axis, keepdims=True)+1e-10,ecog.std(axis=axis, keepdims=True)+1e-10
        ecog = (ecog-statics_ecog[0])/statics_ecog[1]
        return ecog, statics_ecog

    def rearrange(self,data,crop=None,mode = 'ecog'):
        rows = [0,1,2,3,4,5,6,8,9,10,11]
        starts = [1,0,1,0,1,0,1,7,6,7,7]
        ends = [6,6,6,9,12,14,12,14,14,14,8]
        strides = [2,1,2,1,2,1,2,2,1,2,1]
        electrodes = [64,67,73,76,85,91,105,111,115,123,127,128]
        if mode == 'ecog':
            data_new = np.zeros((data.shape[0],15,15))
            data_new[:,::2,::2] = np.reshape(data[:,:64],[-1,8,8])
            for i in range(len(rows)):
                data_new[:,rows[i],starts[i]:ends[i]:strides[i]] = data[:,electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(data_new,[data.shape[0],-1])
            else:
                return np.reshape(data_new[:,crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[data.shape[0],-1]) # TxN

        elif mode == 'coord':
            # data_new = np.zeros((15,15,data.shape[-1]))
            # data_new[::2,::2] = np.reshape(data[:64],[8,8,-1])
            x_new,y_new = np.meshgrid(np.arange(15).astype(np.float32),np.arange(15).astype(np.float32))
            x_org = x_new[::2,::2].flatten()
            y_org = y_new[::2,::2].flatten()
            x_org_ = [x_org]
            y_org_ = [y_org]
            for i in range(len(rows)):
                # data_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
                x_org_+=[x_new[rows[i],starts[i]:ends[i]:strides[i]].flatten()]
                y_org_+=[y_new[rows[i],starts[i]:ends[i]:strides[i]].flatten()]
            x_org = np.concatenate(x_org_)
            y_org = np.concatenate(y_org_)
            corrd_x = data[...,0]
            corrd_y = data[...,1]
            corrd_z = data[...,2]
            interpx = LinearNDInterpolator(list(zip(x_org, y_org)), corrd_x)
            interpy = LinearNDInterpolator(list(zip(x_org, y_org)), corrd_y)
            interpz = LinearNDInterpolator(list(zip(x_org, y_org)), corrd_z)
            x_interped = interpx(x_new,y_new)
            y_interped = interpy(x_new,y_new)
            z_interped = interpz(x_new,y_new)
            data_new = np.stack([x_interped,y_interped,z_interped],axis=-1)

            if crop is None:
                return np.reshape(data_new,[-1,data.shape[-1]]) # Nx3
            else:
                return np.reshape(data_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1,data.shape[-1]]) # Nx3

        elif mode == 'region':
            region_new = np.chararray((15,15),itemsize=100)
            region_new[:] = 'nan'
            region_new[::2,::2] = np.reshape(data[:64],[8,8])
            for i in range(len(rows)):
                region_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(region_new,[-1])
            else:
                return np.reshape(region_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1])

        elif mode == 'mask':
            data_new = np.zeros((15,15))
            data_new[::2,::2] = np.reshape(data[:64],[8,8])
            for i in range(len(rows)):
                data_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(data_new,[-1])
            else:
                return np.reshape(data_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1])
            
    def rearrange_LD(self,data,crop=None,mode = 'ecog',nums = 64):
        #rows = [0,1,2,3,4,5,6,8,9,10,11]
        #starts = [1,0,1,0,1,0,1,7,6,7,7]
        #ends = [6,6,6,9,12,14,12,14,14,14,8]
        #strides = [2,1,2,1,2,1,2,2,1,2,1]
        #electrodes = [64,67,73,76,85,91,105,111,115,123,127,128]
        square_shape = int(np.sqrt(nums))
        if mode == 'ecog':
            #data_new = np.zeros((data.shape[0],8,8))
            #data_new[:,::2,::2] = np.reshape(data[:,:64],[-1,8,8])
            data_new = np.reshape(data[:,:nums],[-1,square_shape,square_shape])
            #for i in range(len(rows)):
            #    data_new[:,rows[i],starts[i]:ends[i]:strides[i]] = data[:,electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(data_new,[data.shape[0],-1])
            else:
                return np.reshape(data_new[:,crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[data.shape[0],-1]) # TxN

        elif mode == 'coord':
            data_new = np.zeros((square_shape,square_shape,data.shape[-1]))
            data_new  = np.reshape(data[:nums],[square_shape,square_shape,-1])
            x_new,y_new = np.meshgrid(np.arange(square_shape).astype(np.float32),np.arange(square_shape).astype(np.float32))
            x_org = x_new .flatten()
            y_org = y_new .flatten()
            x_org_ = [x_org]
            y_org_ = [y_org]
            #for i in range(len(rows)):
                # data_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
                #x_org_+=[x_new[rows[i],starts[i]:ends[i]:strides[i]].flatten()]
                #y_org_+=[y_new[rows[i],starts[i]:ends[i]:strides[i]].flatten()]
            x_org = np.concatenate(x_org_)
            y_org = np.concatenate(y_org_)
            corrd_x = data[...,0]
            corrd_y = data[...,1]
            corrd_z = data[...,2]
            interpx = LinearNDInterpolator(list(zip(x_org, y_org)), corrd_x)
            interpy = LinearNDInterpolator(list(zip(x_org, y_org)), corrd_y)
            interpz = LinearNDInterpolator(list(zip(x_org, y_org)), corrd_z)
            x_interped = interpx(x_new,y_new)
            y_interped = interpy(x_new,y_new)
            z_interped = interpz(x_new,y_new)
            data_new = np.stack([x_interped,y_interped,z_interped],axis=-1)

            if crop is None:
                return np.reshape(data_new,[-1,data.shape[-1]]) # Nx3
            else:
                return np.reshape(data_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1,data.shape[-1]]) # Nx3

        elif mode == 'region':
            region_new = np.chararray((square_shape,square_shape),itemsize=100)
            region_new[:] = 'nan'
            region_new  = np.reshape(data[:nums],[square_shape,square_shape])
            #for i in range(len(rows)):
            #    region_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(region_new,[-1])
            else:
                return np.reshape(region_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1])

        elif mode == 'mask':
            data_new = np.zeros((square_shape,square_shape))
            data_new  = np.reshape(data[:nums],[square_shape,square_shape])
            #for i in range(len(rows)):
            #    data_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(data_new,[-1])
            else:
                return np.reshape(data_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1])
    def transform(self, ):
        return augmentor.Compose(augmentor.RandomizedDropout(0.2, apply_p=0.2),
        #augmentor.Normalize(torch.tensor(dataset.mean), torch.tensor(dataset.std)),
        augmentor.Noise(1.5),
        augmentor.Pepper(0.3, 0.3, apply_p=0.5))
    
    def select_block(self,ecog,regions,mask,mni_coord,mni_coord_raw,T1_coord,T1_coord_raw,select,block):
        if not select and not block:
            return ecog,regions,mask,mni_coord,mni_coord_raw,T1_coord,T1_coord_raw
        if self.ReshapeAsGrid:
            if select:
                ecog_ = np.zeros(ecog.shape)
                mask_ = np.zeros(mask.shape)
                mni_coord_ = np.zeros(mni_coord.shape)
                mni_coord_raw_ = np.zeros(mni_coord_raw.shape)
                T1_coord_ = np.zeros(T1_coord.shape)
                T1_coord_raw_ = np.zeros(T1_coord_raw.shape)
                for region in select:
                    region_ind = [region.encode() == regions[i] for i in range(regions.shape[0])]
                    ecog_[:,region_ind] = ecog[:,region_ind]
                    mask_[region_ind] = mask[region_ind]
                    mni_coord_[region_ind] = mni_coord[region_ind]
                    mni_coord_raw_[region_ind] = mni_coord_raw[region_ind]
                    T1_coord_[region_ind] = T1_coord[region_ind]
                    T1_coord_raw_[region_ind] = T1_coord_raw[region_ind]
                return ecog_,regions,mask_,mni_coord_,mni_coord_raw_,T1_coord_,T1_coord_raw_
            if block:
                for region in block:
                    region_ind = [region.encode() == regions[i] for i in range(regions.shape[0])]
                    ecog[:,region_ind] = 0
                    mask[region_ind] = 0
                    mni_coord[region_ind]=0
                    mni_coord_raw[region_ind]=0
                    T1_coord[region_ind]=0
                    T1_coord_raw[region_ind]=0
                return ecog,regions,mask,mni_coord,mni_coord_raw,T1_coord,T1_coord_raw
        else:
            # region_ind = np.ones(regions.shape[0],dtype=bool)
            region_ind = np.array([])
            if select:
                # region_ind = np.zeros(regions.shape[0],dtype=bool)
                for region in select:
                    region_ind = np.concatenate([region_ind, np.where(np.array([region in regions[i] for i in range(regions.shape[0])]))[0]])
            if block:
                # region_ind = np.zeros(regions.shape[0],dtype=bool)
                for region in block:
                    # region_ind = np.logical_or(region_ind, np.array([region in regions[i] for i in range(regions.shape[0])]))
                    region_ind = np.concatenate([region_ind, np.where(np.array([region in regions[i] for i in range(regions.shape[0])]))[0]])
                # region_ind = np.logical_not(region_ind)
                region_ind = np.delete(np.arange(regions.shape[0]),region_ind)
            region_ind = region_ind.astype(np.int64)
            return ecog[:,region_ind],regions[region_ind],mask[region_ind],mni_coord[region_ind],mni_coord_raw[region_ind],T1_coord[region_ind],T1_coord_raw[region_ind]
    
    def __init__(self, cfg,ReqSubjDict, mode = 'train', train_param = None,BCTS=None,world_size=1,\
        ReshapeAsGrid=None,DEBUG=False, rearrange_elec=0, low_density = True, process_ecog = True, \
        formant_label = False, pitch_label = False, intensity_label = False,allsubj_param=None,use_denoise = False,FAKE_LD=False,extend_grid=False):
        """ ReqSubjDict can be a list of multiple subjects"""
        super(ECoGDataset, self).__init__()
        
        self.DEBUG = DEBUG
        self.FAKE_LD = FAKE_LD #only true for HB e2a exp but only use first 64 electrodes!
        self.extend_grid = extend_grid #true if we want to extend the grid to more than LD (64 to 100?)
        self.world_size = world_size
        self.current_lod=2
        self.ReqSubjDict = ReqSubjDict
        self.mode = mode
        BCTS = cfg.DATASET.BCTS
        self.BCTS = BCTS
        self.rearrange_elec = rearrange_elec#, rearrange_elec=True
        self.SpecBands = cfg.DATASET.SPEC_CHANS
        self.pre_articulate = cfg.DATASET.PRE_ARTICULATE
        self.process_ecog = process_ecog
        self.low_density = low_density
        self.formant = formant_label
        self.pitch = pitch_label
        self.intensity = intensity_label
        self.FAKE_REPROD = cfg.DATASET.FAKE_REPROD
        if not cfg.MODEL.POWER_SYNTH:
            cfg.MODEL.NOISE_DB = cfg.MODEL.NOISE_DB_AMP
            cfg.MODEL.MAX_DB = cfg.MODEL.MAX_DB_AMP
        self.allsubj_param =  allsubj_param
        #with open('AllSubjectInfo.json','r') as rfile:
        #    allsubj_param = json.load(rfile)
        if train_param == None:
            #print ('train_param == None')
            with open('train_param_e2a_production.json','r') as rfile:
                train_param = json.load(rfile)
        else:
            pass
            #print ('train_param not none',train_param)

        self.rootpath = allsubj_param['Shared']['RootPath']
        if low_density:
            #allsubj_param['Shared']['RootPath'] = '/scratch/xc1490/home/projects/ecog/ALAE_1023/data/data/'
            self.rootpath = '/scratch/xc1490/home/projects/ecog/ALAE_1023/data/data/'
            self.phoneme = False
        else:
            self.phoneme = True
        self.use_denoise = use_denoise
        self.ORG_WAVE_FS = allsubj_param['Shared']['ORG_WAVE_FS']
        self.ORG_ECOG_FS = allsubj_param['Shared']['ORG_ECOG_FS']
        self.DOWN_WAVE_FS  = allsubj_param['Shared']['DOWN_WAVE_FS']
        self.ORG_ECOG_FS_NY = allsubj_param['Shared']['ORG_ECOG_FS_NY']
        self.ORG_TF_FS = allsubj_param['Shared']['ORG_TF_FS']
        self.PHONEMES = allsubj_param['Shared']['PHONEMES']
        self.VOICE = allsubj_param['Shared']['VOICE']
        self.UNVOICE = allsubj_param['Shared']['UNVOICE']
        self.SEMIVOICE = allsubj_param['Shared']['SEMIVOICE']
        self.PLOSIVE = allsubj_param['Shared']['PLOSIVE']
        self.FRICATIVE = allsubj_param['Shared']['FRICATIVE']
        voice = what_phonemes_ind(self.PHONEMES,self.VOICE)
        unvoice = what_phonemes_ind(self.PHONEMES,self.UNVOICE)
        semivoice = what_phonemes_ind(self.PHONEMES,self.SEMIVOICE)
        plosive = what_phonemes_ind(self.PHONEMES,self.PLOSIVE)
        fricative = what_phonemes_ind(self.PHONEMES,self.FRICATIVE)
        self.cortex = {}
        self.cortex.update({"AUDITORY" : allsubj_param['Shared']['AUDITORY']})
        self.cortex.update({"BROCA" : allsubj_param['Shared']['BROCA']})
        self.cortex.update({"MOTO" : allsubj_param['Shared']['MOTO']})
        self.cortex.update({"SENSORY" : allsubj_param['Shared']['SENSORY']})
        self.cortex.update({"PARIETAL" : allsubj_param['Shared']['PARIETAL']})
        self.SelectRegion = []
        [self.SelectRegion.extend(self.cortex[area]) for area in cfg.DATASET.SELECTREGION]
        self.BlockRegion = []
        [self.BlockRegion.extend(self.cortex[area]) for area in cfg.DATASET.BLOCKREGION]
        self.wavebased = cfg.MODEL.WAVE_BASED
        if ReshapeAsGrid is not None:
            self.ReshapeAsGrid = ReshapeAsGrid
        else:
            self.ReshapeAsGrid = False if ('lstm_' in cfg.MODEL.MAPPING_FROM_ECOG) or ('Transformer' in cfg.MODEL.MAPPING_FROM_ECOG) or ('Performer' in cfg.MODEL.MAPPING_FROM_ECOG) or ('ECoGMappingPerformer2Dconv_downsample1_posemb' in cfg.MODEL.MAPPING_FROM_ECOG) else True
        print ('self.ReshapeAsGrid: ',self.ReshapeAsGrid,cfg.MODEL.MAPPING_FROM_ECOG)

        # self.ReshapeAsGrid = False if (('Transformer' in cfg.MODEL.MAPPING_FROM_ECOG or 'Performer' in cfg.MODEL.MAPPING_FROM_ECOG) or not train_param["ReshapeAsGrid"]) else True
        self.UseGridOnly,self.SeqLen = train_param['UseGridOnly'],\
                                                                    train_param['SeqLen'],
        self.Prod = cfg.DATASET.PROD
        self.ahead_onset_test = train_param['Test']['ahead_onset']
        self.ahead_onset_train = train_param['Train']['ahead_onset']
        self.DOWN_TF_FS = train_param['DOWN_TF_FS']
        self.DOWN_ECOG_FS = train_param['DOWN_ECOG_FS']
        self.TestNum_cum=np.array([],dtype=np.int32)
        self.Wipenoise = False
        print ('cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB',cfg.MODEL.NOISE_DB, cfg.MODEL.MAX_DB)
                                                         
        datapath = []
        analysispath = []
        ecog_alldataset = []
        ecog_raw_alldataset = []
        spkr_alldataset = []
        spkr_re_alldataset = []
        phoneme_alldataset = []
        voice_alldataset = []
        unvoice_alldataset = []
        semivoice_alldataset = []
        plosive_alldataset = []
        fricative_alldataset = []
        formant_alldataset = []
        formant_re_alldataset = []
        pitch_alldataset = []
        pitch_re_alldataset = []
        intensity_alldataset = []
        intensity_re_alldataset = []
        phoneme_re_alldataset = []
        voice_re_alldataset = []
        unvoice_re_alldataset = []
        semivoice_re_alldataset = []
        plosive_re_alldataset = []
        fricative_re_alldataset = []
        spkr_static_alldataset = []
        spkr_re_static_alldataset = []      
        start_ind_alldataset = []
        start_ind_valid_alldataset = []
        start_ind_wave_alldataset = []
        start_ind_wave_valid_alldataset = []
        end_ind_alldataset = []
        end_ind_valid_alldataset = []
        end_ind_wave_alldataset = []
        end_ind_wave_valid_alldataset = []
        start_ind_re_alldataset = []
        start_ind_re_valid_alldataset = []
        start_ind_re_wave_alldataset = []
        start_ind_re_wave_valid_alldataset = []
        end_ind_re_alldataset = []
        end_ind_re_valid_alldataset = []
        end_ind_re_wave_alldataset = []
        end_ind_re_wave_valid_alldataset = []
        word_alldataset = []
        label_alldataset = []
        wave_alldataset = []
        wave_re_alldataset = []
        wave_re_spec_alldataset = []
        wave_re_spec_amp_alldataset = []
        wave_re_denoise_alldataset = []
        wave_re_spec_denoise_alldataset = []
        wave_re_spec_denoise_amp_alldataset = []
        wave_spec_alldataset = []
        noisesample_re_alldataset = []
        noisesample_alldataset = []
        bad_samples_alldataset = []
        baseline_alldataset = []
        mni_coordinate_alldateset = []
        T1_coordinate_alldateset = []
        mni_coordinate_raw_alldateset = []
        T1_coordinate_raw_alldateset = []
        regions_alldataset =[]
        mask_prior_alldataset = []
        dataset_names = []
        ecog_len = []
        unique_labels = []
        gender_alldataset = []
        # self.ORG_WAVE_FS,self.DOWN_ECOG_FS,self.DOWN_WAVE_FS = allsubj_param['Shared']['ORG_WAVE_FS'],\
        #                                         allsubj_param['Shared']['DOWN_ECOG_FS'],\
        #                                         allsubj_param['Shared']['DOWN_WAVE_FS'],\

        # spkrdata = h5py.File(DATA_DIR[0][0]+'TF32_16k.mat','r')
        # spkr = np.asarray(spkrdata['TFlog'])
        # samples_for_statics_ = spkr[statics_samples_spkr[0][0*2]:statics_samples_spkr[0][0*2+1]]
        flag_zscore = False
        for subj in self.ReqSubjDict:
            subj_param = allsubj_param['Subj'][subj]
            Density = subj_param['Density']
            Gender = subj_param['Gender'] if self.Prod else 'Female'
            Crop = train_param["Subj"][subj]['Crop']
            Trim = train_param["Subj"][subj]['Trim']
            datapath = os.path.join(self.rootpath,subj,'data')
            analysispath = os.path.join(self.rootpath,subj,'analysis')
            #print ('data and analysis path', datapath, analysispath)
            ecog_ = []
            ecog_raw_ = []
            ecog_len_=[0]
            start_ind_train_=[]
            end_ind_train_ = []
            end_ind_valid_train_ = []
            start_ind_valid_train_=[]
            start_ind_wave_down_train_ =[]
            end_ind_wave_down_train_ =[]
            start_ind_wave_down_valid_train_ =[]
            end_ind_wave_down_valid_train_ =[]
            start_ind_re_train_=[]
            end_ind_re_train_ = []
            end_ind_re_valid_train_ = []
            start_ind_re_valid_train_=[]
            start_ind_re_wave_down_train_ =[]
            end_ind_re_wave_down_train_ =[]
            start_ind_re_wave_down_valid_train_ =[]
            end_ind_re_wave_down_valid_train_ =[]
            
            start_ind_test_=[]
            end_ind_ = []           
            end_ind_test_=[]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              = []
            end_ind_valid_test_ = []
            start_ind_valid_test_=[]
            start_ind_wave_down_test_ =[]
            end_ind_wave_down_test_ =[]
            start_ind_wave_down_valid_test_ =[]
            end_ind_wave_down_valid_test_ =[]
            start_ind_re_test_=[]
            end_ind_re_test_ = []
            end_ind_re_valid_test_ = []
            start_ind_re_valid_test_=[]
            start_ind_re_wave_down_test_ =[]
            end_ind_re_wave_down_test_ =[]
            start_ind_re_wave_down_valid_test_ =[]
            end_ind_re_wave_down_valid_test_ =[]
            spkr_=[]
            wave_=[]
            wave_denoise_=[]
            wave_spec_=[]
            spkr_re_=[]
            phoneme_=[]
            phoneme_re_=[]
            formant  = []
            formant_re_  = []
            pitch  = []
            pitch_re_  = []
            intensity  = []
            intensity_re_  = []
            wave_re_=[]
            noisesample_re_=[]
            noisesample_=[]
            wave_re_spec_=[]
            wave_re_spec_amp_=[]
            wave_re_denoise_=[]
            wave_re_spec_denoise_=[]
            wave_re_spec_denoise_amp_=[]
            word_train=[]
            labels_train=[]
            word_test=[]
            labels_test=[]
            bad_samples_=np.array([])
            #N_FFT_num = 256 if Gender =='Female' else 512
            N_FFT_num = cfg.MODEL.N_FFT
            print ('N_FFT_num',N_FFT_num,)
            if cfg.DATASET.PROD:
                if self.FAKE_REPROD:
                    self.TestNum_cum = np.append(self.TestNum_cum, np.array(train_param["Subj"][subj]['TestNum_imagine']).sum().astype(np.int32))
                    testnum = train_param["Subj"][subj]['TestNum_imagine']
                    tasks = train_param["Subj"][subj]['Task_imagine']
                else:
                    self.TestNum_cum = np.append(self.TestNum_cum, np.array(train_param["Subj"][subj]['TestNum']).sum().astype(np.int32))
                    testnum = train_param["Subj"][subj]['TestNum']
                    tasks = train_param["Subj"][subj]['Task']
            else:
                self.TestNum_cum = np.append(self.TestNum_cum, np.array(train_param["Subj"][subj]['TestNum_percept']).sum().astype(np.int32))
                testnum = train_param["Subj"][subj]['TestNum_percept']
                tasks = train_param["Subj"][subj]['Task_percept']
            print (tasks, 'tasks')
            for xx,task_to_use in enumerate(tasks):
                self.TestNum = testnum[xx]
            # for file in range(len(DATA_DIR)):
                HD = True if Density == "HD" else False
                datapath_task = os.path.join(datapath,task_to_use)
                if self.low_density and not self.FAKE_LD:
                    analysispath_task = '/home/xc1490/home/projects/ecog/ALAE_1023/data/data/refined_events/{}/{}/'.format(subj, task_to_use)
                    #Xupeng refined 10/05/2022
                    analysispath_task_ = os.path.join(analysispath,task_to_use)
                else:
                    analysispath_task = os.path.join(analysispath,task_to_use)
                    analysispath_task_ = os.path.join(analysispath,task_to_use)
                
                # if REPRODFLAG is None:
                #     self.Prod = True if 'NY' in DATA_DIR[ds][file] and 'share' in DATA_DIR[ds][file] else False
                # else:
                #     self.Prod = REPRODFLAG
                print("load data from: ", datapath_task)
                print ('analysispath_task',analysispath_task)
                # ecog = np.minimum(ecog,data_range_max[ds][file])
                # import pdb; pdb.set_trace()
                ecog = np.minimum(ecog,30)
                print ('*'*20,'ecog shape', ecog.shape)
                event_range = None if "EventRange" not in subj_param.keys() else subj_param["EventRange"]
                # bad_samples = [] if "BadSamples" not in subj_param.keys() else subj_param["BadSamples"]
                #
                if self.low_density  and not self.FAKE_LD:
                    analysis_event = pd.read_hdf(os.path.join(analysispath_task,'Events.h5'),'/events')
                else:
                    try:
                        analysis_event = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']
                    except:
                        #with h5py.File(os.path.join(analysispath_task,'Events.mat'), 'r') as fileh5:
                        #    analysis_event = fileh5['Events']
                        #fileh5 = h5py.File(os.path.join(analysispath_task,'Events.mat'), 'r')
                        #analysis_event = fileh5['Events']
                        analysis_event = mat73.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']
                        analysis_event = { key:[np.array([[[val]] for val in value])] for (key,value) in analysis_event.items()} #this wrap is used to accommodate the previous codes
                    #print (analysis_event.dtype)
                #start_ind_wave = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['onset'][0]
                #print ('start_ind_wave', start_ind_wave)
                if self.low_density and not self.FAKE_LD:
                    start_ind_wave = np.array(analysis_event['onset'])
                    end_ind_wave = np.array(analysis_event['offset'])
                else:
                    start_ind_wave = analysis_event['onset'][0]
                    start_ind_wave = np.array([start_ind_wave[i][0,0] for i in range(start_ind_wave.shape[0])], dtype=np.float)[:event_range]
                    end_ind_wave = analysis_event['offset'][0]
                    end_ind_wave = np.array([end_ind_wave[i][0,0] for i in range(end_ind_wave.shape[0])], dtype=np.float)[:event_range]
                if self.Prod:
                    if self.FAKE_REPROD:
                        start_ind_re_wave = end_ind_wave+128 #0.25s after stim offset
                        end_ind_re_wave = start_ind_re_wave+256 #0.50s after start_ind_re_wave
                    else:
                        if self.low_density and not self.FAKE_LD:
                            start_ind_re_wave = np.array(analysis_event['onset_r'])
                            end_ind_re_wave = np.array(analysis_event['offset_r'])
                        else:
                            try:
                                start_ind_re_wave = analysis_event['onset_r'][0]
                            except:
                                start_ind_re_wave = analysis_event['onset'][0]
                            #print ('start_ind_re_wave', start_ind_re_wave)
                            start_ind_re_wave = np.nan_to_num(np.array([start_ind_re_wave[i][0,0] for i in range(start_ind_re_wave.shape[0])], dtype=np.float)[:event_range])
                            #print ('start_ind_re_wave', start_ind_re_wave)
                            try:
                                end_ind_re_wave = analysis_event['offset_r'][0]
                            except:
                                end_ind_re_wave = analysis_event['offset'][0]
                            end_ind_re_wave = np.nan_to_num(np.array([end_ind_re_wave[i][0,0] for i in range(end_ind_re_wave.shape[0])], dtype=np.float)[:event_range])
                        
                if HD:
                    start_ind = (start_ind_wave*1.0/self.ORG_WAVE_FS*self.DOWN_ECOG_FS).astype(np.int64) # in ECoG sample
                    start_ind_wave_down = (start_ind_wave*1.0/self.ORG_WAVE_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind = (end_ind_wave*1.0/self.ORG_WAVE_FS*self.DOWN_ECOG_FS).astype(np.int64) # in ECoG sample
                    end_ind_wave_down = (end_ind_wave*1.0/self.ORG_WAVE_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_valid = np.delete(start_ind,bad_samples)
                    end_ind_valid = np.delete(end_ind,bad_samples)
                    start_ind_wave_down_valid = np.delete(start_ind_wave_down,bad_samples)
                    end_ind_wave_down_valid = np.delete(end_ind_wave_down,bad_samples)
                    try:
                        bad_samples = allsubj_param['BadSamples'][subj][task_to_use]
                    except:
                        bad_samples = []
                    bad_samples_ = np.concatenate([bad_samples_,np.array(bad_samples)])
                else:
                    start_ind = (start_ind_wave*1.0/self.ORG_ECOG_FS_NY*self.DOWN_ECOG_FS).astype(np.int64)
                    start_ind_wave_down = (start_ind_wave*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind = (end_ind_wave*1.0/self.ORG_ECOG_FS_NY*self.DOWN_ECOG_FS).astype(np.int64)
                    end_ind_wave_down = (end_ind_wave*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    if self.low_density and not self.FAKE_LD:
                        bad_samples_HD = analysis_event['badevent']
                    else:
                        if self.Prod:
                            if self.FAKE_REPROD:
                                bad_samples_HD = analysis_event['badevent'][0]
                            else:
                                try:
                                    bad_samples_HD = analysis_event['badrsp'][0]
                                except:
                                    bad_samples_HD = analysis_event['badevent'][0]
                        else:
                            bad_samples_HD = analysis_event['badevent'][0]
                        bad_samples_HD = np.asarray([bad_samples_HD[i][0,0] for i in range(bad_samples_HD.shape[0])])
                        
                    bad_samples_ = np.concatenate((bad_samples_,bad_samples_HD))
                    if subj=='NY798':
                        bad_samples_HD = np.where(bad_samples_HD!=0)[0]
                    else:
                        bad_samples_HD = np.where(np.logical_or(np.logical_or(bad_samples_HD==1, bad_samples_HD==2) , bad_samples_HD==4))[0]
                    start_ind_valid = np.delete(start_ind,bad_samples_HD)
                    end_ind_valid = np.delete(end_ind,bad_samples_HD)
                    start_ind_wave_down_valid = np.delete(start_ind_wave_down,bad_samples_HD)
                    end_ind_wave_down_valid = np.delete(end_ind_wave_down,bad_samples_HD)
                    if self.Prod:
                        start_ind_re = (start_ind_re_wave*1.0/self.ORG_ECOG_FS_NY*self.DOWN_ECOG_FS).astype(np.int64)
                        start_ind_re_wave_down = (start_ind_re_wave*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                        end_ind_re = (end_ind_re_wave*1.0/self.ORG_ECOG_FS_NY*self.DOWN_ECOG_FS).astype(np.int64)
                        end_ind_re_wave_down = (end_ind_re_wave*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                        start_ind_re_valid = np.delete(start_ind_re,bad_samples_HD)
                        end_ind_re_valid = np.delete(end_ind_re,bad_samples_HD)
                        start_ind_re_wave_down_valid = np.delete(start_ind_re_wave_down,bad_samples_HD)
                        end_ind_re_wave_down_valid = np.delete(end_ind_re_wave_down,bad_samples_HD)

                baseline_ind = np.concatenate([np.arange(start_ind_valid[i]-self.DOWN_ECOG_FS//4,start_ind_valid[i]-self.DOWN_ECOG_FS//20) \
                                            for i in range(len(start_ind_valid))]) #baseline: 1/4 s - 1/20 s before stimulis onset
                baseline_ind_spec = np.concatenate([np.arange((start_ind_valid[i]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS-self.DOWN_TF_FS//4).astype(np.int64),(start_ind_valid[i]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS-self.DOWN_TF_FS//8).astype(np.int64)) \
                                            for i in range(len(start_ind_valid))]) #baseline: 1/4 s - 1/8 s before stimulis onset

                #if self.process_ecog:
                ecog = signal.resample_poly(ecog,self.DOWN_ECOG_FS*10000,30517625,axis=0) if HD else signal.resample_poly(ecog,self.DOWN_ECOG_FS,self.ORG_ECOG_FS_NY,axis=0) # resample to 125 hz
                #import pdb; pdb.set_trace()
                #print (ecog.shape, baseline_ind)
                baseline_ind = baseline_ind[baseline_ind<ecog.shape[0]]
                baseline = ecog[baseline_ind]
                #import pdb; pdb.set_trace()
                baseline_ind_spec_perceptnoise = np.concatenate([np.arange((start_ind_valid[i]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS).astype(np.int64),(start_ind_valid[i]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS+1).astype(np.int64)) \
                                            for i in range(len(start_ind_valid))])
                # baseline_ind_spec_perceptnoise = np.concatenate([np.arange((end_ind_valid[i]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS-2).astype(np.int64),(end_ind_valid[i]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS-1).astype(np.int64)) \
                #                             for i in range(len(end_ind_valid))])
                statics_ecog = baseline.mean(axis=0,keepdims=True)+1E-10, np.sqrt(baseline.var(axis=0, keepdims=True))+1E-10

                ecog = (ecog - statics_ecog[0])/statics_ecog[1]
                # ecog = (ecog - statics_ecog[0])/statics_ecog[0]
                # import pdb; pdb.set_trace()
                if cfg.DATASET.TRIM:
                    ecog = np.minimum(ecog,Trim)
                print ('ecog_len', ecog.shape )
                ecog_len_+= [ecog.shape[0]]
                ecog_+=[ecog]

                TestInd = start_ind_valid.shape[0]-self.TestNum
                start_ind_train_ += [start_ind[:TestInd] + np.cumsum(ecog_len_)[-2]]
                end_ind_train_ += [end_ind[:TestInd] + np.cumsum(ecog_len_)[-2]]
                end_ind_valid_train_ += [end_ind_valid[:TestInd] + np.cumsum(ecog_len_)[-2]]
                start_ind_valid_train = start_ind_valid[:TestInd] + np.cumsum(ecog_len_)[-2]
                start_ind_valid_train_ += [start_ind_valid_train]
                start_ind_wave_down_train = start_ind_wave_down[:TestInd] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                start_ind_wave_down_train_ += [start_ind_wave_down_train]
                end_ind_wave_down_train = end_ind_wave_down[:TestInd] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                end_ind_wave_down_train_ += [end_ind_wave_down_train]
                start_ind_wave_down_valid_train = start_ind_wave_down_valid[:TestInd] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                start_ind_wave_down_valid_train_ += [start_ind_wave_down_valid_train]
                end_ind_wave_down_valid_train = end_ind_wave_down_valid[:TestInd] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                end_ind_wave_down_valid_train_ += [end_ind_wave_down_valid_train]

                start_ind_test_ += [start_ind[TestInd:] + np.cumsum(ecog_len_)[-2]]
                end_ind_test_ += [end_ind[TestInd:] + np.cumsum(ecog_len_)[-2]]
                end_ind_valid_test_ += [end_ind_valid[TestInd:] + np.cumsum(ecog_len_)[-2]]
                start_ind_valid_test = start_ind_valid[TestInd:] + np.cumsum(ecog_len_)[-2]
                start_ind_valid_test_ += [start_ind_valid_test]
                start_ind_wave_down_test = start_ind_wave_down[TestInd:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                start_ind_wave_down_test_ += [start_ind_wave_down_test]
                end_ind_wave_down_test = end_ind_wave_down[TestInd:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                end_ind_wave_down_test_ += [end_ind_wave_down_test]
                start_ind_wave_down_valid_test = start_ind_wave_down_valid[TestInd:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                start_ind_wave_down_valid_test_ += [start_ind_wave_down_valid_test]
                end_ind_wave_down_valid_test = end_ind_wave_down_valid[TestInd:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                end_ind_wave_down_valid_test_ += [end_ind_wave_down_valid_test]

                if self.Prod:
                    start_ind_re_train_ += [start_ind_re[:TestInd] + np.cumsum(ecog_len_)[-2]]
                    end_ind_re_train_ += [end_ind_re[:TestInd] + np.cumsum(ecog_len_)[-2]]
                    end_ind_re_valid_train_ += [end_ind_re_valid[:TestInd] + np.cumsum(ecog_len_)[-2]]
                    start_ind_re_validtrain_ = start_ind_re_valid[:TestInd] + np.cumsum(ecog_len_)[-2]
                    start_ind_re_valid_train_ += [start_ind_re_validtrain_]
                    start_ind_re_wave_downtrain_ = start_ind_re_wave_down[:TestInd] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_re_wave_down_train_ += [start_ind_re_wave_downtrain_]
                    end_ind_re_wave_downtrain_ = end_ind_re_wave_down[:TestInd] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind_re_wave_down_train_ += [end_ind_re_wave_downtrain_]
                    start_ind_re_wave_down_validtrain_ = start_ind_re_wave_down_valid[:TestInd] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_re_wave_down_valid_train_ += [start_ind_re_wave_down_validtrain_]
                    end_ind_re_wave_down_validtrain_ = end_ind_re_wave_down_valid[:TestInd] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind_re_wave_down_valid_train_ += [end_ind_re_wave_down_validtrain_]

                    start_ind_re_test_ += [start_ind_re[TestInd:] + np.cumsum(ecog_len_)[-2]]
                    end_ind_re_test_ += [end_ind_re[TestInd:] + np.cumsum(ecog_len_)[-2]]
                    end_ind_re_valid_test_ += [end_ind_re_valid[TestInd:] + np.cumsum(ecog_len_)[-2]]
                    start_ind_re_validtest_ = start_ind_re_valid[TestInd:] + np.cumsum(ecog_len_)[-2]
                    start_ind_re_valid_test_ += [start_ind_re_validtest_]
                    start_ind_re_wave_downtest_ = start_ind_re_wave_down[TestInd:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_re_wave_down_test_ += [start_ind_re_wave_downtest_]
                    end_ind_re_wave_downtest_ = end_ind_re_wave_down[TestInd:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind_re_wave_down_test_ += [end_ind_re_wave_downtest_]
                    start_ind_re_wave_down_validtest_ = start_ind_re_wave_down_valid[TestInd:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_re_wave_down_valid_test_ += [start_ind_re_wave_down_validtest_]
                    end_ind_re_wave_down_validtest_ = end_ind_re_wave_down_valid[TestInd:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind_re_wave_down_valid_test_ += [end_ind_re_wave_down_validtest_]
                if not self.Prod:
                    # spkrdata = h5py.File(os.path.join(datapath_task,'TF32_16k.mat'),'r')
                    # spkr = np.asarray(spkrdata['TFlog'])
                    # spkr = signal.resample(spkr,int(1.0*spkr.shape[0]/self.ORG_TF_FS*self.DOWN_TF_FS),axis=0)
                    spkr = np.zeros([end_ind[-1],self.SpecBands])#np.zeros(ecog.shape[0],32)
                    if self.phoneme:
                        phoneme = np.load('/scratch/akg404/SharedECoGData/FunctionalMapping/manually_refine_onsetlabel/manually_refined_perception_new/manually_refined_phoneme_indicator_'+subj+'_'+task_to_use+'_stim.npy')
                    #TODO: formant, pitch, intensity for stimulus
                else:
                    spkr = np.zeros([end_ind[-1],self.SpecBands])
                    if self.phoneme:
                        phoneme = np.zeros([end_ind[-1]])

                samples_for_statics = spkr[start_ind[0]:start_ind[-1]]
                # if HD:
                #     samples_for_statics = samples_for_statics_
                # if not HD:
                #     samples_for_statics = spkr[start_ind[0]:start_ind[-1]]
                if xx==0:
                    statics_spkr = samples_for_statics.mean(axis=0,keepdims=True)+1E-10, np.sqrt(samples_for_statics.var(axis=0, keepdims=True))+1E-10
                # print(statics_spkr)
                if self.Wipenoise:
                    for samples in range(start_ind.shape[0]):
                        if not np.isnan(start_ind[samples]):
                            if samples ==0:
                                spkr[:start_ind[samples]] = 0
                            else:
                                spkr[end_ind[samples-1]:start_ind[samples]] = 0
                            if samples ==start_ind.shape[0]-1:
                                spkr[end_ind[samples]:] = 0
                spkr = (np.clip(spkr,0.,70.)-35.)/35.
                # spkr = (spkr - statics_spkr[0])/statics_spkr[1]
                spkr_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS),spkr.shape[1]])
                if spkr.shape[0]>spkr_trim.shape[0]:
                    spkr_trim = spkr[:spkr_trim.shape[0]]
                    spkr = spkr_trim
                else:
                    spkr_trim[:spkr.shape[0]] = spkr
                    spkr = spkr_trim
                    
                if self.phoneme:
                    phoneme_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS)])            
                    if phoneme.shape[0]>phoneme_trim.shape[0]:
                        phoneme_trim = phoneme[:phoneme_trim.shape[0]]
                        phoneme = phoneme_trim
                    else:
                        phoneme_trim[:phoneme.shape[0]] = phoneme
                        phoneme = phoneme_trim
                    phoneme_+=[phoneme]
                #TODO formant for stimulus
                    
                
                spkr_+=[spkr]
                if not self.Prod:
                    wavedata = h5py.File(os.path.join(datapath_task,'spkr_16k.mat'),'r')
                    wavearray = np.asarray(wavedata['spkr'])
                    wave_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),wavearray.shape[1]])
                else:
                    wavearray = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),1])
                    wave_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),1])

                if wavearray.shape[0]>wave_trim.shape[0]:
                    wave_trim = wavearray[:wave_trim.shape[0]]
                    wavearray = wave_trim
                else:
                    wave_trim[:wavearray.shape[0]] = wavearray
                    wavearray = wave_trim
                wave_+=[wavearray]
                wave_denoise_+=[wavearray]
                if cfg.MODEL.WAVE_BASED:
                    wave_spec = wave2spec(torch.tensor(wavearray.T),n_fft=N_FFT_num,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB,power=2 if cfg.MODEL.POWER_SYNTH else 1)[0].detach().cpu().numpy()
                    # wave_spec_amp = wave2spec(torch.tensor(wavearray.T),n_fft=N_FFT_num,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB,to_db=False,power=2 if cfg.MODEL.POWER_SYNTH else 1)[0].detach().cpu().numpy()
                    #print (wave_spec.shape, baseline_ind_spec)
                    print ('wave_spec shape', wave_spec.shape)
                    baseline_ind_spec = baseline_ind_spec[baseline_ind_spec<wave_spec.shape[0]]
                    noisesample = wave_spec[baseline_ind_spec]
                    for samples in range(start_ind.shape[0]):
                        if not np.isnan(start_ind[samples]):
                            if samples ==0:
                                wave_spec[:start_ind[samples]] = -1
                            else:
                                wave_spec[end_ind[samples-1]:start_ind[samples]] = -1
                            if samples ==start_ind.shape[0]-1:
                                wave_spec[end_ind[samples]:] = -1
                    wave_spec_ +=[wave_spec]
                else:
                    noisesample = spkr[...,baseline_ind_spec]
                noisesample_ += [noisesample]
                
                if self.Prod:
                    if self.FAKE_REPROD:
                        spkr_re = np.zeros([end_ind_re[-1]+1000,128])
                    else:
                        try:
                            spkr_redata = h5py.File(os.path.join(datapath_task,'TFzoom'+str(self.SpecBands)+'_16k_log10.mat'),'r')
                        except:
                            spkr_redata = h5py.File(os.path.join(datapath_task,'TFzoom'+str(self.SpecBands)+'_16k.mat'),'r')
                        spkr_re = np.asarray(spkr_redata['TFlog'])
                        spkr_re = signal.resample(spkr_re,int(1.0*spkr_re.shape[0]/self.ORG_TF_FS*self.DOWN_TF_FS),axis=0)
                    if self.phoneme:
                        if self.FAKE_REPROD:
                            phoneme_re = np.zeros(wave_spec.shape[0])
                        else:
                            phoneme_re = np.load(os.path.join(datapath_task,'manually_refined_phoneme_indicator_'+subj+'_'+ task_to_use +'.npy'))
                    if self.formant:
                        if self.FAKE_REPROD:
                            formant_re = np.zeros([wave_spec.shape[0], 6])
                        else:
                            formant_re = np.load('/scratch/xc1490/projects/ecog/ALAE_1023/data/data/formant_label/formant_label_praat_{}_{}.npy'.format(subj, task_to_use))  
                    if self.pitch:
                        if self.FAKE_REPROD:
                            pitch_re = np.zeros([wave_spec.shape[0]])
                        else:
                            pitch_re = np.load('/scratch/xc1490/projects/ecog/ALAE_1023/data/data/pitch_label/pitch_label_praat_{}_{}.npy'.format(subj, task_to_use))  
                    if self.intensity:
                        if self.FAKE_REPROD:
                            intensity_re = np.zeros([wave_spec.shape[0]])
                        else:
                            intensity_re = np.load('/scratch/xc1490/projects/ecog/ALAE_1023/data/data/intensity_label/intensity_label_praat_{}_{}.npy'.format(subj, task_to_use))  

                            
                    if HD:
                        samples_for_statics_re = samples_for_statics_re_
                    if not HD:
                        samples_for_statics_re = spkr_re[start_ind_re[0]:start_ind_re[-1]]
                    # samples_for_statics_re = spkr_re[statics_samples_spkr_re[ds][file*2]:statics_samples_spkr_re[ds][file*2+1]]
                    if xx==0:
                        statics_spkr_re = samples_for_statics_re.mean(axis=0,keepdims=True)+1E-10, np.sqrt(samples_for_statics_re.var(axis=0, keepdims=True))+1E-10
                    # print(statics_spkr_re)
                    if self.Wipenoise:
                        if subj is not "NY717" or (task_to_use is not 'VisRead' and task_to_use is not 'PicN'):
                            for samples in range(start_ind_re.shape[0]):
                                if not np.isnan(start_ind_re[samples]):
                                    if samples ==0:
                                        spkr_re[:start_ind_re[samples]] = 0
                                    else:
                                        spkr_re[end_ind_re[samples-1]:start_ind_re[samples]] = 0
                                    if samples ==start_ind_re.shape[0]-1:
                                        spkr_re[end_ind_re[samples]:] = 0
                    spkr_re = (np.clip(spkr_re,0.,70.)-35.)/35.
                    # spkr_re = (np.clip(spkr_re,0.,50.)-25.)/25.
                    # spkr_re = (spkr_re - statics_spkr_re[0])/statics_spkr_re[1]
                    spkr_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS),spkr_re.shape[1]])
                    
                    
                        
                    if spkr_re.shape[0]>spkr_re_trim.shape[0]:
                        spkr_re_trim = spkr_re[:spkr_re_trim.shape[0]]
                        spkr_re = spkr_re_trim
                    else:
                        spkr_re_trim[:spkr_re.shape[0]] = spkr_re
                        spkr_re = spkr_re_trim
                    spkr_re_+=[spkr_re]    
                    
                    if self.phoneme:
                        phoneme_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS)])
                        if phoneme_re.shape[0]>phoneme_re_trim.shape[0]:
                            phoneme_re_trim = phoneme_re[:phoneme_re_trim.shape[0]]
                            phoneme_re = phoneme_re_trim
                        else:
                            phoneme_re_trim[:phoneme_re.shape[0]] = phoneme_re
                            phoneme_re = phoneme_re_trim
                        phoneme_re_+=[phoneme_re]
                    if self.formant:
                        formant_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS), 6])
                        if formant_re.shape[0]>formant_re_trim.shape[0]:
                            formant_re_trim = formant_re[:formant_re_trim.shape[0]]
                            formant_re = formant_re_trim
                        else:
                            formant_re_trim[:formant_re.shape[0]] = formant_re
                            formant_re = formant_re_trim
                        formant_re_+=[formant_re]
                    if self.pitch:
                        pitch_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS)])
                        if pitch_re.shape[0]>pitch_re_trim.shape[0]:
                            pitch_re_trim = pitch_re[:pitch_re_trim.shape[0]]
                            pitch_re = pitch_re_trim
                        else:
                            pitch_re_trim[:pitch_re.shape[0]] = pitch_re
                            pitch_re = pitch_re_trim
                        pitch_re_+=[pitch_re]
                    if self.intensity:
                        intensity_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS)])
                        if intensity_re.shape[0]>intensity_re_trim.shape[0]:
                            intensity_re_trim = intensity_re[:intensity_re_trim.shape[0]]
                            intensity_re = intensity_re_trim
                        else:
                            intensity_re_trim[:intensity_re.shape[0]] = intensity_re
                            intensity_re = intensity_re_trim
                        intensity_re_+=[intensity_re]    
                    
                        
                    

                    
                    if not self.FAKE_REPROD:
                        if self.use_denoise:
                            print ('*'*20, 'use denoise')
                            wave_redata = h5py.File(os.path.join(datapath_task,'zoom_16k_denoise_nr.mat'),'r')
                            wave_rearray = wave_redata['zoom'][:].reshape(-1,1) #NY668 has two waveform channels
                            #wave_rearray = wave_rearray.T
                            wave_redata_denoise = h5py.File(os.path.join(datapath_task,'zoom_16k_denoise_nr.mat'),'r')#h5py.File(os.path.join(datapath_task,'zoom_denoise_16k.mat'),'r')
                            wave_rearray_denoise = np.asarray(wave_redata_denoise['zoom']).reshape(-1,1) #NY668 has two waveform channels
                        
                        else:
                            wave_redata = h5py.File(os.path.join(datapath_task,'zoom_16k.mat'),'r')
                            wave_rearray = wave_redata['zoom'][:].reshape(-1,1) #NY668 has two waveform channels
                            #wave_rearray = wave_rearray.T
                            wave_redata_denoise = h5py.File(os.path.join(datapath_task,'zoom_16k.mat'),'r')#h5py.File(os.path.join(datapath_task,'zoom_denoise_16k.mat'),'r')
                            wave_rearray_denoise = np.asarray(wave_redata_denoise['zoom']).reshape(-1,1) #NY668 has two waveform channels
                        #wave_rearray_denoise = wave_rearray_denoise.T
                        wave_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),wave_rearray.shape[1]])
                        wave_re_trim_denoise = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),wave_rearray_denoise.shape[1]])
                    else:
                        wave_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),1])
                        wave_re_trim_denoise = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),1])
                    if not self.FAKE_REPROD:
                        if wave_rearray.shape[0]>wave_re_trim.shape[0]:
                            wave_re_trim = wave_rearray[:wave_re_trim.shape[0]]
                            wave_rearray = wave_re_trim
                            wave_re_trim_denoise = wave_rearray_denoise[:wave_re_trim_denoise.shape[0]]
                            wave_rearray_denoise = wave_re_trim_denoise
                        else:
                            wave_re_trim[:wave_rearray.shape[0]] = wave_rearray
                            wave_rearray = wave_re_trim
                            wave_re_trim_denoise[:wave_rearray_denoise.shape[0]] = wave_rearray_denoise
                            wave_rearray_denoise = wave_re_trim_denoise
                    else:
                        wave_rearray = wave_re_trim
                        wave_rearray_denoise = wave_re_trim_denoise
                    wave_re_+=[wave_rearray]
                    wave_re_denoise_+=[wave_rearray_denoise]

                    if cfg.MODEL.WAVE_BASED:
                        wave_re_spec = wave2spec(torch.tensor(wave_rearray.T),n_fft=N_FFT_num,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB,power=2 if cfg.MODEL.POWER_SYNTH else 1)[0].detach().cpu().numpy()
                        wave_re_spec_amp = wave2spec(torch.tensor(wave_rearray.T),n_fft=N_FFT_num,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB,to_db=False,power=2 if cfg.MODEL.POWER_SYNTH else 1)[0].detach().cpu().numpy()
                        wave_re_spec_denoise = wave2spec(torch.tensor(wave_rearray_denoise.T),n_fft=N_FFT_num,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB,power=2 if cfg.MODEL.POWER_SYNTH else 1)[0].detach().cpu().numpy()
                        wave_re_spec_denoise_amp = wave2spec(torch.tensor(wave_rearray_denoise.T),n_fft=N_FFT_num,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB,to_db=False,power=2 if cfg.MODEL.POWER_SYNTH else 1)[0].detach().cpu().numpy()
                        noisesample_re = wave_re_spec[baseline_ind_spec]
                        if self.Wipenoise:
                            if subj is not "NY717" or (task_to_use is not 'VisRead' and task_to_use is not 'PicN'):
                                for samples in range(start_ind_re.shape[0]):
                                    if not np.isnan(start_ind_re[samples]):
                                        if samples ==0:
                                            wave_re_spec[:start_ind_re[samples]] = -1
                                            wave_re_spec_amp[:start_ind_re[samples]] = 0
                                            wave_re_spec_denoise[:start_ind_re[samples]] = -1
                                            wave_re_spec_denoise_amp[:start_ind_re[samples]] = -1
                                        else:
                                            wave_re_spec[end_ind_re[samples-1]:start_ind_re[samples]] = -1
                                            wave_re_spec_amp[end_ind_re[samples-1]:start_ind_re[samples]] = 0
                                            wave_re_spec_denoise[end_ind_re[samples-1]:start_ind_re[samples]] = -1
                                            wave_re_spec_denoise_amp[end_ind_re[samples-1]:start_ind_re[samples]] = -1
                                        if samples ==start_ind_re.shape[0]-1:
                                            wave_re_spec[end_ind_re[samples]:] = -1
                                            wave_re_spec_amp[end_ind_re[samples]:] = 0
                                            wave_re_spec_denoise[end_ind_re[samples]:] = -1
                                            wave_re_spec_denoise_amp[end_ind_re[samples]:] = -1
                        wave_re_spec_ +=[wave_re_spec]
                        wave_re_spec_amp_ +=[wave_re_spec_amp]
                        wave_re_spec_denoise_ +=[wave_re_spec_denoise]
                        wave_re_spec_denoise_amp_ +=[wave_re_spec_denoise_amp]
                    else:
                        noisesample_re = spkr_re[...,baseline_ind_spec]
                    noisesample_re_ += [noisesample_re]

                        
                try:
                    if HD:
                        label_mat = analysis_event['word'][0][:event_range]
                    else:
                        label_mat = np.array(analysis_event['correctrsp']) #analysis_event['correctrsp'][0][:event_range]
                except:
                    #some early samples do not have this column
                    label_mat = [np.repeat('word', start_ind_wave.shape[0]).reshape(-1,1)[:event_range]]

                label_subset = []
                #import pdb;pdb.set_trace()
                #print (label_mat,label_mat.shape, bad_samples_HD)
                label_mat = np.delete(label_mat,bad_samples_HD)
                #import pdb;pdb.set_trace()
                for i in range(label_mat.shape[0]):
                    if HD:
                        label_mati = label_mat[i][0]
                    else:
                        #try:
                        if self.low_density and not self.FAKE_LD:
                            label_mati = label_mat[i].lower()
                        #except:
                        else:
                            label_mati = label_mat[i][0][0][0].lower().replace('.wav','')
                            #labels.append(str(label_mati).replace('.wav',''))
                    label_subset.append(label_mati)
                    if label_mati not in unique_labels:
                        unique_labels.append(label_mati)
                label_ind = np.zeros([label_mat.shape[0]])
                for i in range(label_mat.shape[0]):
                    label_ind[i] = unique_labels.index(label_subset[i])
                label_ind = np.asarray(label_ind,dtype=np.int16)
                word_train+=[label_ind[:TestInd]]
                labels_train+=[label_subset[:TestInd]]
                word_test+=[label_ind[TestInd:]]
                labels_test+=[label_subset[TestInd:]]

            ######ECoG part ##################
            ################ clean ##################
            if not HD:
                # bad_samples_ = np.where(bad_samples_==1)[0]
                if subj=='NY798':
                    bad_samples_ = np.where(bad_samples_!=0)[0]
                else:
                    bad_samples_ = np.where(np.logical_or(np.logical_or(bad_samples_==1, bad_samples_==2) , bad_samples_==4))[0]
            if HD:
                bad_channels = np.array([]) if "BadElec" not in subj_param.keys() else subj_param["BadElec"]
            else:
                if HD:
                    bad_channels = np.array([]) if "BadElec" not in subj_param.keys() else subj_param["BadElec"]
                else:
                    try:
                        bad_elecs = scipy.io.loadmat(os.path.join(analysispath_task_,'subj_globals.mat'))['bad_elecs']
                        if bad_elecs.shape[0] != 0:
                            bad_channels = bad_elecs[0]-1
                        else:
                            bad_channels = np.array([])
                    except:
                        bad_elecs =  mat73.loadmat(os.path.join(analysispath_task_,'subj_globals.mat'))['bad_elecs'].astype('int')-1
                        if bad_elecs.shape[0] != 0:
                            bad_channels = bad_elecs-1
                        else:
                            bad_channels = np.array([])
            # dataset_name = [name for name in DATA_DIR[ds][0].split('/') if 'NY' in name or 'HD' in name]
            if HD:
                mni_coord = np.array([])
                T1_coord = np.array([])
            else:
                csvfile = os.path.join(analysispath,'coordinates.csv')
                coord = pandas.read_csv(csvfile)
                nonzerorownum = (coord.iloc[:,1:].notna().sum(axis = 1)!=0).sum()
                if self.low_density:
                    if self.extend_grid:
                        row_nums = 100
                    else:
                        row_nums = 64
                else:
                    row_nums = 128 #64 for LD, 128 for HB
                
                row_nums = min(row_nums, nonzerorownum) #cause there might be some all None rows
                mni_coord = np.stack([np.array(coord['MNI_x'][:row_nums]),np.array(coord['MNI_y'][:row_nums]),np.array(coord['MNI_z'][:row_nums])],axis=1)
                # mni_coord = rearrange(mni_coord,Crop,mode = 'coord')
                mni_coord = mni_coord.astype(np.float32)
                mni_coord_raw = mni_coord
                mni_coord = (mni_coord-np.array([-74.,-23.,-20.]))*2/np.array([74.,46.,54.])-1
                T1_coord = np.stack([np.array(coord['T1_x'][:row_nums]),np.array(coord['T1_y'][:row_nums]),np.array(coord['T1_z'][:row_nums])],axis=1)
                # T1_coord = rearrange(T1_coord,NY_crop[ds],mode = 'coord')
                T1_coord = T1_coord.astype(np.float32)
                T1_coord_raw = T1_coord
                T1_coord = (T1_coord-np.array([-74.,-23.,-20.]))*2/np.array([74.,46.,54.])-1
                # for i in range(mni_coord.shape[0]):
                #     print(i,' ',mni_coord[i])
                percent1 = np.array([float(coord['AR_Percentage'][i].strip("%").strip())/100.0 for i in range(row_nums)])
                percent2 = np.array([0.0 if isinstance(coord['AR_7'][i],float) else float(coord['AR_7'][i].strip("%").strip())/100.0 for i in range(row_nums)])
                percent = np.stack([percent1,percent2],1)
                AR1 = np.array([coord['T1_AnatomicalRegion'][i] for i in range(row_nums)])
                AR2 = np.array([coord['AR_8'][i] for i in range(row_nums)])
                AR = np.stack([AR1,AR2],1)
                regions = np.array([AR[i,np.argmax(percent,1)[i]] for i in range(AR.shape[0])])
                regions[np.logical_and(regions=='precentral', mni_coord_raw[:,-1]>=40)] = 'superiorprecentral'
                regions[np.logical_and(regions=='precentral', mni_coord_raw[:,-1]<40)] = 'inferiorprecentral'
                mask = np.ones(ecog_[0].shape[1])
                if bad_channels.shape[0] != 0:
                    mask[bad_channels] = 0.
                if not self.UseGridOnly:
                    lastchannel = ecog_[0].shape[1] 
                else:
                    lastchannel =row_nums
                print ('lastchannel',lastchannel)
                #import pdb; pdb.set_trace()
                if self.process_ecog:
                    if self.ReshapeAsGrid:
                        #print ('rearrange ecog')
                        #print ('regions.shape, mask.shape, mni_coord.shape,mni_coord_raw.shape,T1_coord.shape,T1_coord_raw.shape',\
                              #regions.shape, mask.shape, mni_coord.shape,mni_coord_raw.shape,T1_coord.shape,T1_coord_raw.shape)
                        if self.low_density:
                            regions = self.rearrange_LD(regions,Crop,mode = 'region',nums = row_nums)
                            mask = self.rearrange_LD(mask,Crop,mode = 'mask',nums = row_nums)
                            mni_coord = self.rearrange_LD(mni_coord,Crop,mode = 'coord',nums = row_nums)
                            mni_coord_raw = self.rearrange_LD(mni_coord_raw,Crop,mode = 'coord',nums = row_nums)
                            T1_coord = self.rearrange_LD(T1_coord,Crop,mode = 'coord',nums = row_nums)
                            T1_coord_raw = self.rearrange_LD(T1_coord_raw,Crop,mode = 'coord',nums = row_nums)
                        else:
                            regions = self.rearrange(regions,Crop,mode = 'region')
                            mask = self.rearrange(mask,Crop,mode = 'mask')
                            mni_coord = self.rearrange(mni_coord,Crop,mode = 'coord')
                            mni_coord_raw = self.rearrange(mni_coord_raw,Crop,mode = 'coord')
                            T1_coord = self.rearrange(T1_coord,Crop,mode = 'coord')
                            T1_coord_raw = self.rearrange(T1_coord_raw,Crop,mode = 'coord')
                    else:
                        mask = mask if HD else mask[:lastchannel]
                        regions = regions if HD else regions[:lastchannel]
                        mni_coord = mni_coord if HD else mni_coord[:lastchannel]
                        mni_coord_raw = mni_coord_raw if HD else mni_coord_raw[:lastchannel]
                        mni_coord = mni_coord if HD else mni_coord[:lastchannel]
                        T1_coord_raw = T1_coord_raw if HD else T1_coord_raw[:lastchannel]

            
            # elec_dict = {'mask':mask.reshape([15,15]), 'mni':mni_coord_raw.reshape([15,15,3]).transpose([2,0,1]), 'T1':T1_coord_raw.reshape([15,15,3]).transpose([2,0,1]), 'regions':regions.reshape([15,15])}
            # np.save(os.path.join('NY829_elec_entiregrid.npy'),elec_dict)
            # savemat(os.path.join('NY829_elec_entiregrid.mat'),elec_dict)
            # import pdb; pdb.set_trace();
        
            ecog_ = ecog_ if HD else [ecog_[i][:,:lastchannel] for i in range(len(ecog_))]
            ecog_ = np.concatenate(ecog_,axis=0)
            
            # start_ind_valid_ = np.concatenate(start_ind_valid_,axis=0)
            ecog_raw = ecog_
            if bad_channels.shape[0] != 0:
                ecog_raw[:,bad_channels[bad_channels<lastchannel]]=0
            #import pdb; pdb.set_trace();
            bad_channels = bad_channels[bad_channels<lastchannel]
            if HD:
                ecog_,statics_ecog_zscore = self.zscore(ecog_,badelec = bad_channels)
            elif not flag_zscore:
                ecog_,statics_ecog_zscore = self.zscore(ecog_,badelec = bad_channels)
                flag_zscore = True
            else:
                ecog_ = (ecog_-statics_ecog_zscore[0])/statics_ecog_zscore[1]
            if bad_channels.size !=0: # if bad_channels is not empty
                ecog_[:,bad_channels[bad_channels<lastchannel]]=0

            if not HD:
                # ecog_ = ecog_ # graph
                if self.process_ecog:
                    if self.ReshapeAsGrid:
                        #print ('rearrange ecog', ecog_.shape, ecog_raw.shape)
                        if self.low_density:
                            #import pdb; pdb.set_trace()
                            ecog_ = self.rearrange_LD(ecog_,Crop,mode = 'ecog',nums = row_nums) #conv
                            ecog_raw = self.rearrange_LD(ecog_raw,Crop,mode = 'ecog',nums = row_nums)
                            
                        else:
                            ecog_ = self.rearrange(ecog_,Crop,mode = 'ecog') #conv
                            ecog_raw = self.rearrange(ecog_raw,Crop,mode = 'ecog')
                    ecog_, regions, mask, mni_coord,mni_coord_raw,T1_coord,T1_coord_raw = self.select_block(ecog_,regions,mask,mni_coord,mni_coord_raw,T1_coord,T1_coord_raw,self.SelectRegion,self.BlockRegion)
            #print ('regions, ',regions.shape)
            mni_coordinate_alldateset += [mni_coord]
            T1_coordinate_alldateset += [T1_coord]
            mni_coordinate_raw_alldateset += [mni_coord_raw]
            T1_coordinate_raw_alldateset += [T1_coord_raw]
            regions_alldataset += [regions]
            mask_prior_alldataset += [mask]
            print ('ecog_, subj',ecog_.shape, subj)
            if self.process_ecog:
                if subj =='NY742':
                    #print ('process NY742 ecog, rotate')
                    if self.FAKE_LD:
                        ecog_ = np.rot90(ecog_.reshape(-1,8,8),k=3,axes=(1,2)).reshape(-1,64)
                    else:
                        ecog_ = np.rot90(ecog_.reshape(-1,15,15),k=3,axes=(1,2)).reshape(-1,225)
            ecog_alldataset+= [ecog_]
            ecog_raw_alldataset+= [ecog_raw]
            baseline_alldataset+=[(-statics_ecog_zscore[0]/statics_ecog_zscore[1]).reshape([1])]
            #import pdb;pdb.set_trace()
            
            
            #print ('spkr', spkr.shape, spkr_.shape)
            spkr_alldataset +=[np.concatenate(spkr_,axis=0)]
            if self.phoneme:
                phoneme_alldataset +=[np.concatenate(phoneme_,axis=0)]
            #TODO formant for stimulus
            
            if self.wavebased:
                wave_spec_alldataset +=[np.concatenate(wave_spec_,axis=0)]
            wave_alldataset +=[np.concatenate(wave_,axis=0)]
            start_ind_alldataset += [np.concatenate([np.concatenate(start_ind_train_,axis=0),np.concatenate(start_ind_test_,axis=0)])]
            start_ind_valid_alldataset += [np.concatenate([np.concatenate(start_ind_valid_train_,axis=0),np.concatenate(start_ind_valid_test_,axis=0)])]
            start_ind_wave_alldataset += [np.concatenate([np.concatenate(start_ind_wave_down_train_,axis=0),np.concatenate(start_ind_wave_down_test_,axis=0)])]
            start_ind_wave_valid_alldataset += [np.concatenate([np.concatenate(start_ind_wave_down_valid_train_,axis=0),np.concatenate(start_ind_wave_down_valid_test_,axis=0)])]
            end_ind_alldataset += [np.concatenate([np.concatenate(end_ind_train_,axis=0),np.concatenate(end_ind_test_,axis=0)])]
            end_ind_valid_alldataset += [np.concatenate([np.concatenate(end_ind_valid_train_,axis=0),np.concatenate(end_ind_valid_test_,axis=0)])]
            end_ind_wave_alldataset += [np.concatenate([np.concatenate(end_ind_wave_down_train_,axis=0),np.concatenate(end_ind_wave_down_test_,axis=0)])]
            end_ind_wave_valid_alldataset += [np.concatenate([np.concatenate(end_ind_wave_down_valid_train_,axis=0),np.concatenate(end_ind_wave_down_valid_test_,axis=0)])]
            spkr_static_alldataset +=[statics_spkr]
            noise = 10**(((np.concatenate(noisesample_,axis=0).mean(0)+1)/2*(cfg.MODEL.MAX_DB-cfg.MODEL.NOISE_DB)+cfg.MODEL.NOISE_DB)/10)
            noisesample_alldataset +=[noise]
            if self.Prod:
                spkr_re_alldataset +=[np.concatenate(spkr_re_,axis=0)]
                if self.phoneme:
                    phoneme_re_alldataset +=[np.concatenate(phoneme_re_,axis=0)]
                if self.formant:
                    formant_re_alldataset +=[np.concatenate(formant_re_,axis=0)]
                if self.pitch:
                    pitch_re_alldataset +=[np.concatenate(pitch_re_,axis=0)]
                if self.intensity:
                    intensity_re_alldataset +=[np.concatenate(intensity_re_,axis=0)]
                if self.wavebased:
                    wave_re_spec_alldataset +=[np.concatenate(wave_re_spec_,axis=0)]
                    wave_re_spec_amp_alldataset +=[np.concatenate(wave_re_spec_amp_,axis=0)]
                    wave_re_spec_denoise_alldataset +=[np.concatenate(wave_re_spec_denoise_,axis=0)]
                    wave_re_spec_denoise_amp_alldataset +=[np.concatenate(wave_re_spec_denoise_amp_,axis=0)]
                noise = 10**(((np.concatenate(noisesample_re_,axis=0).mean(0)+1)/2*(cfg.MODEL.MAX_DB-cfg.MODEL.NOISE_DB)+cfg.MODEL.NOISE_DB)/10)
                noisesample_re_alldataset +=[noise]
                #print ('wave_re_',)
                #for tmp in  wave_re_ :
                #    print (tmp.shape)
                wave_re_alldataset +=[np.concatenate(wave_re_,axis=0)]
                wave_re_denoise_alldataset +=[np.concatenate(wave_re_denoise_,axis=0)]
                start_ind_re_alldataset += [np.concatenate([np.concatenate(start_ind_re_train_,axis=0),np.concatenate(start_ind_re_test_,axis=0)])]
                start_ind_re_valid_alldataset += [np.concatenate([np.concatenate(start_ind_re_valid_train_,axis=0),np.concatenate(start_ind_re_valid_test_,axis=0)])]
                start_ind_re_wave_alldataset += [np.concatenate([np.concatenate(start_ind_re_wave_down_train_,axis=0),np.concatenate(start_ind_re_wave_down_test_,axis=0)])]
                start_ind_re_wave_valid_alldataset += [np.concatenate([np.concatenate(start_ind_re_wave_down_valid_train_,axis=0),np.concatenate(start_ind_re_wave_down_valid_test_,axis=0)])]
                end_ind_re_alldataset += [np.concatenate([np.concatenate(end_ind_re_train_,axis=0),np.concatenate(end_ind_re_test_,axis=0)])]
                end_ind_re_valid_alldataset += [np.concatenate([np.concatenate(end_ind_re_valid_train_,axis=0),np.concatenate(end_ind_re_valid_test_,axis=0)])]
                end_ind_re_wave_alldataset += [np.concatenate([np.concatenate(end_ind_re_wave_down_train_,axis=0),np.concatenate(end_ind_re_wave_down_test_,axis=0)])]
                end_ind_re_wave_valid_alldataset += [np.concatenate([np.concatenate(end_ind_re_wave_down_valid_train_,axis=0),np.concatenate(end_ind_re_wave_down_valid_test_,axis=0)])]
                spkr_re_static_alldataset +=[statics_spkr_re]
                if self.phoneme:
                    voice_re_alldataset += [np.isin(np.concatenate(phoneme_re_,axis=0),voice)]
                    unvoice_re_alldataset += [np.isin(np.concatenate(phoneme_re_,axis=0),unvoice)]
                    semivoice_re_alldataset += [np.isin(np.concatenate(phoneme_re_,axis=0),semivoice)]
                    plosive_re_alldataset += [np.isin(np.concatenate(phoneme_re_,axis=0),plosive)]
                    fricative_re_alldataset += [np.isin(np.concatenate(phoneme_re_,axis=0),fricative)]
            bad_samples_alldataset += [bad_samples_]
            word_alldataset += [np.concatenate([np.concatenate(word_train,axis=0),np.concatenate(word_test,axis=0)])]
            # word_alldataset += [np.concatenate(word_,axis=0)]
            dataset_names += [subj]
            
            # import pdb; pdb.set_trace();
            label_alldataset+=[np.concatenate([np.concatenate(labels_train,axis=0),np.concatenate(labels_test,axis=0)])]
            gender_alldataset +=[Gender]
            if self.phoneme:    
                voice_alldataset += [np.isin(np.concatenate(phoneme_,axis=0),voice)]
                unvoice_alldataset += [np.isin(np.concatenate(phoneme_,axis=0),unvoice)]
                semivoice_alldataset += [np.isin(np.concatenate(phoneme_,axis=0),semivoice)]
                plosive_alldataset += [np.isin(np.concatenate(phoneme_,axis=0),plosive)]
                fricative_alldataset += [np.isin(np.concatenate(phoneme_,axis=0),fricative)]
            
        if self.rearrange_elec:
            #for mask regional based attention
            map_index_dict = np.load('/scratch/xc1490/projects/ecog/ALAE_1023/data/region_index_rearrange_mapping_dict.npy',allow_pickle=1).item()
            ecog_alldataset_new = []
            ecog_raw_alldataset_new = []
            mni_coordinate_alldateset_new = []
            mni_coordinate_raw_alldateset_new = []
            for sample in range(len(ecog_alldataset)):
                mni_coordinate_alldateset_new += [np.zeros([  256,3])]
                mni_coordinate_alldateset_new[sample][ map_index_dict[sample][:,0]] = mni_coordinate_alldateset[sample][ map_index_dict[sample][:,1]]
                mni_coordinate_raw_alldateset_new += [np.zeros([ 256,3])]
                mni_coordinate_raw_alldateset_new[sample][ map_index_dict[sample][:,0]] = mni_coordinate_raw_alldateset[sample][ map_index_dict[sample][:,1]]
                ecog_alldataset_new += [np.zeros([ecog_alldataset[sample].shape[0], 256])]
                ecog_alldataset_new[sample][:,map_index_dict[sample][:,0]] = ecog_alldataset[sample][:,map_index_dict[sample][:,1]]
                ecog_raw_alldataset_new += [np.zeros([ecog_raw_alldataset[sample].shape[0], 256])]
                ecog_raw_alldataset_new[sample][:,map_index_dict[sample][:,0]] = ecog_raw_alldataset[sample][:,map_index_dict[sample][:,1]]
                
            mni_coordinate_alldateset , mni_coordinate_raw_alldateset,ecog_alldataset, ecog_raw_alldataset =mni_coordinate_alldateset_new, mni_coordinate_raw_alldateset_new,ecog_alldataset_new, ecog_raw_alldataset_new
        #print ('ecog_alldataset',ecog_alldataset.shape)
        #
        from scipy.signal import savgol_filter
        ecog_alldataset = [savgol_filter(i.T,11,3).T for i in ecog_alldataset]
        ecog_raw_alldataset = [savgol_filter(i.T,11,3).T for i in ecog_raw_alldataset]
        #import pdb; pdb.set_trace();
        if self.low_density:
                        self.meta_data = {'ecog_alldataset':ecog_alldataset,
                    'ecog_raw_alldataset':ecog_raw_alldataset,
                    'spkr_alldataset':spkr_alldataset,
                    'wave_alldataset':wave_alldataset,
                    'wave_spec_alldataset':wave_spec_alldataset,
                    'wave_spec_amp_alldataset':wave_spec_alldataset,
                    'wave_denoise_alldataset':wave_alldataset,
                    'wave_spec_denoise_alldataset':wave_spec_alldataset,
                    'wave_spec_denoise_amp_alldataset':wave_spec_alldataset,
                    'start_ind_alldataset':start_ind_alldataset,
                    'start_ind_wave_alldataset': start_ind_wave_alldataset,
                    'start_ind_valid_alldataset':start_ind_valid_alldataset,
                    'start_ind_wave_valid_alldataset': start_ind_re_wave_valid_alldataset,

                    'spkr_re_alldataset':spkr_re_alldataset,
                    'voice_re_alldataset':voice_re_alldataset,
                    'unvoice_re_alldataset':unvoice_re_alldataset,
                    'semivoice_re_alldataset':semivoice_re_alldataset,
                    'plosive_re_alldataset':plosive_re_alldataset,
                    'fricative_re_alldataset':fricative_re_alldataset,
                    'wave_re_alldataset':wave_re_alldataset,
                    'wave_re_spec_alldataset':wave_re_spec_alldataset,
                    'wave_re_spec_amp_alldataset':wave_re_spec_amp_alldataset,
                    'wave_re_denoise_alldataset':wave_re_denoise_alldataset,
                    'wave_re_spec_denoise_alldataset':wave_re_spec_denoise_alldataset,
                    'wave_re_spec_denoise_amp_alldataset':wave_re_spec_denoise_amp_alldataset,
                    'start_ind_re_alldataset':start_ind_re_alldataset,
                    'start_ind_re_wave_alldataset': start_ind_re_wave_alldataset,
                    'start_ind_re_valid_alldataset':start_ind_re_valid_alldataset,
                    'start_ind_re_wave_valid_alldataset': start_ind_re_wave_valid_alldataset,

                    'end_ind_alldataset':end_ind_alldataset,
                    'end_ind_wave_alldataset':end_ind_wave_alldataset,
                    'end_ind_valid_alldataset':end_ind_valid_alldataset,
                    'end_ind_wave_valid_alldataset':end_ind_wave_valid_alldataset,
                    'end_ind_re_alldataset':end_ind_re_alldataset,
                    'end_ind_re_wave_alldataset':end_ind_re_wave_alldataset,
                    'end_ind_re_valid_alldataset':end_ind_re_valid_alldataset,
                    'end_ind_re_wave_valid_alldataset':end_ind_re_wave_valid_alldataset,

                    'bad_samples_alldataset': bad_samples_alldataset,
                    'dataset_names': dataset_names,
                    'baseline_alldataset': baseline_alldataset,
                    'label_alldataset':label_alldataset,
                    'mni_coordinate_alldateset': mni_coordinate_alldateset,
                    'T1_coordinate_alldateset': T1_coordinate_alldateset,
                    'mni_coordinate_raw_alldateset': mni_coordinate_raw_alldateset,
                    'T1_coordinate_raw_alldateset': T1_coordinate_raw_alldateset,
                    'regions_alldataset' : regions_alldataset,
                    'mask_prior_alldataset': mask_prior_alldataset,
                    'spkr_static_alldataset': spkr_static_alldataset,
                    'spkr_re_static_alldataset': spkr_re_static_alldataset,
                    'word_alldataset':word_alldataset,
                    'noisesample_re_alldataset':noisesample_re_alldataset,
                    'noisesample_alldataset':noisesample_alldataset,

                    'gender_alldataset' : gender_alldataset,
                    'formant_re_alldataset':formant_re_alldataset if self.formant else None,
                    'pitch_re_alldataset':pitch_re_alldataset if self.pitch else None,
                    'intensity_re_alldataset':intensity_re_alldataset if self.intensity else None,
                    }
        else:
            self.meta_data = {'ecog_alldataset':ecog_alldataset,
                    'ecog_raw_alldataset':ecog_raw_alldataset,
                    'spkr_alldataset':spkr_alldataset,
                    'phoneme_alldataset':phoneme_alldataset if self.phoneme else None,
                    'voice_alldataset':voice_alldataset,
                    'unvoice_alldataset':unvoice_alldataset,
                    'semivoice_alldataset':semivoice_alldataset,
                    'plosive_alldataset':plosive_alldataset,
                    'fricative_alldataset':fricative_alldataset,
                    'wave_alldataset':wave_alldataset,
                    'wave_spec_alldataset':wave_spec_alldataset,
                    'wave_spec_amp_alldataset':wave_spec_alldataset,
                    'wave_denoise_alldataset':wave_alldataset,
                    'wave_spec_denoise_alldataset':wave_spec_alldataset,
                    'wave_spec_denoise_amp_alldataset':wave_spec_alldataset,
                    'start_ind_alldataset':start_ind_alldataset,
                    'start_ind_wave_alldataset': start_ind_wave_alldataset,
                    'start_ind_valid_alldataset':start_ind_valid_alldataset,
                    'start_ind_wave_valid_alldataset': start_ind_re_wave_valid_alldataset,

                    'spkr_re_alldataset':spkr_re_alldataset,
                    'phoneme_re_alldataset':phoneme_re_alldataset if self.phoneme else None,
                    'voice_re_alldataset':voice_re_alldataset,
                    'unvoice_re_alldataset':unvoice_re_alldataset,
                    'semivoice_re_alldataset':semivoice_re_alldataset,
                    'plosive_re_alldataset':plosive_re_alldataset,
                    'fricative_re_alldataset':fricative_re_alldataset,
                    'wave_re_alldataset':wave_re_alldataset,
                    'wave_re_spec_alldataset':wave_re_spec_alldataset,
                    'wave_re_spec_amp_alldataset':wave_re_spec_amp_alldataset,
                    'wave_re_denoise_alldataset':wave_re_denoise_alldataset,
                    'wave_re_spec_denoise_alldataset':wave_re_spec_denoise_alldataset,
                    'wave_re_spec_denoise_amp_alldataset':wave_re_spec_denoise_amp_alldataset,
                    'start_ind_re_alldataset':start_ind_re_alldataset,
                    'start_ind_re_wave_alldataset': start_ind_re_wave_alldataset,
                    'start_ind_re_valid_alldataset':start_ind_re_valid_alldataset,
                    'start_ind_re_wave_valid_alldataset': start_ind_re_wave_valid_alldataset,

                    'end_ind_alldataset':end_ind_alldataset,
                    'end_ind_wave_alldataset':end_ind_wave_alldataset,
                    'end_ind_valid_alldataset':end_ind_valid_alldataset,
                    'end_ind_wave_valid_alldataset':end_ind_wave_valid_alldataset,
                    'end_ind_re_alldataset':end_ind_re_alldataset,
                    'end_ind_re_wave_alldataset':end_ind_re_wave_alldataset,
                    'end_ind_re_valid_alldataset':end_ind_re_valid_alldataset,
                    'end_ind_re_wave_valid_alldataset':end_ind_re_wave_valid_alldataset,

                    'bad_samples_alldataset': bad_samples_alldataset,
                    'dataset_names': dataset_names,
                    'baseline_alldataset': baseline_alldataset,
                    'label_alldataset':label_alldataset,
                    'mni_coordinate_alldateset': mni_coordinate_alldateset,
                    'T1_coordinate_alldateset': T1_coordinate_alldateset,
                    'mni_coordinate_raw_alldateset': mni_coordinate_raw_alldateset,
                    'T1_coordinate_raw_alldateset': T1_coordinate_raw_alldateset,
                    'regions_alldataset' : regions_alldataset,
                    'mask_prior_alldataset': mask_prior_alldataset,
                    'spkr_static_alldataset': spkr_static_alldataset,
                    'spkr_re_static_alldataset': spkr_re_static_alldataset,
                    'word_alldataset':word_alldataset,
                    'noisesample_re_alldataset':noisesample_re_alldataset,
                    'noisesample_alldataset':noisesample_alldataset,

                    'gender_alldataset' : gender_alldataset,
                    'formant_re_alldataset':formant_re_alldataset if self.formant else None,
                    'pitch_re_alldataset':pitch_re_alldataset if self.pitch else None,
                    'intensity_re_alldataset':intensity_re_alldataset if self.intensity else None,
                    
                    }



    def __len__(self):
        if self.DEBUG:
            repeattimes = 1
        else:
            repeattimes = 128
        if self.mode == 'train':
            if self.Prod:
                return np.array([start_ind_re_alldataset.shape[0]*(repeattimes if 'NY798' in self.ReqSubjDict else repeattimes)//self.world_size for start_ind_re_alldataset in self.meta_data['start_ind_re_alldataset']]).sum()
            else:
                return np.array([start_ind_alldataset.shape[0]*repeattimes for start_ind_alldataset in self.meta_data['start_ind_alldataset']]).sum()
        else:
            return self.TestNum_cum[0]

    def __getitem__(self, idx):
        ecog_alldataset = self.meta_data['ecog_alldataset']
        ecog_raw_alldataset = self.meta_data['ecog_raw_alldataset']
        bad_samples_alldataset = self.meta_data['bad_samples_alldataset']
        dataset_names = self.meta_data['dataset_names']
        label_alldataset = self.meta_data['label_alldataset']
        word_alldataset = self.meta_data['word_alldataset']
        spkr_alldataset = self.meta_data['spkr_alldataset']
        if self.phoneme:
            phoneme_alldataset = self.meta_data['phoneme_alldataset']
            voice_alldataset = self.meta_data['voice_alldataset']
            unvoice_alldataset = self.meta_data['unvoice_alldataset']
            semivoice_alldataset = self.meta_data['semivoice_alldataset']
            plosive_alldataset = self.meta_data['plosive_alldataset']
            fricative_alldataset = self.meta_data['fricative_alldataset']
        start_ind_alldataset = self.meta_data['start_ind_alldataset']
        start_ind_valid_alldataset = self.meta_data['start_ind_valid_alldataset']
        end_ind_valid_alldataset = self.meta_data['end_ind_valid_alldataset']
        start_ind_wave_alldataset = self.meta_data['start_ind_wave_alldataset']
        end_ind_alldataset = self.meta_data['end_ind_alldataset']
        wave_spec_alldataset = self.meta_data['wave_spec_alldataset']
        wave_spec_amp_alldataset = self.meta_data['wave_spec_amp_alldataset']
        wave_alldataset = self.meta_data['wave_alldataset']
        wave_spec_denoise_alldataset = self.meta_data['wave_spec_denoise_alldataset']
        wave_spec_denoise_amp_alldataset = self.meta_data['wave_spec_denoise_amp_alldataset']
        wave_denoise_alldataset = self.meta_data['wave_denoise_alldataset']
        spkr_static_alldataset = self.meta_data['spkr_static_alldataset']
        if self.Prod:
            spkr_re_alldataset = self.meta_data['spkr_re_alldataset']
            if self.phoneme:
                phoneme_re_alldataset = self.meta_data['phoneme_re_alldataset']
                voice_re_alldataset = self.meta_data['voice_re_alldataset']
                unvoice_re_alldataset = self.meta_data['unvoice_re_alldataset']
                semivoice_re_alldataset = self.meta_data['semivoice_re_alldataset']
                plosive_re_alldataset = self.meta_data['plosive_re_alldataset']
                fricative_re_alldataset = self.meta_data['fricative_re_alldataset']
                start_ind_re_alldataset = self.meta_data['start_ind_re_alldataset']
            if self.formant:
                formant_re_alldataset = self.meta_data['formant_re_alldataset']
            if self.pitch:
                pitch_re_alldataset = self.meta_data['pitch_re_alldataset']
            if self.intensity:
                intensity_re_alldataset = self.meta_data['intensity_re_alldataset']        
            start_ind_re_valid_alldataset = self.meta_data['start_ind_re_valid_alldataset']
            end_ind_re_valid_alldataset = self.meta_data['end_ind_re_valid_alldataset']
            start_ind_re_wave_alldataset = self.meta_data['start_ind_re_wave_alldataset']
            end_ind_re_alldataset = self.meta_data['end_ind_re_alldataset']
            wave_spec_re_alldataset = self.meta_data['wave_re_spec_alldataset']
            wave_spec_re_amp_alldataset = self.meta_data['wave_re_spec_amp_alldataset']
            wave_re_alldataset = self.meta_data['wave_re_alldataset']
            wave_spec_re_denoise_alldataset = self.meta_data['wave_re_spec_denoise_alldataset']
            wave_spec_re_denoise_amp_alldataset = self.meta_data['wave_re_spec_denoise_amp_alldataset']
            wave_re_denoise_alldataset = self.meta_data['wave_re_denoise_alldataset']
            spkr_re_static_alldataset = self.meta_data['spkr_re_static_alldataset']
        if not self.Prod:
            n_delay_1 = -16#28 # samples
            n_delay_2 = 0#92#120#92 # samples
        #ifg
        else:
            n_delay_1 = -16#28 # samples
            n_delay_2 = 0#92#120#92 # samples

        num_dataset = len(ecog_alldataset)
        mni_coordinate_all = []
        mni_coordinate_raw_all = []
        T1_coordinate_all = []
        T1_coordinate_raw_all = []
        regions_all =[]
        mask_all = []
        gender_all = []
        ecog_batch_all = []
        ecog_raw_batch_all = []
        spkr_batch_all = []
        phoneme_batch_all = []
        voice_batch_all = []
        unvoice_batch_all = []
        semivoice_batch_all = []
        plosive_batch_all = []
        fricative_batch_all = []
        wave_batch_all = []
        wave_spec_batch_all = []
        wave_spec_amp_batch_all = []
        wave_denoise_batch_all = []
        wave_spec_denoise_batch_all = []
        wave_spec_denoise_amp_batch_all = []
        ecog_re_batch_all = []
        ecog_raw_re_batch_all = []
        spkr_re_batch_all = []
        phoneme_re_batch_all = []
        formant_re_batch_all = []
        intensity_re_batch_all = []
        pitch_re_batch_all = []
        voice_re_batch_all = []
        unvoice_re_batch_all = []
        semivoice_re_batch_all = []
        plosive_re_batch_all = []
        fricative_re_batch_all = []
        wave_re_batch_all = []
        wave_spec_re_batch_all = []
        wave_spec_re_amp_batch_all = []
        wave_re_denoise_batch_all = []
        wave_spec_re_denoise_batch_all = []
        wave_spec_re_denoise_amp_batch_all = []
        label_batch_all = []
        word_batch_all = []
        on_stage_batch_all = []
        on_stage_re_batch_all = []
        on_stage_wider_batch_all = []
        on_stage_wider_re_batch_all = []
        self.SeqLenSpkr = self.SeqLen*int(self.DOWN_TF_FS*1.0/self.DOWN_ECOG_FS)
        pre_articulate_len = self.ahead_onset_test
        imagesize = 2**self.current_lod
        for i in range(num_dataset):
            # bad_samples = bad_samples_alldataset[i]
            if self.mode =='train':
                rand_ind = np.random.choice(np.arange(start_ind_valid_alldataset[i].shape[0])[:-self.TestNum_cum[i]],1,replace=False)[0]
            elif self.mode =='test':
                if self.Prod:
                    rand_ind = idx+start_ind_re_valid_alldataset[i].shape[0]-self.TestNum_cum[i]
                else:
                    rand_ind = idx+start_ind_valid_alldataset[i].shape[0]-self.TestNum_cum[i]
            # label_valid = np.delete(label_alldataset[i],bad_samples_alldataset[i])
            label = [label_alldataset[i][rand_ind]]
            word = word_alldataset[i][rand_ind]
            start_indx = start_ind_valid_alldataset[i][rand_ind]
            end_indx = end_ind_valid_alldataset[i][rand_ind]
            ecog_batch = np.zeros((self.SeqLen+n_delay_2-n_delay_1 ,ecog_alldataset[i].shape[-1]))
            ecog_raw_batch = np.zeros((self.SeqLen+n_delay_2-n_delay_1 ,ecog_alldataset[i].shape[-1]))
            # ecog_batch = np.zeros((self.SeqLen ,ecog_alldataset[i].shape[-1]))
            spkr_batch = np.zeros(( self.SeqLenSpkr,spkr_alldataset[i].shape[-1]))
            phoneme_batch = np.zeros(( self.SeqLenSpkr))
            
            voice_batch = np.zeros(( self.SeqLenSpkr))
            unvoice_batch = np.zeros(( self.SeqLenSpkr))
            semivoice_batch = np.zeros(( self.SeqLenSpkr))
            plosive_batch = np.zeros(( self.SeqLenSpkr))
            fricative_batch = np.zeros(( self.SeqLenSpkr))
            wave_batch = np.zeros(( (self.SeqLen*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)),wave_alldataset[i].shape[-1]))
            if self.wavebased:
                wave_spec_batch = np.zeros(( self.SeqLen, wave_spec_alldataset[i].shape[-1]))
            if self.Prod:
                start_indx_re = start_ind_re_valid_alldataset[i][rand_ind]
                end_indx_re = end_ind_re_valid_alldataset[i][rand_ind]
                if self.pre_articulate:
                    ecog_batch_re = np.zeros((pre_articulate_len ,ecog_alldataset[i].shape[-1]))
                    ecog_raw_batch_re = np.zeros((pre_articulate_len ,ecog_alldataset[i].shape[-1]))
                else:
                    ecog_batch_re = np.zeros((self.SeqLen+n_delay_2-n_delay_1 ,ecog_alldataset[i].shape[-1]))
                    ecog_raw_batch_re = np.zeros((self.SeqLen+n_delay_2-n_delay_1 ,ecog_alldataset[i].shape[-1]))
                # ecog_batch_re = np.zeros((self.SeqLen ,ecog_alldataset[i].shape[-1]))
                spkr_batch_re = np.zeros(( self.SeqLenSpkr,spkr_alldataset[i].shape[-1]))
                phoneme_batch_re = np.zeros(( self.SeqLenSpkr))
                formant_batch_re = np.zeros(( self.SeqLenSpkr, 6))
                pitch_batch_re = np.zeros(( self.SeqLenSpkr))
                intensity_batch_re = np.zeros(( self.SeqLenSpkr))
                voice_batch_re = np.zeros(( self.SeqLenSpkr))
                unvoice_batch_re = np.zeros(( self.SeqLenSpkr))
                semivoice_batch_re = np.zeros(( self.SeqLenSpkr))
                plosive_batch_re = np.zeros(( self.SeqLenSpkr))
                fricative_batch_re = np.zeros(( self.SeqLenSpkr))
                wave_batch_re = np.zeros(( (self.SeqLen*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)),wave_alldataset[i].shape[-1]))
                if self.wavebased:
                    wave_spec_batch_re = np.zeros(( self.SeqLen, wave_spec_alldataset[i].shape[-1]))

            
            if self.mode =='test' or self.pre_articulate:
                indx = start_indx - self.ahead_onset_test
                if self.Prod:
                    indx_re = start_indx_re-self.ahead_onset_test
            elif self.mode =='train':
                # indx = np.maximum(indx+np.random.choice(np.arange(np.minimum(-(self.SeqLenSpkr-(end_indx-indx)),-1),np.maximum(-(self.SeqLenSpkr-(end_indx-indx)),0)),1)[0],0)
                chosen_start = np.random.choice(np.arange(-64,end_indx-start_indx-64),1)[0]
                indx = np.maximum(start_indx+chosen_start,0)
                # indx = indx - self.ahead_onset_test
                if self.Prod:
                    # indx_re = np.maximum(indx_re+np.random.choice(np.arange(np.minimum(-(self.SeqLenSpkr-(end_indx_re-indx_re)),-1),np.maximum(-(self.SeqLenSpkr-(end_indx_re-indx_re)),0)),1)[0],0)
                    chosen_start_re = np.random.choice(np.arange(-64,end_indx_re-start_indx_re-64),1)[0]
                    indx_re = np.maximum(start_indx_re+chosen_start_re,0)
                    # indx_re = indx_re-self.ahead_onset_test
            

            # indx = indx.item()
            ecog_batch = ecog_alldataset[i][indx+n_delay_1:indx+self.SeqLen+n_delay_2]
            ecog_raw_batch = ecog_raw_alldataset[i][indx+n_delay_1:indx+self.SeqLen+n_delay_2]
            # ecog_batch = ecog_alldataset[i][indx+n_delay_1:indx+self.SeqLen+n_delay_1]
            on_stage_batch = np.zeros([1,self.SeqLenSpkr])
            on_stage_batch[:,np.maximum(start_indx-indx,0): np.minimum(end_indx-indx,self.SeqLenSpkr-1)] = 1.0
            on_stage_wider_batch = np.zeros([1,self.SeqLenSpkr])
            on_stage_wider_batch[:,np.maximum(start_indx-indx-5,0): np.minimum(end_indx-indx+5,self.SeqLenSpkr-1)] = 1.0
            spkr_batch = spkr_alldataset[i][indx:indx+self.SeqLenSpkr]
            if self.phoneme:
                phoneme_batch = phoneme_alldataset[i][indx:indx+self.SeqLenSpkr]
                voice_batch = voice_alldataset[i][indx:indx+self.SeqLenSpkr]
                unvoice_batch = unvoice_alldataset[i][indx:indx+self.SeqLenSpkr]
                semivoice_batch = semivoice_alldataset[i][indx:indx+self.SeqLenSpkr]
                plosive_batch = plosive_alldataset[i][indx:indx+self.SeqLenSpkr]
                fricative_batch = fricative_alldataset[i][indx:indx+self.SeqLenSpkr]
            if self.wavebased:
                wave_spec_batch = wave_spec_alldataset[i][indx:indx+self.SeqLen]
                wave_spec_batch_amp = wave_spec_amp_alldataset[i][indx:indx+self.SeqLen]
                wave_spec_batch_denoise = wave_spec_denoise_alldataset[i][indx:indx+self.SeqLen]
                wave_spec_batch_amp_denoise = wave_spec_denoise_amp_alldataset[i][indx:indx+self.SeqLen]
            wave_batch = wave_alldataset[i][(indx*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)):((indx+self.SeqLen)*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS))]
            wave_batch_denoise = wave_denoise_alldataset[i][(indx*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)):((indx+self.SeqLen)*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS))]
            if self.Prod:
                # indx_re = indx_re.item()
                on_stage_re_batch = np.zeros([1,self.SeqLenSpkr])
                on_stage_re_batch[:,np.maximum(start_indx_re-indx_re,0): np.minimum(end_indx_re-indx_re,self.SeqLenSpkr-1)] = 1.0
                on_stage_wider_re_batch = np.zeros([1,self.SeqLenSpkr])
                on_stage_wider_re_batch[:,np.maximum(start_indx_re-indx_re-5,0): np.minimum(end_indx_re-indx_re+5,self.SeqLenSpkr-1)] = 1.0
                if self.pre_articulate:
                    ecog_batch_re = ecog_alldataset[i][indx_re:indx_re+pre_articulate_len]
                    ecog_raw_batch_re = ecog_raw_alldataset[i][indx_re:indx_re+pre_articulate_len]
                else:
                    ecog_batch_re = ecog_alldataset[i][indx_re+n_delay_1:indx_re+self.SeqLen+n_delay_2]
                    ecog_raw_batch_re = ecog_raw_alldataset[i][indx_re+n_delay_1:indx_re+self.SeqLen+n_delay_2]
                    # ecog_batch_re = ecog_alldataset[i][indx_re+n_delay_1:indx_re+self.SeqLen+n_delay_1]
                spkr_batch_re = spkr_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                if self.phoneme:
                    phoneme_batch_re = phoneme_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                    
                    voice_batch_re = voice_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                    unvoice_batch_re = unvoice_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                    semivoice_batch_re = semivoice_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                    plosive_batch_re = plosive_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                    fricative_batch_re = fricative_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                if self.formant:
                    formant_batch_re = formant_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                if self.pitch:
                    pitch_batch_re = pitch_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                if self.intensity:
                    intensity_batch_re = intensity_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                    
                if self.wavebased:
                    wave_spec_batch_re = wave_spec_re_alldataset[i][indx_re:indx_re+self.SeqLen]
                    wave_spec_batch_amp_re = wave_spec_re_amp_alldataset[i][indx_re:indx_re+self.SeqLen]
                    wave_spec_batch_re_denoise = wave_spec_re_denoise_alldataset[i][indx_re:indx_re+self.SeqLen]
                    wave_spec_batch_re_amp_denoise = wave_spec_re_denoise_amp_alldataset[i][indx_re:indx_re+self.SeqLen]
                wave_batch_re = wave_re_alldataset[i][(indx_re*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)):((indx_re+self.SeqLen)*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS))]
                wave_batch_re_denoise = wave_re_denoise_alldataset[i][(indx_re*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)):((indx_re+self.SeqLen)*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS))]
            
            mni_batch = self.meta_data['mni_coordinate_alldateset'][i]
            mni_raw_batch = self.meta_data['mni_coordinate_raw_alldateset'][i]
            T1_batch = self.meta_data['T1_coordinate_alldateset'][i]
            T1_raw_batch = self.meta_data['T1_coordinate_raw_alldateset'][i]
            # ecog_batch = ecog_batch[np.newaxis,:,:]
            # mni_batch = np.transpose(mni_batch,[1,0])
            if not self.BCTS:
                spkr_batch = np.transpose(spkr_batch,[1,0])
                if self.Prod:
                    # ecog_batch_re = ecog_batch_re[np.newaxis,:,:]
                    spkr_batch_re = np.transpose(spkr_batch_re,[1,0])

            #print ('ecog_batch',ecog_batch.shape)
            ecog_batch_all += [ecog_batch]
            ecog_raw_batch_all += [ecog_raw_batch]
            spkr_batch_all += [spkr_batch[np.newaxis,...]]
            if self.phoneme:
                phoneme_batch_all += [phoneme_batch[np.newaxis,...]]
                voice_batch_all += [voice_batch[np.newaxis,...]]
                unvoice_batch_all += [unvoice_batch[np.newaxis,...]]
                semivoice_batch_all += [semivoice_batch[np.newaxis,...]]
                plosive_batch_all += [plosive_batch[np.newaxis,...]]
                fricative_batch_all += [fricative_batch[np.newaxis,...]]
            if self.wavebased:
                wave_spec_batch_all += [wave_spec_batch[np.newaxis,...]]
                wave_spec_amp_batch_all += [wave_spec_batch_amp[np.newaxis,...]]
                wave_spec_denoise_batch_all += [wave_spec_batch_denoise[np.newaxis,...]]
                wave_spec_denoise_amp_batch_all += [wave_spec_batch_amp_denoise[np.newaxis,...]]
            wave_batch_all += [wave_batch.swapaxes(-2,-1)]
            wave_denoise_batch_all += [wave_batch_denoise.swapaxes(-2,-1)]
            on_stage_batch_all += [on_stage_batch]
            on_stage_wider_batch_all += [on_stage_wider_batch]
            if self.Prod:
                ecog_re_batch_all += [ecog_batch_re]
                ecog_raw_re_batch_all += [ecog_raw_batch_re]
                spkr_re_batch_all += [spkr_batch_re[np.newaxis,...]]
                if self.phoneme:
                    phoneme_re_batch_all += [phoneme_batch_re[np.newaxis,...]]
                    
                    voice_re_batch_all += [voice_batch_re[np.newaxis,...]]
                    unvoice_re_batch_all += [unvoice_batch_re[np.newaxis,...]]
                    semivoice_re_batch_all += [semivoice_batch_re[np.newaxis,...]]
                    plosive_re_batch_all += [plosive_batch_re[np.newaxis,...]]
                    fricative_re_batch_all += [fricative_batch_re[np.newaxis,...]]
                if self.formant:
                    formant_re_batch_all += [formant_batch_re[np.newaxis,...]]
                if self.pitch:
                    pitch_re_batch_all += [pitch_batch_re[np.newaxis,...]]
                if self.intensity:
                    intensity_re_batch_all += [intensity_batch_re[np.newaxis,...]]
                    
                if self.wavebased:
                    wave_spec_re_batch_all += [wave_spec_batch_re[np.newaxis,...]]
                    wave_spec_re_amp_batch_all += [wave_spec_batch_amp_re[np.newaxis,...]]
                    wave_spec_re_denoise_batch_all += [wave_spec_batch_re_denoise[np.newaxis,...]]
                    wave_spec_re_denoise_amp_batch_all += [wave_spec_batch_re_amp_denoise[np.newaxis,...]]
                wave_re_batch_all += [wave_batch_re.swapaxes(-2,-1)]
                wave_re_denoise_batch_all += [wave_batch_re_denoise.swapaxes(-2,-1)]
                on_stage_re_batch_all += [on_stage_re_batch]
                on_stage_wider_re_batch_all += [on_stage_wider_re_batch]
            label_batch_all +=[label]
            word_batch_all +=[word]
            mni_coordinate_all +=[mni_batch.swapaxes(-2,-1)]
            mni_coordinate_raw_all +=[mni_raw_batch.swapaxes(-2,-1)]
            T1_coordinate_all +=[T1_batch.swapaxes(-2,-1)]
            T1_coordinate_raw_all +=[T1_raw_batch.swapaxes(-2,-1)]
            # import pdb; pdb.set_trace()
            gender_all +=[np.array([0.],dtype=np.float32) if self.meta_data['gender_alldataset'][i]=='Male' else np.array([1.],dtype=np.float32)]
            regions_all +=[self.meta_data['regions_alldataset'][i]]
            mask_all +=[self.meta_data['mask_prior_alldataset'][i][np.newaxis,...]]

        # # spkr_batch_all = np.concatenate(spkr_batch_all,axis=0)
        # # wave_batch_all = np.concatenate(wave_batch_all,axis=0)
        # # if self.Prod:
        # #     spkr_re_batch_all = np.concatenate(spkr_re_batch_all,axis=0)
        # #     wave_re_batch_all = np.concatenate(wave_re_batch_all,axis=0)
        # label_batch_all = np.concatenate(label_batch_all,axis=0).tolist()
        # word_batch_all = np.array(word_batch_all)
        # # baseline_batch_all = np.concatenate(self.meta_data['baseline_alldataset'],axis=0)
        # # mni_coordinate_all = np.concatenate(mni_coordinate_all,axis=0)
        # regions_all = np.concatenate(regions_all,axis=0).tolist()


        spkr_batch_all = np.concatenate(spkr_batch_all,axis=0)
        if self.phoneme:
            phoneme_batch_all = np.concatenate(phoneme_batch_all,axis=0)
            voice_batch_all = np.concatenate(voice_batch_all,axis=0)
            unvoice_batch_all = np.concatenate(unvoice_batch_all,axis=0)
            semivoice_batch_all = np.concatenate(semivoice_batch_all,axis=0)
            plosive_batch_all = np.concatenate(plosive_batch_all,axis=0)
            fricative_batch_all = np.concatenate(fricative_batch_all,axis=0)
        wave_batch_all = np.concatenate(wave_batch_all,axis=0)
        if self.wavebased:
            wave_spec_batch_all = np.concatenate(wave_spec_batch_all,axis=0)
            wave_spec_amp_batch_all = np.concatenate(wave_spec_amp_batch_all,axis=0)
            wave_spec_denoise_batch_all = np.concatenate(wave_spec_denoise_batch_all,axis=0)
            wave_spec_denoise_amp_batch_all = np.concatenate(wave_spec_denoise_amp_batch_all,axis=0)
        wave_denoise_batch_all = np.concatenate(wave_denoise_batch_all,axis=0)
        on_stage_batch_all = np.concatenate(on_stage_batch_all,axis=0)
        on_stage_wider_batch_all = np.concatenate(on_stage_wider_batch_all,axis=0)
        if self.Prod:
            spkr_re_batch_all = np.concatenate(spkr_re_batch_all,axis=0)
            if self.phoneme:
                phoneme_re_batch_all = np.concatenate(phoneme_re_batch_all,axis=0)
                voice_re_batch_all = np.concatenate(voice_re_batch_all,axis=0)
                unvoice_re_batch_all = np.concatenate(unvoice_re_batch_all,axis=0)
                semivoice_re_batch_all = np.concatenate(semivoice_re_batch_all,axis=0)
                plosive_re_batch_all = np.concatenate(plosive_re_batch_all,axis=0)
                fricative_re_batch_all = np.concatenate(fricative_re_batch_all,axis=0)
            if self.formant:
                formant_re_batch_all = np.concatenate(formant_re_batch_all,axis=0)
            if self.pitch:
                pitch_re_batch_all = np.concatenate(pitch_re_batch_all,axis=0)
            if self.intensity:
                intensity_re_batch_all = np.concatenate(intensity_re_batch_all,axis=0)
                
            if self.wavebased:
                wave_spec_re_batch_all = np.concatenate(wave_spec_re_batch_all,axis=0)
                wave_spec_re_amp_batch_all = np.concatenate(wave_spec_re_amp_batch_all,axis=0)
                wave_spec_re_denoise_batch_all = np.concatenate(wave_spec_re_denoise_batch_all,axis=0)
                wave_spec_re_denoise_amp_batch_all = np.concatenate(wave_spec_re_denoise_amp_batch_all,axis=0)
            wave_re_batch_all = np.concatenate(wave_re_batch_all,axis=0)
            wave_re_denoise_batch_all = np.concatenate(wave_re_denoise_batch_all,axis=0)
            on_stage_re_batch_all = np.concatenate(on_stage_re_batch_all,axis=0)
            on_stage_wider_re_batch_all = np.concatenate(on_stage_wider_re_batch_all,axis=0)
        label_batch_all = np.concatenate(label_batch_all,axis=0).tolist()
        word_batch_all = np.array(word_batch_all)
        baseline_batch_all = np.concatenate(self.meta_data['baseline_alldataset'],axis=0)
        mni_coordinate_all = np.concatenate(mni_coordinate_all,axis=0)
        mni_coordinate_raw_all = np.concatenate(mni_coordinate_raw_all,axis=0)
        T1_coordinate_all = np.concatenate(T1_coordinate_all,axis=0)
        T1_coordinate_raw_all = np.concatenate(T1_coordinate_raw_all,axis=0)
        regions_all = np.concatenate(regions_all,axis=0).tolist()
        gender_all = np.concatenate(gender_all,axis=0)
        
        #if self.mode == 'train':
        #    if self.process_ecog:
        #        #print (ecog_re_batch_all)
        #        for i in range(len(ecog_re_batch_all)):
        #            ecog_re_batch_all[i] = self.transform(ecog_re_batch_all[i])

        if self.low_density:
            return_dict = {'ecog_batch_all':ecog_batch_all,
                'ecog_raw_batch_all':ecog_raw_batch_all,
                'spkr_batch_all':spkr_batch_all,

                'wave_batch_all':wave_batch_all,
                'wave_spec_batch_all':wave_spec_batch_all,
                'wave_spec_amp_batch_all':wave_spec_amp_batch_all,
                'wave_denoise_batch_all':wave_denoise_batch_all,
                'wave_spec_denoise_batch_all':wave_spec_denoise_batch_all,
                'wave_spec_denoise_amp_batch_all':wave_spec_denoise_amp_batch_all,

                'ecog_re_batch_all':ecog_re_batch_all,
                'ecog_raw_re_batch_all':ecog_raw_re_batch_all,
                'spkr_re_batch_all':spkr_re_batch_all,
                'voice_re_batch_all':voice_re_batch_all,
                'unvoice_re_batch_all':unvoice_re_batch_all,
                'semivoice_re_batch_all':semivoice_re_batch_all,
                'plosive_re_batch_all':plosive_re_batch_all,
                'fricative_re_batch_all':fricative_re_batch_all,
                'wave_re_batch_all':wave_re_batch_all,
                'wave_spec_re_batch_all':wave_spec_re_batch_all,
                'wave_spec_re_amp_batch_all':wave_spec_re_amp_batch_all,
                'wave_re_denoise_batch_all':wave_re_denoise_batch_all,
                'wave_spec_re_denoise_batch_all':wave_spec_re_denoise_batch_all,
                'wave_spec_re_denoise_amp_batch_all':wave_spec_re_denoise_amp_batch_all,
                # 'baseline_batch_all':baseline_batch_all,
                'label_batch_all':label_batch_all,
                'dataset_names':dataset_names,
                'mni_coordinate_all': mni_coordinate_all,
                'mni_coordinate_raw_all': mni_coordinate_raw_all,
                'T1_coordinate_all': T1_coordinate_all,
                'T1_coordinate_raw_all': T1_coordinate_raw_all,
                'regions_all':regions_all,
                'gender_all':gender_all,
                'mask_all': mask_all,
                'word_batch_all':word_batch_all,
                'on_stage_batch_all':on_stage_batch_all,
                'on_stage_re_batch_all':on_stage_re_batch_all,
                'on_stage_wider_batch_all':on_stage_wider_batch_all,
                'on_stage_wider_re_batch_all':on_stage_wider_re_batch_all,
                
                }
            if self.formant:
                return_dict['formant_re_batch_all'] = formant_re_batch_all 
            if self.pitch:
                return_dict['pitch_re_batch_all'] = pitch_re_batch_all 
            if self.intensity:
                return_dict['intensity_re_batch_all'] = intensity_re_batch_all 
            return return_dict 
        else:
            return_dict = {'ecog_batch_all':ecog_batch_all,
                'ecog_raw_batch_all':ecog_raw_batch_all,
                'spkr_batch_all':spkr_batch_all,
                'phoneme_batch_all':phoneme_batch_all if self.phoneme else None,
                'voice_batch_all':voice_batch_all if self.phoneme else None,
                'unvoice_batch_all':unvoice_batch_all if self.phoneme else None,
                'semivoice_batch_all':semivoice_batch_all if self.phoneme else None,
                'plosive_batch_all':plosive_batch_all if self.phoneme else None,
                'fricative_batch_all':fricative_batch_all if self.phoneme else None,
                'wave_batch_all':wave_batch_all,
                'wave_spec_batch_all':wave_spec_batch_all,
                'wave_spec_amp_batch_all':wave_spec_amp_batch_all,
                'wave_denoise_batch_all':wave_denoise_batch_all,
                'wave_spec_denoise_batch_all':wave_spec_denoise_batch_all,
                'wave_spec_denoise_amp_batch_all':wave_spec_denoise_amp_batch_all,
                
                'ecog_re_batch_all':ecog_re_batch_all,
                'ecog_raw_re_batch_all':ecog_raw_re_batch_all,
                'spkr_re_batch_all':spkr_re_batch_all,
                'phoneme_re_batch_all':phoneme_re_batch_all if self.phoneme else None,
                'voice_re_batch_all':voice_re_batch_all if self.phoneme else None,
                'unvoice_re_batch_all':unvoice_re_batch_all if self.phoneme else None,
                'semivoice_re_batch_all':semivoice_re_batch_all if self.phoneme else None,
                'plosive_re_batch_all':plosive_re_batch_all if self.phoneme else None,
                'fricative_re_batch_all':fricative_re_batch_all if self.phoneme else None,
                'wave_re_batch_all':wave_re_batch_all,
                'wave_spec_re_batch_all':wave_spec_re_batch_all,
                'wave_spec_re_amp_batch_all':wave_spec_re_amp_batch_all,
                'wave_re_denoise_batch_all':wave_re_denoise_batch_all,
                'wave_spec_re_denoise_batch_all':wave_spec_re_denoise_batch_all,
                'wave_spec_re_denoise_amp_batch_all':wave_spec_re_denoise_amp_batch_all,
                # 'baseline_batch_all':baseline_batch_all,
                'label_batch_all':label_batch_all,
                'dataset_names':dataset_names,
                'mni_coordinate_all': mni_coordinate_all,
                'mni_coordinate_raw_all': mni_coordinate_raw_all,
                'T1_coordinate_all': T1_coordinate_all,
                'T1_coordinate_raw_all': T1_coordinate_raw_all,
                'regions_all':regions_all,
                'gender_all':gender_all,
                'mask_all': mask_all,
                'word_batch_all':word_batch_all,
                'on_stage_batch_all':on_stage_batch_all,
                'on_stage_re_batch_all':on_stage_re_batch_all,
                'on_stage_wider_batch_all':on_stage_wider_batch_all,
                'on_stage_wider_re_batch_all':on_stage_wider_re_batch_all,
                }
            if self.formant:
                return_dict['formant_re_batch_all'] = formant_re_batch_all 
            if self.pitch:
                print ('we use pitch')
                return_dict['pitch_re_batch_all'] = pitch_re_batch_all 
            if self.intensity:
                return_dict['intensity_re_batch_all'] = intensity_re_batch_all 
            return return_dict 


def concate_batch(metabatch,expection_keys=['ecog_batch_all','mask_all','label_batch_all','dataset_names','regions_all','word_batch_all']):
    for key in metabatch.keys():
        if key not in expection_keys:
            metabatch[key] = torch.cat(metabatch[key],dim=0)
    return metabatch