import json, os, random
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
import dareblopy as db
import torch.utils
import torch.utils.data
cpu = torch.device('cpu')


class ECoGDataset(Dataset):
    """docstring for ECoGDataset"""
        
    def __init__(self, cfg,ReqSubjDict, mode = 'train', train_param = None,BCTS=None,world_size=1,ReshapeAsGrid=None,
                 DEBUG=False, rearrange_elec=0, low_density = True, process_ecog = True, formant_label = False, allsubj_param = None, 
                 pitch_label = False, intensity_label = False, data_dir = 'data/',infer=False,repeattimes=128):
        """ ReqSubjDict can be a list of multiple subjects"""
        super(ECoGDataset, self).__init__()
        self.DEBUG = DEBUG
        self.world_size = world_size
        self.current_lod=2
        self.ReqSubjDict = ReqSubjDict
        self.mode = mode
        BCTS = cfg.DATASET.BCTS
        self.BCTS = BCTS
        self.infer = infer
        self.rearrange_elec = rearrange_elec
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

        
        self.meta_data = {}
        for sample in ReqSubjDict:
            meta_h5_file = os.path.join(data_dir,sample + '.h5')
            if not os.path.exists(meta_h5_file):
                raise FileNotFoundError(meta_h5_file) 
            else:
                meta_data =  h5py.File(meta_h5_file,'r') 
            for k, v in meta_data.items():
                if k in self.meta_data:
                    self.meta_data[k].extend([v])
                else:
                    self.meta_data[k] = [v]
        
        self.dataset_names  = ReqSubjDict
        if train_param == None:
            with open('configs/train_param_production.json','r') as rfile:
                train_param = json.load(rfile)
        else:
            pass
        self.ORG_WAVE_FS = allsubj_param['Shared']['ORG_WAVE_FS']
        self.ORG_ECOG_FS = allsubj_param['Shared']['ORG_ECOG_FS']
        self.DOWN_WAVE_FS  = allsubj_param['Shared']['DOWN_WAVE_FS']
        self.ORG_ECOG_FS_NY = allsubj_param['Shared']['ORG_ECOG_FS_NY']
        self.ORG_TF_FS = allsubj_param['Shared']['ORG_TF_FS']
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
        self.UseGridOnly,self.SeqLen = train_param['UseGridOnly'],\
                                                                    train_param['SeqLen']
        self.Prod = cfg.DATASET.PROD
        self.ahead_onset_test = train_param['Test']['ahead_onset']
        self.ahead_onset_train = train_param['Train']['ahead_onset']
        self.DOWN_TF_FS = train_param['DOWN_TF_FS']
        self.DOWN_ECOG_FS = train_param['DOWN_ECOG_FS']
        self.Wipenoise = False
        self.meta_data['gender_alldataset'] = [allsubj_param["Subj"][subj]['Gender'] for subj in ReqSubjDict]
    
        self.ecog_alldataset = self.meta_data['ecog_alldataset']
        #self.label_alldataset = self.meta_data['label_alldataset']
        self.label_alldataset = [np.array([i.decode('utf-8') for i in meta_data['label_alldataset'][:]]).astype('str')]

        self.TestNum_cum = np.array([np.sum(train_param["Subj"][subj]['TestNum'] ).astype(np.int32) for subj in ReqSubjDict])
        if self.formant:
            self.formant_re_alldataset = self.meta_data['formant_re_alldataset']
        if self.pitch:
            self.pitch_re_alldataset = self.meta_data['pitch_re_alldataset']
        if self.intensity:
            self.intensity_re_alldataset = self.meta_data['intensity_re_alldataset']        
        self.start_ind_re_valid_alldataset = self.meta_data['start_ind_re_valid_alldataset']
        self.end_ind_re_valid_alldataset = self.meta_data['end_ind_re_valid_alldataset']
        self.wave_spec_re_alldataset = self.meta_data['wave_re_spec_alldataset']
        self.wave_re_alldataset = self.meta_data['wave_re_alldataset']
        self.wave_spec_re_amp_alldataset = self.meta_data['wave_re_spec_amp_alldataset']
        self.repeattimes =  repeattimes
    def __len__(self):
        repeattimes = self.repeattimes
        if self.mode == 'train':
            return np.array([start_ind_re_alldataset.shape[0]*(repeattimes if 'NY798' in self.ReqSubjDict else repeattimes)//self.world_size for start_ind_re_alldataset in self.meta_data['start_ind_re_valid_alldataset']]).sum()
        else:
            return self.TestNum_cum[0]

    def __getitem__(self, idx):
        n_delay_1 = -16 
        n_delay_2 = 0
        num_dataset = len(self.ecog_alldataset)
        gender_all = []
        ecog_re_batch_all = []
        formant_re_batch_all = []
        intensity_re_batch_all = []
        pitch_re_batch_all = []
        wave_re_batch_all = []
        wave_spec_re_batch_all = []
        wave_spec_re_amp_batch_all = []
        label_batch_all = []
        on_stage_re_batch_all = []
        on_stage_wider_re_batch_all = []
        self.SeqLenSpkr = self.SeqLen*int(self.DOWN_TF_FS*1.0/self.DOWN_ECOG_FS)
        pre_articulate_len = self.ahead_onset_test
        for i in range(num_dataset):
            if self.mode =='train':
                if self.infer:
                    rand_ind = idx
                else:
                    rand_ind = np.random.choice(np.arange(self.start_ind_re_valid_alldataset[i].shape[0])[:-self.TestNum_cum[i]],1,replace=False)[0]
            elif self.mode =='test':
                rand_ind = idx+self.start_ind_re_valid_alldataset[i].shape[0]-self.TestNum_cum[i]
            #print ('rand_ind',idx, rand_ind)
            label = [self.label_alldataset[i][rand_ind]]
            
            if self.Prod:
                start_indx_re = self.start_ind_re_valid_alldataset[i][rand_ind]
                end_indx_re = self.end_ind_re_valid_alldataset[i][rand_ind]
                if self.pre_articulate:
                    ecog_batch_re = np.zeros((pre_articulate_len ,self.ecog_alldataset[i].shape[-1]))
                else:
                    ecog_batch_re = np.zeros((self.SeqLen+n_delay_2-n_delay_1 ,self.ecog_alldataset[i].shape[-1]))
                formant_batch_re = np.zeros(( self.SeqLenSpkr, 6))
                pitch_batch_re = np.zeros(( self.SeqLenSpkr))
                intensity_batch_re = np.zeros(( self.SeqLenSpkr))
                wave_batch_re = np.zeros(( (self.SeqLen*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)),self.wave_re_alldataset[i].shape[-1]))
                if self.wavebased:
                    wave_spec_batch_re = np.zeros(( self.SeqLen, self.wave_spec_re_alldataset[i].shape[-1]))
            if self.mode =='test' or self.pre_articulate:
                indx_re = start_indx_re-self.ahead_onset_test
            elif self.mode =='train':
                chosen_start_re = np.random.choice(np.arange(-64,end_indx_re-start_indx_re-64),1)[0]
                indx_re = np.maximum(start_indx_re+chosen_start_re,0)
            on_stage_re_batch = np.zeros([1,self.SeqLenSpkr])
            on_stage_re_batch[:,np.maximum(start_indx_re-indx_re,0): np.minimum(end_indx_re-indx_re,self.SeqLenSpkr-1)] = 1.0
            on_stage_wider_re_batch = np.zeros([1,self.SeqLenSpkr])
            on_stage_wider_re_batch[:,np.maximum(start_indx_re-indx_re-5,0): np.minimum(end_indx_re-indx_re+5,self.SeqLenSpkr-1)] = 1.0
            if self.pre_articulate:
                ecog_batch_re = self.ecog_alldataset[i][indx_re:indx_re+pre_articulate_len]
            else:
                ecog_batch_re = self.ecog_alldataset[i][indx_re+n_delay_1:indx_re+self.SeqLen+n_delay_2]
            if self.formant:
                formant_batch_re = self.formant_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
            if self.pitch:
                pitch_batch_re = self.pitch_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
            if self.intensity:
                intensity_batch_re = self.intensity_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
            if self.wavebased:
                wave_spec_batch_re = self.wave_spec_re_alldataset[i][indx_re:indx_re+self.SeqLen]
                wave_spec_batch_amp_re = self.wave_spec_re_amp_alldataset[i][indx_re:indx_re+self.SeqLen]
            wave_batch_re = self.wave_re_alldataset[i][(indx_re*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)):((indx_re+self.SeqLen)*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS))]
            if self.Prod:
                ecog_re_batch_all += [ecog_batch_re]
                if self.formant:
                    formant_re_batch_all += [formant_batch_re[np.newaxis,...]]
                if self.pitch:
                    pitch_re_batch_all += [pitch_batch_re[np.newaxis,...]]
                if self.intensity:
                    intensity_re_batch_all += [intensity_batch_re[np.newaxis,...]]
                if self.wavebased:
                    wave_spec_re_batch_all += [wave_spec_batch_re[np.newaxis,...]]
                    wave_spec_re_amp_batch_all += [wave_spec_batch_amp_re[np.newaxis,...]]
                wave_re_batch_all += [wave_batch_re.swapaxes(-2,-1)]
                on_stage_re_batch_all += [on_stage_re_batch]
                on_stage_wider_re_batch_all += [on_stage_wider_re_batch]
            label_batch_all +=[label]
            gender_all +=[np.array([0.],dtype=np.float32) if self.meta_data['gender_alldataset'][i]=='Male' else np.array([1.],dtype=np.float32)]
        ecog_re_batch_all  = np.concatenate(ecog_re_batch_all,axis=0)
        if self.Prod:
            if self.formant:
                formant_re_batch_all = np.concatenate(formant_re_batch_all,axis=0)
            if self.pitch:
                pitch_re_batch_all = np.concatenate(pitch_re_batch_all,axis=0)
            if self.intensity:
                intensity_re_batch_all = np.concatenate(intensity_re_batch_all,axis=0)
            if self.wavebased:
                wave_spec_re_batch_all = np.concatenate(wave_spec_re_batch_all,axis=0)
                wave_spec_re_amp_batch_all = np.concatenate(wave_spec_re_amp_batch_all,axis=0)
            wave_re_batch_all = np.concatenate(wave_re_batch_all,axis=0)
            on_stage_re_batch_all = np.concatenate(on_stage_re_batch_all,axis=0)
            on_stage_wider_re_batch_all = np.concatenate(on_stage_wider_re_batch_all,axis=0)
        label_batch_all = np.concatenate(label_batch_all,axis=0).tolist()
        gender_all = np.concatenate(gender_all,axis=0)
        
        return_dict = {'ecog_re_batch_all':ecog_re_batch_all,
                'wave_re_batch_all':wave_re_batch_all,
                'wave_spec_re_batch_all':wave_spec_re_batch_all,
                'wave_spec_re_amp_batch_all':wave_spec_re_amp_batch_all,
                'label_batch_all':label_batch_all,
                'dataset_names':self.dataset_names,
                'gender_all':gender_all,
                'on_stage_re_batch_all':on_stage_re_batch_all,
                'on_stage_wider_re_batch_all':on_stage_wider_re_batch_all,
                }
        if self.formant:
            return_dict['formant_re_batch_all'] = formant_re_batch_all 
        if self.pitch:
            return_dict['pitch_re_batch_all'] = pitch_re_batch_all 
        if self.intensity:
            return_dict['intensity_re_batch_all'] = intensity_re_batch_all 
        return return_dict




class TFRecordsDataset:
    def __init__(self, cfg, logger, rank=0, world_size=1, buffer_size_mb=200,data_dir = 'data/',\
        infer=False, channels=3, seed=None, train=True, needs_labels=False,param=None,\
            ReshapeAsGrid=None,SUBJECT='NY742',rearrange_elec=False,low_density = True,\
                process_ecog = True, formant_label = False, pitch_label = False, \
                    intensity_label = False,DEBUG=False,allsubj_param=None,repeattimes=128):
        self.param = param
        self.dataset = ECoGDataset(cfg, SUBJECT, mode='train' if train else 'test', world_size = world_size, \
            train_param = param, allsubj_param = allsubj_param, ReshapeAsGrid = ReshapeAsGrid, rearrange_elec = rearrange_elec, low_density = low_density, \
                process_ecog = process_ecog, formant_label = formant_label, pitch_label = pitch_label, \
                    intensity_label = intensity_label,DEBUG=DEBUG,infer=infer,data_dir = data_dir,repeattimes=repeattimes)
        self.noise_dist = self.dataset.meta_data['noisesample_re_alldataset'][0][:]
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.infer = infer
        self.last_data = ""
        if train:
            self.part_count = cfg.DATASET.PART_COUNT
            self.part_size = cfg.DATASET.SIZE // self.part_count
        else:
            self.part_count = cfg.DATASET.PART_COUNT_TEST
            self.part_size = cfg.DATASET.SIZE_TEST // self.part_count
        self.workers = []
        self.workers_active = 0
        self.iterator = None
        self.filenames = {}
        self.batch_size = cfg.TRAIN.BATCH_SIZE if train else len(self.dataset)
        self.features = {}
        self.channels = channels
        self.seed = seed
        self.train = train
        self.needs_labels = needs_labels
        assert self.part_count % world_size == 0
        self.part_count_local = self.part_count // world_size
        self.buffer_size_b = 1024 ** 2 * buffer_size_mb
        self.iterator = torch.utils.data.DataLoader(self.dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True if (self.train and not self.infer) else False,
                                               drop_last=True if self.train else False)
    def reset(self, lod, batch_size):
        if batch_size!=self.batch_size:
            self.iterator = torch.utils.data.DataLoader(self.dataset,
                                               batch_size=batch_size,
                                               shuffle=True if (self.train and not self.infer) else False,
                                               drop_last=True if self.train else False)
        self.batch_size = batch_size
        self.dataset.current_lod=lod
    def __iter__(self):
        return iter(self.iterator)

    def __len__(self):
        return len(self.dataset)#self.part_count_local * self.part_size
    
def make_dataloader(cfg, logger, dataset, GPU_batch_size, local_rank, numpy=False):
    class BatchCollator(object):
        def __init__(self, device=torch.device("cpu")):
            self.device = device
            self.flip = cfg.DATASET.FLIP_IMAGES
            self.numpy = numpy

        def __call__(self, batch):
            with torch.no_grad():
                x, = batch
                if self.flip:
                    flips = [(slice(None, None, None), slice(None, None, None), slice(None, None, random.choice([-1, None]))) for _ in range(x.shape[0])]
                    x = np.array([img[flip] for img, flip in zip(x, flips)])
                if self.numpy:
                    return x
                x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)
                return x

    batches = db.data_loader(iter(dataset), BatchCollator(local_rank), len(dataset) // GPU_batch_size)

    return batches