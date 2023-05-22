# ECoG to Speech Decoding using differentiable speech synthesizer


## Data

Prepare the data in h5 format

```


```

## Speech to Speech 


## ECoG to Speech

```
usage: train_formant_e_production_0328.py [-h] [-c FILE] [--DENSITY DENSITY] [--wavebased WAVEBASED]
                                          [--bgnoise_fromdata BGNOISE_FROMDATA] [--ignore_loading IGNORE_LOADING]
                                          [--finetune FINETUNE] [--learnedmask LEARNEDMASK]
                                          [--dynamicfiltershape DYNAMICFILTERSHAPE]
                                          [--formant_supervision FORMANT_SUPERVISION]
                                          [--pitch_supervision PITCH_SUPERVISION]
                                          [--intensity_supervision INTENSITY_SUPERVISION]
                                          [--n_filter_samples N_FILTER_SAMPLES] [--n_fft N_FFT]
                                          [--reverse_order REVERSE_ORDER] [--lar_cap LAR_CAP]
                                          [--intensity_thres INTENSITY_THRES] [--ONEDCONFIRST ONEDCONFIRST]
                                          [--RNN_TYPE RNN_TYPE] [--RNN_LAYERS RNN_LAYERS]
                                          [--RNN_COMPUTE_DB_LOUDNESS RNN_COMPUTE_DB_LOUDNESS] [--BIDIRECTION BIDIRECTION]
                                          [--MAPPING_FROM_ECOG MAPPING_FROM_ECOG] [--OUTPUT_DIR OUTPUT_DIR]
                                          [--COMPONENTKEY COMPONENTKEY] [--old_formant_file OLD_FORMANT_FILE]
                                          [--subject SUBJECT] [--trainsubject TRAINSUBJECT] [--testsubject TESTSUBJECT]
                                          [--HIDDEN_DIM HIDDEN_DIM] [--reshape RESHAPE] [--fastattentype FASTATTENTYPE]
                                          [--phone_weight PHONE_WEIGHT] [--ld_loss_weight LD_LOSS_WEIGHT]
                                          [--alpha_loss_weight ALPHA_LOSS_WEIGHT]
                                          [--consonant_loss_weight CONSONANT_LOSS_WEIGHT]
                                          [--amp_formant_loss_weight AMP_FORMANT_LOSS_WEIGHT]
                                          [--component_regression COMPONENT_REGRESSION]
                                          [--freq_single_formant_loss_weight FREQ_SINGLE_FORMANT_LOSS_WEIGHT]
                                          [--amp_minmax AMP_MINMAX] [--amp_energy AMP_ENERGY] [--f0_midi F0_MIDI]
                                          [--alpha_db ALPHA_DB] [--network_db NETWORK_DB]
                                          [--consistency_loss CONSISTENCY_LOSS] [--delta_time DELTA_TIME]
                                          [--delta_freq DELTA_FREQ] [--cumsum CUMSUM] [--distill DISTILL]
                                          [--noise_db NOISE_DB] [--classic_pe CLASSIC_PE]
                                          [--temporal_down_before TEMPORAL_DOWN_BEFORE] [--conv_method CONV_METHOD]
                                          [--classic_attention CLASSIC_ATTENTION] [--batch_size BATCH_SIZE]
                                          [--param_file PARAM_FILE] [--pretrained_model_dir PRETRAINED_MODEL_DIR]
                                          [--return_filtershape RETURN_FILTERSHAPE] [--causal CAUSAL]
                                          [--anticausal ANTICAUSAL] [--mapping_layers MAPPING_LAYERS]
                                          [--single_patient_mapping SINGLE_PATIENT_MAPPING] [--region_index REGION_INDEX]
                                          [--multiscale MULTISCALE] [--rdropout RDROPOUT] [--epoch_num EPOCH_NUM]
                                          [--cv CV] [--cv_ind CV_IND] [--LOO LOO] [--n_layers N_LAYERS]
                                          [--n_rnn_units N_RNN_UNITS] [--n_classes N_CLASSES] [--dropout DROPOUT]
                                          [--use_stoi USE_STOI] [--use_denoise USE_DENOISE] [--FAKE_LD FAKE_LD]
                                          [--extend_grid EXTEND_GRID] [--occlusion OCCLUSION]
                                          ...

ecog formant model

positional arguments:
  opts                  Modify config options using the command-line

optional arguments:
  -h, --help            show this help message and exit
  -c FILE, --config-file FILE
                        path to config file
  --DENSITY DENSITY     Data density, LD for low density, HB for hybrid density
  --wavebased WAVEBASED
                        wavebased or not
  --bgnoise_fromdata BGNOISE_FROMDATA
                        bgnoise_fromdata or not, if false, means learn from spec
  --ignore_loading IGNORE_LOADING
                        ignore_loading true: from scratch, false: finetune
  --finetune FINETUNE   finetune could influence load checkpoint
  --learnedmask LEARNEDMASK
                        finetune could influence load checkpoint
  --dynamicfiltershape DYNAMICFILTERSHAPE
                        finetune could influence load checkpoint
  --formant_supervision FORMANT_SUPERVISION
                        formant_supervision
  --pitch_supervision PITCH_SUPERVISION
                        pitch_supervision
  --intensity_supervision INTENSITY_SUPERVISION
                        intensity_supervision
  --n_filter_samples N_FILTER_SAMPLES
                        distill use or not
  --n_fft N_FFT         deliberately set a wrong default to make sure feed a correct n fft
  --reverse_order REVERSE_ORDER
                        reverse order of learn filter shape from spec, which is actually not appropriate
  --lar_cap LAR_CAP     larger capacity for male encoder
  --intensity_thres INTENSITY_THRES
                        used to determine onstage, 0 means we use the default setting in Dataset.json
  --ONEDCONFIRST ONEDCONFIRST
                        use one d conv before lstm
  --RNN_TYPE RNN_TYPE   LSTM or GRU
  --RNN_LAYERS RNN_LAYERS
                        lstm layers/3D swin transformer model ind
  --RNN_COMPUTE_DB_LOUDNESS RNN_COMPUTE_DB_LOUDNESS
                        RNN_COMPUTE_DB_LOUDNESS
  --BIDIRECTION BIDIRECTION
                        BIDIRECTION
  --MAPPING_FROM_ECOG MAPPING_FROM_ECOG
                        MAPPING_FROM_ECOG
  --OUTPUT_DIR OUTPUT_DIR
                        OUTPUT_DIR
  --COMPONENTKEY COMPONENTKEY
                        COMPONENTKEY
  --old_formant_file OLD_FORMANT_FILE
                        check if use old formant could fix the bug?
  --subject SUBJECT     e.g.
  --trainsubject TRAINSUBJECT
                        if None, will use subject info, if specified, the training subjects might be different from
                        subject
  --testsubject TESTSUBJECT
                        if None, will use subject info, if specified, the test subjects might be different from subject
  --HIDDEN_DIM HIDDEN_DIM
                        HIDDEN_DIM
  --reshape RESHAPE     -1 None, 0 no reshape, 1 reshape
  --fastattentype FASTATTENTYPE
                        full,mlinear,local,reformer
  --phone_weight PHONE_WEIGHT
                        phoneneme classifier CE weight
  --ld_loss_weight LD_LOSS_WEIGHT
                        ld_loss_weight use or not
  --alpha_loss_weight ALPHA_LOSS_WEIGHT
                        alpha_loss_weight use or not
  --consonant_loss_weight CONSONANT_LOSS_WEIGHT
                        consonant_loss_weight use or not
  --amp_formant_loss_weight AMP_FORMANT_LOSS_WEIGHT
                        amp_formant_loss_weight use or not
  --component_regression COMPONENT_REGRESSION
                        component_regression or not
  --freq_single_formant_loss_weight FREQ_SINGLE_FORMANT_LOSS_WEIGHT
                        freq_single_formant_loss_weight use or not
  --amp_minmax AMP_MINMAX
                        amp_minmax use or not
  --amp_energy AMP_ENERGY
                        amp_energy use or not, amp times loudness
  --f0_midi F0_MIDI     f0_midi use or not,
  --alpha_db ALPHA_DB   alpha_db use or not,
  --network_db NETWORK_DB
                        network_db use or not, change in net_formant
  --consistency_loss CONSISTENCY_LOSS
                        consistency_loss use or not
  --delta_time DELTA_TIME
                        delta_time use or not
  --delta_freq DELTA_FREQ
                        delta_freq use or not
  --cumsum CUMSUM       cumsum use or not
  --distill DISTILL     distill use or not
  --noise_db NOISE_DB   distill use or not
  --classic_pe CLASSIC_PE
                        classic_pe use or not
  --temporal_down_before TEMPORAL_DOWN_BEFORE
                        temporal_down_before use or not
  --conv_method CONV_METHOD
                        conv_method
  --classic_attention CLASSIC_ATTENTION
                        classic_attention
  --batch_size BATCH_SIZE
                        batch_size
  --param_file PARAM_FILE
                        param_file
  --pretrained_model_dir PRETRAINED_MODEL_DIR
                        pretrained_model_dir
  --return_filtershape RETURN_FILTERSHAPE
                        return_filtershape or not
  --causal CAUSAL       causal
  --anticausal ANTICAUSAL
                        anticausal
  --mapping_layers MAPPING_LAYERS
                        mapping_layers
  --single_patient_mapping SINGLE_PATIENT_MAPPING
                        single_patient_mapping
  --region_index REGION_INDEX
                        region_index
  --multiscale MULTISCALE
                        multiscale
  --rdropout RDROPOUT   rdropout
  --epoch_num EPOCH_NUM
                        epoch num
  --cv CV               do not use cross validation for default!
  --cv_ind CV_IND       k fold CV ind
  --LOO LOO             Leave One Out experiment word
  --n_layers N_LAYERS   RNN n_layers
  --n_rnn_units N_RNN_UNITS
                        RNN n_rnn_units
  --n_classes N_CLASSES
                        RNN n_classes
  --dropout DROPOUT     RNN dropout
  --use_stoi USE_STOI   Use STOI+ loss or not
  --use_denoise USE_DENOISE
                        Use denoise audio or not
  --FAKE_LD FAKE_LD     only true for HB e2a exp but only use first 64 electrodes!
  --extend_grid EXTEND_GRID
                        for LD, extend grids to more than 64 electrodes!
  --occlusion OCCLUSION
                        occlusion analysis to locate electrodes
```


## TODOs
- [ ] the framework starts from ECoG, Speech, onset and offset
- [ ] change dataloader
- [ ] model_formant
- [ ] net_formant
- [ ] train

 