function [multichannel_spectrogram, time, frequencies] =...
    computeMultichannelSpectrogram(multichannel_data, spectrogram_fs, varargin)

% [multichannel_spectrogram, time, frequencies] = computeMultichannelSpectrogram(multichannel_data, spectrogram_fs, varargin)
% 
% Compute spectrogram of multi-channel signal. Based on the
% 'computeSpectrogram' function.
%
% INPUTS 
%
% multichannel_data:
%   Should be a (samples x channels) 2-D array or the path to a
%   'recording MAT-file' (see 'createRecordingMatfile' function).
% spectrogram_fs:
%   Desired sampling frequency of the resulting spectrograms.
%
% OUTPUTS
%
% multichannel_spectrogram:
%   A (samples x frequencies x channels) 3-D array or the handle to a
%   matfile containing the same data in its 'values' field.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

%%% PARSING
p = inputParser;

addRequired(p, 'multichannel_data');
addRequired(p, 'spectrogram_fs');

addParameter(p, 'window_duration', [])
addParameter(p, 'window_length', [])
addParameter(p, 'fs', [])
addParameter(p, 'fft_length', [])
addParameter(p, 'apply_log', false)
addParameter(p, 'output_path', [])
addParameter(p, 'padding_method', "none")
addParameter(p, 'display_waitbar', true)
addParameter(p, 'frequency_limits', [])
addParameter(p, 'interpolate_bool', false)
addParameter(p, 'channels', [])

parse(p, multichannel_data, spectrogram_fs, varargin{:})
r = p.Results;

%%% CHECK INPUT DATA
if ~(isstring(r.multichannel_data) || ischar(r.multichannel_data)) % array
    from_matfile = false;
    sample_nb = size(r.multichannel_data, 1);
    chan_nb = size(r.multichannel_data, 2);
    if isempty(r.fs)
        disp("(computeMultichannelSpectrogram) Error: sampling frequency of source data 'fs' is required.")
        return
    else
        fs = r.fs;
    end
    init_time = 0;
else % recording MAT-file
    from_matfile = true;
    multichannel_matfile = matfile(r.multichannel_data);
    matfile_values_dims = size(multichannel_matfile, 'values');
    sample_nb = matfile_values_dims(1);
    chan_nb = matfile_values_dims(2);
    fs = multichannel_matfile.fs;
    init_time = multichannel_matfile.time(1, 1);
    
    if isMatfileField(multichannel_matfile, 'channels')
        channels = multichannel_matfile.channels;
    end
end

if ~isempty(r.channels)
    channels = r.channels;
end

% if not in source file and not in input arguments
if ~exist('channels', 'var') || isempty(channels)
    channels = table(num2cell(1:chan_nb)', compose('%u', 1:chan_nb)',...
        'VariableNames', {'index', 'id'});
end

%%% CHECK WINDOW/FFT SIZE ARGUMENTS
if isempty(r.window_duration) && isempty(r.window_length)
    disp("(computeMultichannelSpectrogram) Error: 'window_duration' or 'window_length' is required.")
    return
elseif isempty(r.window_length)
    window_length = round(r.window_duration * fs);
else
    window_length = r.window_length;
    if ~isempty(r.window_duration)
        disp("(computeMultichannelSpectrogram) Warning: 'window_duration' is ignored because 'window_length' has been supplied.")
    end
end

if isempty(r.fft_length)
    fft_length = window_length;
end

%%% PRE-COMPUTE THE OUTPUT SIZE
% estimate the number of samples in the output spectrogram by anticipating
% the behavior of 'computeSpectrogram'
if r.interpolate_bool
    data_last_T = (sample_nb - 1) / fs;
    target_T = 0:(1/spectrogram_fs):data_last_T;
    spg_sample_nb = length(target_T);
    
else
    fft_overlap = floor(window_length - fs / r.spectrogram_fs);
    
    % estimate spectrogram samples number
    switch r.padding_method
        case "none"
            after_padding_sample_nb = sample_nb;
        case {"symmetrize", "zero-padding"}
            after_padding_sample_nb = sample_nb + 2 * floor(window_length / 2);
        otherwise
            disp("(computeMultichannelSpectrogram): Unknown padding method.")
            return
    end
    
    spg_sample_nb = floor((after_padding_sample_nb - fft_overlap) / (window_length - fft_overlap));
end

% estimate spectrogram frequency bins number
if ~mod(fft_length, 2)
    spg_freq_nb = fft_length / 2 + 1;
else
    spg_freq_nb = (fft_length + 1)/ 2;
end

frequencies = linspace(0, fs / 2, spg_freq_nb);
if ~isempty(r.frequency_limits)
    sel_freq_lgcs = (frequencies >= r.frequency_limits(1)) & (frequencies <= r.frequency_limits(2));
    frequencies = frequencies(sel_freq_lgcs);
    spg_freq_nb = length(frequencies);
end

%%% CHECK OUTPUT TYPE
if ~isempty(r.output_path) % save in Matfile
    save_bool = true; 
else
    save_bool = false; % return array
end

if save_bool
    multichannel_spectrogram = createChunkedMatfile(r.output_path, 'values',...
        [spg_sample_nb spg_freq_nb chan_nb],...
        [max(1, round(spg_sample_nb/100)) max(1, round(spg_freq_nb/10)) 1], 0, 'double');
else
    multichannel_spectrogram = struct('values', zeros(spg_sample_nb, spg_freq_nb, chan_nb));
end

%%% COMPUTE
if r.display_waitbar
    wb_h = timeWaitbar('Computing spectrograms');
end

for chan_i = 1:chan_nb
    
    if from_matfile
        [pspg, ~, pspg_time] = computeSpectrogram(multichannel_matfile.values(:, chan_i),...
            fs, window_length, r.spectrogram_fs, fft_length,...
            r.apply_log, r.padding_method, r.frequency_limits,...
            r.interpolate_bool);
    else
        [pspg, ~, pspg_time] = computeSpectrogram(multichannel_data(:, chan_i),...
            fs, window_length, r.spectrogram_fs, fft_length,...
            r.apply_log, r.padding_method, r.frequency_limits,...
            r.interpolate_bool);
    end
    
    if chan_nb > 1
        multichannel_spectrogram.values(:, :, chan_i) = permute(pspg, [2 1 3]);
    else
        multichannel_spectrogram.values(:, :) = permute(pspg, [2 1]);
    end
    
    if r.display_waitbar
        updateTimeWaitbar(wb_h, chan_i / chan_nb)
    end
    
end

if r.display_waitbar
    close(wb_h)
end

% time based on initial file
time = pspg_time + init_time;

% process output
if ~save_bool
    multichannel_spectrogram = multichannel_spectrogram.values;
else
    multichannel_spectrogram.time = time;
    multichannel_spectrogram.frequencies = frequencies;
    multichannel_spectrogram.fs = spectrogram_fs;
    multichannel_spectrogram.channels = channels;
    
    multichannel_spectrogram.dimension_ids = {'time', 'frequency', 'channel'};
    multichannel_spectrogram.dimension_values = {time(:), frequencies, channels.id};
end

end