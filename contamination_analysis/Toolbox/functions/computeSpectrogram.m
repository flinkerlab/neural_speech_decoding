function [PSD, frequencies, T] = computeSpectrogram(data, data_freq,...
    spec_window_size, spec_freq, spec_nfft, log_bool, padding_method,...
    frequency_limits, interpolate_bool)
% [PSD, frequencies, T] = computeSpectrogram(data, data_freq,...
%   spec_window_size, spec_freq, spec_nfft, log_bool, padding_method,...
%   frequency_limits, interpolate_bool)
%
% The output is the power spectral density (PSD) as a function of time,
% each column being a frequency coefficient and each row a time sample.
%
% data:
%   Vector on which the spectrogram is computed.
% data_freq:
%   Sampling frequency of 'data'.
% spec_window_size:
%   Size of the spectrogram window (in number of samples).
% spec_freq:
%   Desired spectrogram frequency.
% spec_nfft:
%   Size of the FFT window (equal to 'spec_window_size' by default).
% log_bool:
%   Boolean used to apply a log on the resulting spectral powers.
% padding_method:
%   "symmetrize" or "zero-padding" or "none".
% frequency_limits:
%   Vector of two elements indicating the lowest and the highest
%   frequencies to be included in the results.
% interpolate_bool:
%   Boolean used to apply interpolation on the results. Interpolation is
%   applied so that the resulting spectrogram timepoints are actually
%   sampled at 'spec_freq'. Interpolation is only available when padding is
%   applied.
%
% NOTE: If you change the way this function handles input arguments,
% apply these changes to computeMultichannelSpectrogram function.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Florent Bocquelet, Philémon Roussel

% ensure column vector
data = data(:);
original_sample_nb = length(data);

if ~exist('interpolate_bool', 'var') || isempty(interpolate_bool)
    interpolate_bool = false;
elseif interpolate_bool
    if strcmp(padding_method, "none")
        disp('(computeSpectrogram) Warning: Interpolation can only be used when padding is applied.')
    end
end

if ~exist('log_bool', 'var') || isempty(log_bool)
    log_bool = false;
end

if ~exist('padding_method', 'var') || isempty(padding_method)
    padding_method = "none";
elseif isnumeric(padding_method)
    padding_method = num2str(padding_method);
end

if ~exist('frequency_limits', 'var')
    frequency_limits = [];
end

if isempty(spec_nfft)
    spec_nfft = spec_window_size;
end

% estimate spectrogram frequency bins number
if ~mod(spec_nfft, 2)
    spg_freq_nb = spec_nfft / 2 + 1;
else
    spg_freq_nb = (spec_nfft + 1)/ 2;
end

% select frequency bins
frequencies = linspace(0, data_freq / 2, spg_freq_nb);
if ~isempty(frequency_limits)
    sel_freq_lgcs = (frequencies >= frequency_limits(1))...
        & (frequencies <= frequency_limits(2));
    frequencies = frequencies(sel_freq_lgcs);
    spg_freq_nb = length(frequencies);
else
    sel_freq_lgcs = true(spg_freq_nb, 1);
end

half_wdw_len = floor(spec_window_size / 2);

switch padding_method
    case {"0", "none"}
        % nothing
    case {"1", "symmetrize"}
        data = addPadding(data, half_wdw_len, 1, 'symmetrize', 'both');
    case {"2", "zero-padding", "zero"}
        data = addPadding(data, half_wdw_len, 1, 'zero', 'both');
    case {"extend"}
        data = addPadding(data, half_wdw_len, 1, 'extend', 'both');
    otherwise
        disp("(computeSpectrogram) Error: Padding method unknown.")
        return
end

fft_overlap = floor(spec_window_size - data_freq / spec_freq);
[~, frequencies, T, PSD] = spectrogram(...
    double(data), hamming(spec_window_size), fft_overlap, spec_nfft,...
    double(data_freq), 'yaxis');

switch padding_method
    case {"0", "none"}
        % nothing
    case {"1", "symmetrize", "extend"}
        T = T - T(1);
    case {"2", "zero-padding", "zero"}
        % In the previous spectrogram computation, the windowing was done
        % after the padding of the edges. Thus, the windowing did not
        % remove the jumps between the zero-padded periods and the original
        % signal. The present version corrects the problem by applying a
        % variable-sized hamming window exclusively on the non-zero
        % portions of the signal.
        
        T = T - T(1);
        
        sample_nb = length(data);
        
        wdw_start_indices = 1:(spec_window_size - fft_overlap):(sample_nb - spec_window_size + 1);
        wdw_end_indices = spec_window_size:(spec_window_size - fft_overlap):sample_nb;
        wdw_nb = length(wdw_end_indices);
        
        last_left_padding_index = half_wdw_len;
        last_left_wdw_to_recompute = find(wdw_start_indices > last_left_padding_index, 1, 'first') - 1;
        
        first_right_padding_index = length(data) - half_wdw_len + 1;
        first_right_wdw_to_recompute = find(wdw_end_indices < first_right_padding_index, 1, 'last') + 1;
        
        for i = 1:last_left_wdw_to_recompute
            wdw_start = wdw_start_indices(i);
            wdw_end = wdw_end_indices(i);
            
            zero_samples = wdw_start:half_wdw_len;
            PSD(:, i) = abs(spectrum(data(wdw_start:wdw_end) .* [zeros(length(zero_samples), 1) ; hamming(spec_window_size - length(zero_samples))], data_freq)).^2;
        end
        
        for i = first_right_wdw_to_recompute:wdw_nb
            wdw_start = wdw_start_indices(i);
            wdw_end = wdw_end_indices(i);
            
            zero_samples = (sample_nb - half_wdw_len +1):wdw_end;
            PSD(:, i) = abs(spectrum(data(wdw_start:wdw_end) .* [hamming(spec_window_size - length(zero_samples)) ; zeros(length(zero_samples), 1)], data_freq)).^2;
        end
end

if ~isempty(frequency_limits)
    frequencies = frequencies(sel_freq_lgcs);
    PSD = PSD(sel_freq_lgcs, :);
end

if mean(imag(PSD(:))>0) > 0
    disp("(computeSpectrogram) Warning: Complex values in power spectrum.")
else
    PSD = real(PSD);
end

if log_bool
    PSD = log(PSD + eps);
end

if interpolate_bool && ~strcmp(padding_method, "none")
    
    data_last_T = (original_sample_nb - 1) / data_freq;
    target_T = 0:(1/spec_freq):data_last_T; 
    
    % for windows with an even number of samples, the exact time of the
    % window should correspond to its center
    if mod(spec_window_size, 2) == 0
        T = T - (1/data_freq/2);
    end
    
    if ~isequal(T, target_T)
        PSD = interp1(T, PSD', target_T, 'linear')';
        T = target_T;
    end
end

end


