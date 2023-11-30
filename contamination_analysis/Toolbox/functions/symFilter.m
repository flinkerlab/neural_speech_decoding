function filtered_data = symFilter(fir_coef, data, varargin)
% filtered_data = symFilter(fir_coef, data, varargin)
%
% Filtering for non causal symmetric FIR filter
%
% The filter size must be odd so that the middle element is centered on the
% current sample. The symmetry of the filter allows to avoid the
% introduction of phase delay in the output signal.
%
% Input:
%   - fir_coef : vector of symmetric FIR filter coefficients 
%   - data : (sample number)*(channel number) input data
% Output:
%   - f_data : (sample number)*(channel number) filtered data
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel, Blaise Yvert

%%% PARSING
p = inputParser;

addParameter(p, 'warning', true)
addParameter(p, 'padding', 'symmetrize')

parse(p, varargin{:})
r = p.Results;

if isrow(data)
    is_row = true;
    data = data';
else
    is_row = false;
end

fir_coef = fir_coef(:); % forcing f to be Nx1 to avoid having it 1xN
fir_len = length(fir_coef);

if r.warning
    if mod(fir_len, 2) == 0
        warning('Filter has an even number of coefficients, should ideally be odd.');
    end
    if any(abs(fir_coef - fliplr(fir_coef)) > eps)
        warning('Filter is not symmetric.');
    end
end

half_len = round((fir_len - 1) / 2); % half length of the filter
sample_nb = size(data, 1);             % number of samples
chan_nb = size(data, 2);               % number of channels

% padding
data = addPadding(data, half_len, 1, r.padding, 'both');

% filtering
filtered_data = NaN(sample_nb, chan_nb);
for c_i = 1:chan_nb
    filtered_data(:, c_i) = conv(data(:, c_i), fir_coef, 'valid');
end

if is_row
    filtered_data = filtered_data';
end
