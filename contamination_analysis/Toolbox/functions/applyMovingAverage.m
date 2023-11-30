function smoothed = applyMovingAverage(signal, span, fs, padding_type)
% smoothed = applyMovingAverage(signal, span, fs, padding_type)
%
% Apply moving average filtering to signal.
%
%   The input signal should be an array of dimension (nb of samples)x(nb of
%   channels).
%   The span represents the length of the window used for computing the
%   local average. If the sampling frequency of the input signal is not
%   specified, the unit of the span is the number of sample, otherwise it
%   is in seconds. 
%   The filter is made odd so that it can be centered on the
%   current sample (avoiding phase delay).
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

% check if the sampling frequency of the signal is specified
if ~exist('fs', 'var') || isempty(fs)
    n = span;
else
    n = round(span * fs);
end

if ~exist('padding_type', 'var') || isempty(padding_type)
    padding_type = 'zero';
end


% round to the nearest odd integer
len = round((n - 1) / 2) * 2 + 1;

switch padding_type
    case 'zero' % default behavior of the algorithm
    case {'extend', 'symmetrize'}
        signal = addPadding(signal, len, 1, padding_type, 'both');
end

% build the FIR filter
b = 1 / len * ones( 1, len );

% apply the filter
smoothed = symFilter(b, signal);

switch padding_type
    case 'zero' % default behavior of the algorithm
    case {'extend', 'symmetrize'}
        smoothed = removePadding(smoothed, len, 1, 'both');
end

end

