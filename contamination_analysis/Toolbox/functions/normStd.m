function [norm_A, offset, scale] = normStd(A, dim, center_type, var_type)
% function [norm_A, offset, scale] = normStd(A, dim, center_type, var_type)
%
% Get standard normalization offset and scale (to 0 mean and 1 standard
% deviation) and apply it.
%
% Omit NaNs.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Florent Bocquelet, Philémon Roussel

if ~exist('dim', 'var') || isempty(dim)
    dim = 1;
end

if ~exist('center_type', 'var') || isempty(center_type)
    center_type = "mean";
end

if ~exist('var_type', 'var') || isempty(var_type)
    var_type = "std";
end

if strcmp(center_type, "mean")
    offset = mean(A, dim, 'omitnan');
elseif strcmp(center_type, "median")
    offset = median(A, dim, 'omitnan');
else
    error("Unknown centering type.")
end

if strcmp(var_type, "std")
    scale = std(A, 0, dim, 'omitnan');
elseif strcmp(var_type, "mad")
    scale = median(abs(applyNormalization(A, offset, 1, dim)), dim, 'omitnan');
else
    error("Unknown scaling type.")
end

% avoid 0 division
zero_scale_idxs = (scale == 0);
if any(zero_scale_idxs)
    scale(zero_scale_idxs) = 1;
    warning("Scaling by zero ignored.")
end

norm_A = applyNormalization(A, offset, scale, dim);