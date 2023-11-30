function padded_data = addPadding(data, padding_width, dim, padding_type, padding_side)
% addPadding(data, padding_width, dim, padding_type, padding_side)
%
% Apply padding on signal of any dimension.
%
% data:
%   Array of any dimension.
% padding_width:
%   Number of samples of the padding.
% dim:
%   Indicates the dimension of 'data' on which to apply padding.
% padding_type:
%   Can be a scalar value used for padding or one of the following string:
%   "zero" for zero-padding, "extend" to replicate the initial/final
%   values, "symmetrize" to mirror the begining/end of the signal.
% padding_side:
%   "start", "end" or "both".
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

if ~exist('dim', 'var')  || isempty(dim)
    if(isvector(data) && isrow(data))
        dim = 2;
    else
        dim = 1;
    end
end

if ~exist('padding_type', 'var')  || isempty(padding_type)
    padding_type = 'zero';
end

if isscalar(padding_type)
    padding_value = str2double(padding_type);
    padding_type = 'value';
end

if ~exist('padding_side', 'var')  || isempty(padding_side)
    padding_side = 'both';
end

data_size = size(data);
data_dims = ndims(data);
padding_size = data_size;
padding_size(dim) = padding_width;
sample_nb = size(data, dim);

switch padding_type
    case 'zero'
        start_padding_block = zeros(padding_size);
        end_padding_block = zeros(padding_size);
        
    case 'value'
        start_padding_block = padding_value .* ones(padding_size);
        end_padding_block = padding_value .* ones(padding_size);
        
    case 'extend'
        start_slice = sliceArray(data, dim, 1);
        end_slice = sliceArray(data, dim, sample_nb);
        
        rep_indices = ones(1, data_dims);
        rep_indices(dim) = padding_width;
        
        start_padding_block = repmat(start_slice, rep_indices);
        end_padding_block = repmat(end_slice, rep_indices);
        
    case 'symmetrize'
        
        start_padding_block = sliceArray(data, dim, 1:padding_width);
        start_padding_block = flip(start_padding_block, dim);
        end_padding_block = sliceArray(data, dim, (sample_nb - padding_width + 1):sample_nb);
        end_padding_block = flip(end_padding_block, dim);
end

data_size = size(data);
padding_size = data_size;
padding_size(dim) = padding_width;

switch padding_side
    case 'start'
        padded_data = cat(dim, start_padding_block, data);
    case 'end'
        padded_data = cat(dim, data, end_padding_block);
    case 'both'
        padded_data = cat(dim, start_padding_block, data, end_padding_block);
end

end