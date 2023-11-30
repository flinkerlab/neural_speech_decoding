function unpadded_data = removePadding(data, padding_width, dim, padding_side)
% removePadding(data, padding_width, dim, padding_side)
%
% Remove padding on signal of any dimension (see addPadding)
%
% data:
%   Array of any dimension.
% padding_width:
%   Number of samples of the padding.
% dim:
%   Indicates the dimension of 'data' on which padding was applied.
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

if ~exist('padding_side', 'var')  || isempty(padding_side)
    padding_side = 'both';
end

data_size = size(data);

S.type = '()';
S.subs = cell(size(data_size));
S.subs(:) = {':'};

switch padding_side
    case 'start'
        S.subs(dim) = {(padding_width + 1):data_size(dim)};
        unpadded_data = subsref(data, S);
    case 'end'
        S.subs(dim) = {1:(data_size(dim) - padding_width)};
        unpadded_data = subsref(data, S);
    case 'both'
        S.subs(dim) = {(padding_width + 1):(data_size(dim) - padding_width)};
        unpadded_data = subsref(data, S);
end

end