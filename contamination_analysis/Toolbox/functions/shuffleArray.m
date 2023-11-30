function a = shuffleArray(a, dim)
% a = shuffleArray(a, dim)
%
% Shuffle array along a defined dimension or along all dimensions.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

if ~exist('dim', 'var') || isempty(dim)
    dim = 1;
end

is_row = isrow(a);

if is_row
    a = a';
end

switch dim
    case 'all'
        indices = randperm(numel(a));
        a = reshape(a(indices), size(a));
        
    otherwise
        indices = randperm(size(a, dim));
        a = sliceArray(a, dim, indices);
end

if is_row
    a = a';
end

end

