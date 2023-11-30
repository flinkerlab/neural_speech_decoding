function slices = sliceArray(array, dim, indices)
% array_slices = sliceArray(array, dim, indices)
%
% Return slices of array corresponding to defined indices along a defined
% dimension, independently of the number of dimensions of the array.
% If 'array' is a scalar, returns the subscripts of the slices in an array
% with 'array' dimensions.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

if isscalar(array)
    index_only = true;
    n_dims = array;
    
else
    index_only = false;
    n_dims = ndims(array);
end

subs = repmat({':'}, 1, n_dims);
subs{dim} = indices;

if index_only
    slices = subs;
    
else
    S = struct(...
        'subs', {subs},...
        'type', '()');
    
    slices = subsref(array, S);
end

end
