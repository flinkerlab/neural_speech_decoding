function norm_A = applyNormalization(A, offset, scale, dim)
% norm_A = applyNormalization(A, offset, scale, dim)
%
% Apply offset and scale to all elements in A.
% 
% All the elements at the same position along dimension 'dim' are
% normalized equally.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Florent Bocquelet, Philémon Roussel

if ~exist('dim', 'var') || isempty(dim)
    dim = 1;
end

rep_dims = ones(1, ndims(A));
rep_dims(dim) = size(A,dim);

norm_A = (A - repmat(offset, rep_dims)) ./ repmat(scale, rep_dims);

end
