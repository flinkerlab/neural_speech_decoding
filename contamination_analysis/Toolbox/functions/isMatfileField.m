function logicals = isMatfileField(matfile_handle, field_names)
% logicals = isMatfileField(matfile_handle, field_names)
%
% Check if defined fields are present in a MAT-file.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

mat_field_names = who(matfile_handle);
logicals = ismember(field_names, mat_field_names);

end

