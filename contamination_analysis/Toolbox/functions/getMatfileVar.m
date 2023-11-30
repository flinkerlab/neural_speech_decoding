function value = getMatfileVar(source_matfile, variable_name)
% value = getMatfileVar(source_matfile, variable_name)
%   
% Returns the value of variable 'variable_name' in MAT-file
% 'source_matfile'.
%
% This function was built to load a particular variable without having to
% use load which returns a structure or to create a MAT-file handle.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

    if ~isa(source_matfile, 'matlab.io.MatFile')
        source_matfile = matfile(source_matfile);
    end
    
     value = source_matfile.(variable_name);
end

