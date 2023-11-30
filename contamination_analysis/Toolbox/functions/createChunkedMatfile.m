function matfile_handle = createChunkedMatfile(filepath, var_name, size,...
    chunk_size, fill_value, data_format)
% matfile_handle = createChunkedMatfile(filepath, var_name, size,
%  chunk_size, fill_value, data_format)
%
% Create MAT-File with chosen chunking. 
%
% This function was created because MATLAB's built-in chunking made reading
% large files slow.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

if ~exist('fill_value', 'var') || isempty(fill_value)
    fill_value = 0;
end

if ~exist('data_format', 'var') || isempty(data_format)
    data_format = 'double';
end

[path, filename] = fileparts( filepath );

mat_filename = fullfile(path, [filename '.mat']);

% initialize the file with a temporary variable
x = 1;
save(mat_filename, 'x', '-v7.3');

% open the created MAT file as an H5 file
file = H5F.open(mat_filename, 'H5F_ACC_RDWR', 'H5P_DEFAULT');

% delete the temporary variable
H5L.delete(file, 'x', 'H5P_DEFAULT');

% create dataspace
% set maximum size to [] sets the maximum size to be the current size
space = H5S.create_simple(length(size), fliplr(size), []);

% create the dataset creation property list
dcpl = H5P.create('H5P_DATASET_CREATE');

% set compression
%H5P.set_deflate( dcpl, 3 );

% set the chunk size
H5P.set_chunk(dcpl, fliplr(chunk_size));

% set the fill value as 0 for the dataset
fillval = eval([data_format '(fill_value)']);
switch data_format
    case 'double'
        h5_format = 'H5T_NATIVE_DOUBLE';
    case 'single'
        h5_format = 'H5T_NATIVE_FLOAT';
end
H5P.set_fill_value(dcpl, h5_format, fillval);

% set the allocation time to 'early' to ensure that reading from the
% dataset immediately after creation will return the fill value
H5P.set_alloc_time (dcpl, 'H5D_ALLOC_TIME_EARLY');

% create the chunked dataset
type_id = H5T.copy(h5_format);
dset = H5D.create(file, var_name, type_id, space,'H5P_DEFAULT',...
    dcpl,'H5P_DEFAULT');

% closing
H5P.close( dcpl );
H5D.close( dset );
H5S.close( space );
H5F.close( file );

% return handle
if nargout > 0
    matfile_handle = matfile(filepath, 'Writable', true);
end

end


