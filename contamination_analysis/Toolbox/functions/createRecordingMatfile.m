function varagout = createRecordingMatfile(path, data, fs, varargin)
% matfile_h = createRecordingMatfile(path, data, fs, varargin)
%
% Create MAT-file with a defined structure.
%
% All RecordingMatfiles contain the following variables:
%  - values (data: samples x channels) - fs (sampling frequency) - time
%  (time vector: samples x 1) - channels (channels table: channels x (id |
%  index))
% If the two last elements are not supplied during creation, default ones
% are created. It is possible to simply initialize the file by supplying a
% scalar initialization value in 'data' and the size of 'values' in
% 'init_dims'.
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-laters
% Author(s): Philémon Roussel

p = inputParser;

addRequired(p, 'path', @ischar);
addRequired(p, 'data', @isnumeric);
addRequired(p, 'fs', @isscalar);

addParameter(p,'channels', [])
addParameter(p,'time', [])
addParameter(p,'info', '')
addParameter(p,'init_dims', [])

addParameter(p, 'dimension_ids', [])
addParameter(p, 'dimension_values', [])

parse(p, path, data, fs, varargin{:})

initialize_only = false;

if ~isempty(p.Results.init_dims)
    if isscalar(p.Results.data)
        values_dims = p.Results.init_dims;
        init_value = data;
        initialize_only = true;
    else
        disp('(createRecordingMatfile) Error: "data" argument should be scalar for file initialization.')
    end
else
    values_dims = size(data);
    init_value = 0;
end

sample_nb = values_dims(1);
chan_nb = values_dims(2);

if ~isempty(p.Results.channels)
    channels = p.Results.channels;
else
    channels = table((1:chan_nb)', compose('%u', 1:chan_nb)',...
        'VariableNames', {'index', 'id'});
end

if ~isempty(p.Results.time)
    time = p.Results.time;
else
    time = (0:(sample_nb - 1)) / fs;
end

% dimension ids
if isempty(p.Results.dimension_ids)
    dimension_ids = {'time', 'channel'};
else
    dimension_ids = p.Results.dimension_ids;
end

% dimension values
if isempty(p.Results.dimension_values)
    dimension_values = {time(:), channels};
else
    dimension_values = p.Results.dimension_values;
end

% create chunked matfile
path = pathify(path, 'create');

if sample_nb <= 256
    sample_chunk_dim = sample_nb;
else
    sample_chunk_dim = max(256, round(sample_nb / 100));
end

chunk_dims = [sample_chunk_dim, 1];
mat_format = 'double';
matfile_h = createChunkedMatfile(path, 'values', values_dims, chunk_dims, init_value, mat_format);

% fill descriptive fields
if ~isequal(channels, false)
    matfile_h.channels = channels;
end
matfile_h.fs = fs;
matfile_h.time = time(:);
matfile_h.info = p.Results.info;
matfile_h.dimension_ids = dimension_ids;
matfile_h.dimension_values = dimension_values;

% fill the data if provided
if ~initialize_only
    matfile_h.values(:, :) = data;
end

if nargout > 0
    varagout = matfile_h;
end

end

