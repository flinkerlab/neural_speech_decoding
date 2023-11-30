function varargout = displaySpectrogram(pspg, spg_time, spg_freqs,...
    freq_lims, display_type)
% varargout = displaySpectrogram(pspg, spg_time, spg_freqs,...
%    freq_lims, display_type)
%
% INPUTS:
%   spg_freqs: the window overlap will be automatically chosen so that the
%       output spectrogram is sampled as close as possible to 'spg_freqs'
%   freq_lims: 2-element array [min_frequency max_frequency]
% 
% Note: NaN values are set as transparent (appear white if default background)
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

if ~exist('display_type', 'var') || isempty(display_type)
    display_type = 'imagesc';
end   

if exist('freq_lims', 'var') && ~isempty(freq_lims)
    sel_freqs = (spg_freqs >= freq_lims(1) & spg_freqs <= freq_lims(2));
else
    sel_freqs = true(size(spg_freqs));
end

switch display_type
    case 'imagesc'
        h = imagesc(spg_time, spg_freqs(sel_freqs), pspg(sel_freqs, :),...
            'AlphaData', ~isnan(pspg(sel_freqs, :)));
    case 'pcolor'
        h = pcolor(spg_time, spg_freqs(sel_freqs), pspg(sel_freqs, :));
        shading interp
end

ax = gca;
colormap(ax, viridis())
set(ax,'YDir','normal')

% returns axes handle if required
if nargout > 0
    varargout{1} = h;
end

end