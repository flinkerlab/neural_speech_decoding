function linkFigureAxes(varargin)
% linkFigureAxes 
% Finds all visible axes in figure and links them for zooming
% 
% Syntax: linkFigureAxes(varargin)
% - argument 1: can be 'x', 'y', 'xy', 'off'
%   Works the same as built-in function linkaxes.
% - argument 2: figure handle
%   Default value is the currently active figure.
%
% Example:
% f = figure;
% subplot(2,1,1);
% plot(rand(10,1));
% subplot(2,1,2);
% plot(1:10);
% linkFigureAxes('x', f) % OR linkFigureAxes('x')
%
% Based on: linkaxesInFigure
% Copyright 2012 Prime Photonics, LC
% License: 2-clause BSD
% Author(s): Dan Kominsky
% Source: https://www.mathworks.com/matlabcentral/fileexchange/34476-link-all-axes-in-figure 
%
% Modifications:
% Copyright 2020 INSERM
% License: GPL-3.0-or-later
% Author(s): Philémon Roussel

if nargin < 1 || ~ischar(varargin{1})
    linkAx = 'xy';
else
    linkAx = lower(varargin{1});
end

if nargin < 2
    fig_handle = gcf;
else
    fig_handle = varargin{2};
end

x = findobj(fig_handle, 'Type', 'axes', 'Visible', 'on');
try
    linkaxes(x, linkAx)
catch ME
    disp(ME.message)
end
