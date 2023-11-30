function varargout = setFigure(figure_name)
% [figure_handle, new_bool] = setFigure(figure_name)
%
% Allow to call a figure by name to set it as current figure. 
%
% 'new_bool' is false if a figure with the name 'figure_name' already
% exists.
% Useful for scripts when you have different figures and want to be able to
% recompute and overwrite each of them individually.
%
% Example:
%  setFigure('Display data')
%  plot(...)
%  setFigure('Results)
%  plot(...)
%  setFigure('Display data')
%  hold on 
%  plot(...)
%  hold off
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

figure_handles = findobj('Type', 'figure');

if isempty(figure_handles)
    figure_handle = [];
else
    figure_names = {figure_handles.Name};
    figure_handle = figure_handles(ismember(figure_names, figure_name));
end

if isempty(figure_handle)
    figure_handle = figure('Name', figure_name);
    new_bool = true;
else
    set(0, 'CurrentFigure', figure_handle)
    new_bool = false;
end

if nargout > 0
    varargout{1} = figure_handle;
end

if nargout > 1
    varargout{2} = new_bool;
end

end