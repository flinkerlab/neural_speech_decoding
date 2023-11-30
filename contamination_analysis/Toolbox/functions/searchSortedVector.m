function indices = searchSortedVector(x, values, condition)
% indices = searchSortedVector(x, values, condition)
%
% Binary search in sorted vector x.
%
% Several times faster than logical operations in the case of large
% vectors (>10000 elements).
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

if isempty(x)
    indices = {};
    return
end

if ~exist('condition', 'var') || isempty(condition)
    condition = 'exact';
elseif ~(isstring(condition) || ischar(condition))
    tol = condition;
    condition = 'near';
end

values_nb = length(values);
indices = cell(values_nb, 1);

for v_i = 1:values_nb
    
    val = values(v_i);
    
    % handling particular cases
    switch condition
        case { 'inf', 'infeg' }
            if val < x(1) || ( val == x(1) && strcmp( condition, 'inf' ) )
                continue
            elseif val > x(end) || ( val == x(end) && strcmp( condition, 'infeg' ) )
                indices{v_i} = numel(x);
                continue
            end
            
        case { 'sup', 'supeg' }
            if val > x(end) || ( val == x(end) && strcmp( condition, 'sup' ) )
                continue
            elseif val < x(1) || ( val == x(1) && strcmp( condition, 'supeg' ) )
                indices{v_i} = 1;
                continue
            end
            
        case 'exact'
            if val < x(1) || val > x(end)
                continue
            end
            
        case 'near'
            if (val < x(1) - tol) || (val > x(end) + tol)
                continue
            end
    end
    
    % initialization
    idx_inf = 1;
    idx_sup = length(x);
    
    % binary search
    switch condition
        case { 'inf', 'supeg', 'exact', 'near' }
            
            while idx_inf + 1 < idx_sup
                halfway_idx = ( floor( ( idx_inf + idx_sup ) / 2 ) );
                if x( halfway_idx ) < val
                    idx_inf = halfway_idx;
                else
                    idx_sup = halfway_idx;
                end
            end
            
        case { 'infeg', 'sup' }
            
            while idx_inf + 1 < idx_sup
                halfway_idx = ( floor( ( idx_inf + idx_sup ) / 2 ) );
                if x( halfway_idx ) <= val
                    idx_inf = halfway_idx;
                else
                    idx_sup = halfway_idx;
                end
            end
            
    end
    
    % output
    switch condition
        case {'inf', 'infeg'}
            idx = idx_inf;
            
        case {'sup', 'supeg'}
            idx = idx_sup;
            
        case 'exact'
            match_idx = find( x([ idx_inf idx_sup ]) == val, 1 );
            if isempty( match_idx )
                idx = [];
            elseif match_idx == 1
                idx = idx_inf;
            else
                idx = idx_sup;
            end
            
        case 'near'
            [ min_dist, min_idx ] = min( x([ idx_inf idx_sup ]) - val );
            if min_dist > tol
                idx = [];
            elseif min_idx == 1
                idx = idx_inf;
            else
                idx = idx_sup;
            end
    end
    
    indices{v_i} = idx;
    
end

end