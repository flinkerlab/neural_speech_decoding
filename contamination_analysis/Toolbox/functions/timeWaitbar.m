classdef timeWaitbar
    % timeWaitbar class
    %
    % Waitbar showing elapsed time and estimated remaining time.
    %
    % Construct: obj = timeWaitbar(info_text)
    % Update: updateTimeWaitbar(obj, x, info_text)
    % Close: close(obj)
    %
    % Copyright 2020 INSERM
    % Licence: GPL-3.0-or-later
    % Author(s): Philémon Roussel
    
    properties
        waitbar_handle
        
        start_t
        
        info_text
        
    end
    
    methods
        function obj = timeWaitbar(info_text)
            obj.start_t = tic;
            
            obj.info_text = char(info_text);
            
            obj.waitbar_handle = waitbar(0, obj.info_text);
            set(obj.waitbar_handle, 'HandleVisibility', 'on')
        end
        
        function updateTimeWaitbar(obj, x, info_text)
            
            if exist('info_text', 'var')
                obj.info_text = info_text;
            end
            
            elapsed_t = toc(obj.start_t);
            
            remaining_t = elapsed_t / x * (1 - x);
            
            full_text = {obj.info_text,...
                strcat("Elapsed: ", datestr(elapsed_t / 86400, 'HH:MM:SS'),...
                " | ",...
                "Remaining: ", datestr(remaining_t / 86400, 'HH:MM:SS'))};
            
            waitbar(x, obj.waitbar_handle, full_text)
            
        end
        
        function close(obj)
            close(obj.waitbar_handle)
            
        end
        
    end
end

