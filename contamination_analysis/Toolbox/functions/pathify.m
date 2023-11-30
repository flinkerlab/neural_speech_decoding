function path = pathify(path, instruction)
% Return path in standard format and optionally execute initialization
% operations
%
%   Input path target is considered as a directory if it ends with '/' or
%   '\' and as a file otherwise. The instructions are interpreted as
%   follow:
%   - Directory
%       - delete: remove directory if it exists
%       - create: create directory if it does not exist
%       - replace: create empty directory (delete existing content)
%       - check: check if directory exists and return warning if not
%   - File
%       - delete: remove file if it exists
%       - create: create containing directory if it does not exist
%       - replace: create empty containing directory (delete existing content)
%       - check: check if file exists and return warning if not
%
% Copyright 2020 INSERM
% Licence: GPL-3.0-or-later
% Author(s): Philémon Roussel

if nargin < 2
    instruction = 'none';
end

% transform relative path into absolute
path = GetFullPath(char(path));

% replace '\' or '//' with '/'
path = strrep(path, '\', '/');
path = strrep(path, '//', '/');

if strcmp(path(end), '/')
    
    type = 'dir';
    
else
    
    type = 'file';
    
end


switch type
    
    case 'dir'
        
        switch instruction
            case 'delete'
                if exist(path, type)
                    rmdir(path);
                end
                
            case 'create'
                if ~exist(path, type)
                    mkdir(path);
                end
                
            case 'replace'
                if exist(path, type)
                    rmdir(path, 's');
                end
                mkdir(path);
                
            case 'check'
                if ~exist(path, 'dir')
                    disp(['Warning: Folder ' path ' does not exist.'])
                end
            
            case 'none'
                % nothing
                
        end
        
        
        
    case 'file'
        
        [folder_path, ~, ~] = fileparts( path );
        
        switch instruction
            case 'delete'
                if exist( path, type )
                    delete( path );
                end
                
            case 'create'
                if ~exist(folder_path, 'dir')
                    mkdir(folder_path);
                end
                
            case 'replace'
                if exist( path, type )
                    delete( path );
                end
                if ~exist( folder_path, 'dir' )
                    mkdir( folder_path );
                end
                
            case 'check'
                if ~exist( path, type )
                    disp( [ 'Warning: File ' path ' does not exist.' ] )
                end
				
			case 'none'
                % nothing
				
        end
end


