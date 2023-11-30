%% EXAMPLE OF DATA FORMATTING

% This script shows how to create files should be MAT-files with a
% defined structure called RecordingMatfiles (see files in 'Test data'
% folder).

%% Add path to class and functions
addpath(genpath('../Toolbox/'));

%% Output file path



%% Prepare fake recording data

fs = 512; % sampling frequency (Hz)
output_path_root = '/Users/james/contamination/final_codes/';
DIR = {'NY742/'};
for j = 1:length(DIR)
        if exist([output_path_root, 'data/mat/',DIR{j},'/test_ecog_data.mat'], 'file') == 0
            disp( 'file not exist')
            try
                load([output_path_root, 'data/',DIR{j},'/test_ecog_data.mat']);
                output_path = [output_path_root, 'data/mat/',DIR{j},'/test_ecog_data.mat'];
                createRecordingMatfile(output_path, values, fs)
                load([output_path_root, 'data/',DIR{j},'/test_audio_data.mat']);
                output_path = [output_path_root, 'data/mat/',DIR{j},'/test_audio_data.mat'];
                createRecordingMatfile(output_path, values, fs)
            catch exception
                disp([DIR{j} ,'failed\n'])
                continue; 
            end
        else
            disp('file exits continue')
        end
end


