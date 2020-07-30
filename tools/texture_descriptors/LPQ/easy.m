% Consts
INPUT_PATH = '';
OUTPUT_PATH = './features/';
OUTPUT_FILE_NAME = 'lpq_7.csv';

% if you dont have a target file
% keep the TARGET_EXT as an empty string: ''
TARGET_EXT = '.png';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LPQ parameters
WIN_SIZE = 7; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creat output dir, if necessary
if ~exist(OUTPUT_PATH, 'dir')
    [status, msg] = mkdir(OUTPUT_PATH);
    
    if status
        disp(['Output directory created: ', OUTPUT_PATH]);
    else
        disp([
            'An error occurred while creating the output directory.\n';
            'Error: ', msg 
        ]);
    end
    
    clear status msg;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out_file = fopen(strcat(OUTPUT_PATH, OUTPUT_FILE_NAME),'w');

files_list = dir(fullfile(INPUT_PATH, strcat('*', TARGET_EXT)));

tic
for i = 1 : length(files_list)
    if (files_list(i).isdir == 0)
        disp(['File: ', files_list(i).name, ' (#', int2str(i), ')']);
        
        file_path = fullfile(INPUT_PATH, files_list(i).name);
        
        img = imread(file_path);
        
        histogram = lpq(img, WIN_SIZE);
        
        fprintf(out_file, '%f, ', histogram);
        fprintf(out_file, '%s\n', files_list(i).name);

        clear file_path img histogram;
    end
end
toc

fclose(out_file);