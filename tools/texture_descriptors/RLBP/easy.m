% Consts
INPUT_PATH = '';
OUTPUT_PATH = './features/';
OUTPUT_FILE_NAME = 'rlbp_2_8.csv';

% if you dont have a target file
% keep the TARGET_EXT as an empty string: ''
TARGET_EXT = '.png';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RLBP parameters
NEIGHBORS = 8; 
RADIUS = 2;

% Possible values for MAPPINGTYPE are
%      'u2'   for uniform LBP
%      'ri'   for rotation-invariant LBP
%      'riu2' for uniform rotation-invariant LBP.
MAPPINGTYPE = 'u2';

% Possible values for MODE are
%      'h' or 'hist'  to get a histogram of LBP codes
%      'nh'           to get a normalized histogram
MODE = 'nh';
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
        
        mapping = getmapping(NEIGHBORS, MAPPINGTYPE);
        histogram = rlbp(img, RADIUS, NEIGHBORS, mapping, MODE);
        
        fprintf(out_file, '%f, ', histogram);
        fprintf(out_file, '%s\n', files_list(i).name);

        clear file_path img mapping histogram;
    end
end
toc

fclose(out_file);