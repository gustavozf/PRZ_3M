% Consts
INPUT_PATH = '';
OUTPUT_PATH = './features/';
OUTPUT_FILE_NAME = 'eqp_2_18.csv';

% if you dont have a target file
% keep the TARGET_EXT as an empty string: ''
TARGET_EXT = '.png';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
NEIGHBORS = 18; 
RADIUS = 2;

% loci: 
%   'el'    ellipse; 
%   'ip'    hyperbole
%   'par'   parabola 
%   'sp'    spiral 
%   'ci'    circle
LOCI = 'el';

% Threshold Encoding
% 'binary'   =>   [ ]
% 'ternary'  =>   [ T1 ]
% 'quinary'  =>   [ T1, T2 ]
THRESHOLD = [2, 5];

% p1 and p2 are the parameters of the loci of points (see Table 1), 
% e.g. p1 is the radius if you use �circle�; p1 and p2 are the semimajor 
% and semiminor axis lengths if you use an ellipse
P1 = 3;
P2 = 2;

% Rotating angle (in degrees)
% 0, 45, 90, 135
BETA = 0;

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
        
        mapping = getmapping02(NEIGHBORS, MAPPINGTYPE);
        
        histogram = OurLBP(img, THRESHOLD, NEIGHBORS, LOCI, ...
                            P1, P2, BETA, mapping, MODE);

        fprintf(out_file, '%f, ', histogram);
        fprintf(out_file, '%s\n', files_list(i).name);

        clear file_path img mapping histogram;
    end
end
toc

fclose(out_file);