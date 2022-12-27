import sys, os, re, platform
# The following line is needed if the lib is being accessed localy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PRZ_3M/')))

from prz.resources.data.image_sample import ImageSample
from prz.resources.data.io import DataIO
from prz.definitions.configs import Configs

DATA_MAP_FILE = '../Work/mapped_dataset.txt'
OUTPUT_PATH = '../outputs/dataset_gs/'

FILE_EXT = {
    'c1.JPG': 'ENC',
    'c2.JPG': 'EGC',
    '(c1+c2).JPG': 'MRG',
}

count = {i:0 for i in FILE_EXT.values()}
anim_map = {}

def out_map_to_file():
    global count
    global anim_map

    with open(f'{OUTPUT_PATH}/out_map.txt', 'w') as out_file:
        out_file.write('> Total number of files:\n')
        for ext in FILE_EXT.values():
            out_file.write(f'  |__ {ext}: {count[ext]}\n')

        out_file.write('\n> Total number of files per animal:\n')
        for anim in anim_map.keys():
            out_file.write(f'  |__ {anim}\n')

            for ext in FILE_EXT.values():
                out_file.write(f'    |__ {ext}: {len(anim_map[anim][ext])}\n')


def img_to_gs(img_path):
    rgb = ImageSample.read(img_path)
    hsv = ImageSample.cvt_color(
        rgb, 
        config={'from': 'RGB', 'to': 'HSV'}
    )

    return ImageSample.to_grayscale(img_src=hsv, src_color_sch='HSV')

if __name__ == '__main__':
    # Windows: '\\'; Others: '/'
    SEP = Configs.dir_sep

    if not DataIO.isDir(OUTPUT_PATH):
        DataIO.createDir(f'{OUTPUT_PATH}')


    with open(DATA_MAP_FILE) as map_file:
        possible_ext = FILE_EXT.keys()
        
        for line in map_file:
            # line[:-1].split(SEP)[6:] = sample's useful information
            # result ex.: ['7C', '9', '9_c2.JPG']
            anim_tag, img_num, img_name = line[:-1].split(SEP)[6:]

            ext = img_name.split('_')[-1]

            if not ext in possible_ext:
                continue

            img_type = FILE_EXT[ext]
            count[img_type] += 1


            if (anim_tag not in anim_map.keys()):
                anim_map[anim_tag] = {
                    i:[] for i in FILE_EXT.values()
                }

            out_name = f'{anim_tag}_{img_num}_{img_type}.png'
            
            anim_map[anim_tag][img_type].append(out_name)

            print(count[img_type], line)
            img = img_to_gs(line[:-1])
            ImageSample.write(img=img, 
                              out_path=OUTPUT_PATH, 
                              file_name=out_name)
    
    print('Wrinting output to file...')
    out_map_to_file()