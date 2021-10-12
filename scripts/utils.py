import os

def walk(fpath, target_file_ext=None):    
    files_list = [
        os.path.join(root, _file)
        for root, _, files in os.walk(fpath)
        for _file in files
        if not target_file_ext or _file.endswith(target_file_ext)
    ]

    return files_list, len(files_list)

def split_path(fpath, num_iter:int=None):
    out_list = []
    continue_cond = True

    _, fpath = os.path.splitdrive(fpath)
    while continue_cond:
        fpath, tail = os.path.split(fpath)
        out_list = [tail] + out_list 
        continue_cond = bool(fpath)
        print(fpath, continue_cond, out_list)
        input()
    return out_list