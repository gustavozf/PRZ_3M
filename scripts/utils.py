import os

def walk(fpath: str, target_file_ext: str=None):    
    files_list = [
        os.path.join(root, _file)
        for root, _, files in os.walk(fpath)
        for _file in files
        if not target_file_ext or _file.endswith(target_file_ext)
    ]

    return files_list, len(files_list)

def split_path(fpath: str):
    out_list = []
    _, fpath = os.path.splitdrive(fpath)
    
    fpath, tail = os.path.split(fpath)
    while tail:
        out_list = [tail] + out_list
        fpath, tail = os.path.split(fpath)

    return out_list