import os

from prz.definitions.strings import Strings

class DataSearch:
    @staticmethod
    def bfs(fpath, target_file_ext=None):
        assert os.path.exists(fpath), Strings.no_path
        assert os.path.isdir(fpath), Strings.path_not_dir

        found_files = []
        unseen_paths = [os.path.join(fpath, i) for i in os.listdir(fpath)]

        while(len(unseen_paths) > 0):
            cur_path = unseen_paths.pop()
            is_target_file = (
                not target_file_ext or cur_path.endswith(target_file_ext)
            )

            if (os.path.isfile(cur_path) and is_target_file):
                found_files.append(cur_path)
            elif os.path.isdir(cur_path):
                for sub_dir in os.listdir(cur_path):
                    unseen_paths.append(os.path.join(cur_path, sub_dir))

        return found_files, len(found_files)

    @staticmethod
    def walk(fpath, target_file_ext=None):
        assert os.path.exists(fpath), Strings.no_path
        assert os.path.isdir(fpath), Strings.path_not_dir
    
        files_list = [
            os.path.join(root, _file)
            for root, _, files in os.walk(fpath)
            for _file in files
            if not target_file_ext or _file.endswith(target_file_ext)
        ]

        return files_list, len(files_list)
