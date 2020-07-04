from prz.definitions.strings import Strings
from prz.definitions.configs import Configs
from prz.resources.data.io import DataIO


class DataSearch:
    @staticmethod
    def bfs(fpath, target_file_ext=None):
        assert DataIO.pathExists(fpath), Strings.noPath
        assert DataIO.isDir(fpath), Strings.pathNotDir

        def check_file_ext(
            fname): return True if not target_file_ext else fname.endswith(target_file_ext)

        sep = Configs.dirSep
        fpath += sep if not fpath.endswith(sep) else ''
        found_files = []
        unseen_paths = [f'{fpath}{i}' for i in DataIO.listDir(fpath)]

        while(len(unseen_paths) > 0):
            cur_path = unseen_paths.pop()
            is_target_file = check_file_ext(cur_path)

            if (DataIO.isFile(cur_path) and is_target_file):
                found_files.append(cur_path)
            elif DataIO.isDir(cur_path):
                for sub_dir in DataIO.listDir(cur_path):
                    unseen_paths.append(f'{cur_path}{sep}{sub_dir}')

        return found_files, len(found_files)
