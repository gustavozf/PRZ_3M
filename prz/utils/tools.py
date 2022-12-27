import os

from prz.utils.search import DataSearch

DEFAULT_OUT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'outputs', 'maps'
    )
)

class DataTools:
    @staticmethod
    def create_data_map(
            dpath,
            target_file_ext=None,
            to_file: bool=True,
            out_path: str=DEFAULT_OUT_PATH,
            out_file_name: str='mapped_dataset.txt',
        ):

        found_files, _ = DataSearch.bfs(dpath, target_file_ext=target_file_ext)

        if (to_file):
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            output_file = os.path.join(out_path, out_file_name)
            with open(output_file, 'w',) as output_file:
                output_file.write('\n'.join(found_files))                    
        else:
            print(found_files)
