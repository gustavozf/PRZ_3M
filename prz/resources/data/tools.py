import os

from prz.resources.data.io import DataIO
from prz.resources.data.search import DataSearch


class DataTools:
    @staticmethod
    def create_data_map(
            dpath,
            target_file_ext=None,
            output_config={
                'to_file': True,
                'file_name': 'mapped_dataset.txt',
                'path': './outputs/maps/',
            },):

        found_files, _ = DataSearch.bfs(dpath, target_file_ext=target_file_ext)

        if (output_config['to_file']):
            DataIO.createDirIfNotExists(output_config['path'])
            
            with open(
                    output_config['path'] + output_config['file_name'], 'w',) as output_file:

                for found_file in found_files:
                    output_file.write(found_file + '\n')
        else:
            print(found_files)
