import os

from prz.definitions.strings import Strings


class DataIO:

    @staticmethod
    def isFile(fpath): return os.path.isfile(fpath)

    @staticmethod
    def isDir(fpath): return os.path.isdir(fpath)

    @staticmethod
    def pathExists(fpath): return os.path.exists(fpath)

    @staticmethod
    def absPath(fpath):
        assert DataIO.pathExists(fpath), Strings.no_path
        assert DataIO.isDir(fpath), Strings.path_not_dir

        return os.path.abspath(fpath)

    @staticmethod
    def listDir(fpath):
        assert DataIO.pathExists(fpath), Strings.no_path
        assert DataIO.isDir(fpath), Strings.path_not_dir

        return os.listdir(fpath)

    @staticmethod
    def createDir(fpath):
        assert not DataIO.pathExists(fpath), Strings.path_exists

        try:
            os.makedirs(fpath)
        except OSError:
            print(Strings.dir_creation_failed % fpath)

    @staticmethod
    def createDirIfNotExists(fpath):
        try:
            if (not DataIO.pathExists(fpath)):
                os.makedirs(fpath)
        except OSError:
            print(Strings.dir_creation_failed % fpath)

    @staticmethod
    def removeDir(fpath):
        assert DataIO.pathExists(fpath), Strings.no_path

        try:
            os.makedirs(fpath)
        except OSError:
            print(Strings.dir_remove_failed % fpath)
