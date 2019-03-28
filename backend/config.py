import platform
import os


class Config:
    def __init__(self):
        if platform.system() == 'Windows':
            self.data_path = os.path.abspath('./MovieData')
        else:
            self.data_path = os.path.abspath('/data/wangke/MovieProject/MovieData')


config = Config()
