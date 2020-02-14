import os


class Config:
    def __init__(self):
        self.data_path = os.path.abspath('..//MovieData')


config = Config()
