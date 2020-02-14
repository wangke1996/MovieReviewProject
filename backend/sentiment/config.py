import pathlib
import os

print(pathlib.Path(__file__).parent.absolute())
print(pathlib.Path(__file__).parent)
print(pathlib.Path())
print(pathlib.Path().absolute())


class Config(object):
    def __init__(self):
        self.work_path = pathlib.Path(__file__).parent.absolute()

        self.data_path = os.path.join(self.work_path, 'data')
        self.word_set_path = os.path.join(self.data_path, 'word_set.pkl')
        self.word_embedding_path = os.path.join(self.data_path, 'word_embedding.pkl')
        self.knowledge_base_path = os.path.join(self.data_path, 'knowledge_base.pkl')

        self.model_path = os.path.join(self.work_path, 'model')
        self.model_PAIR_path = os.path.join(self.model_path, 'model_CNN_PAIR')
        self.model_SENT_path = os.path.join(self.model_path, 'model_MNKG_SENT')


CONFIG = Config()
