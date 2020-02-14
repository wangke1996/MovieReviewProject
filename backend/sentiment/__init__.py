from backend.sentiment import model_CNN, model_MNKG
import sys

sys.modules['model_CNN'] = model_CNN
sys.modules['model_MNKG'] = model_MNKG
from .preprocess import WordSet, WordEmbedding, KnowledgeBase