from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import numpy as np


def calculate_score(predictions, labels, metric='ACC', min_value=1, max_value=5):
    outputs = list(map(round, predictions))
    outputs = np.clip(outputs, min_value, max_value)
    if metric == 'MSE':
        return mean_squared_error(labels, outputs)
    elif metric == 'ACC':
        return accuracy_score(labels, outputs)
    elif metric == 'micro_F1':
        return f1_score(labels, outputs, labels=list(range(min_value, max_value + 1)), average='micro')
    elif metric == 'macro_F1':
        return f1_score(labels, outputs, labels=list(range(min_value, max_value + 1)), average='macro')
    elif metric == 'weighted_F1':
        return f1_score(labels, outputs, labels=list(range(min_value, max_value + 1)), average='weighted')
