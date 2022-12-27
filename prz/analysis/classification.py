import numpy as np

def accuracy_from_confusion_matrix(confusion_matrix: np.array):
    length = range(len(confusion_matrix))
    sum_main_diagonal = float(sum([confusion_matrix[i][i] for i in length]))
    sum_matrix = float(
        sum([confusion_matrix[i][j] for i in length for j in length])
    )

    return sum_main_diagonal / sum_matrix

def accuracy_from_labels(y_label: np.array, y_pred: np.array):
    return np.mean(np.equal(y_label, y_pred, dtype=np.int32))
    
def labels_from_confusion_matrix(confusion_matrix: np.array):
    length = range(len(confusion_matrix))
    labels = []
    predicts = []

    for clas in length:
        for pred in length:
            count = confusion_matrix[clas][pred]
            labels.extend([clas for _ in range(count)])
            predicts.extend([pred for _ in range(count)])

    return labels, predicts

def get_stats_report(data: np.array):
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std_dev': np.std(data),
        'variance': np.var(data),
    }