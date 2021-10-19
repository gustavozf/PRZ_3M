class EgczDataset(object):
    def __init__(self, data: list, labels: list, groups: list):
        self.data = data
        self.labels = labels
        self.groups = groups

    @staticmethod
    def from_csv(fpath: str):
        # TODO: read from csv and split the data
        data = []
        labels = []
        groups = []

        return EgczDataset(data, labels, groups)

    def leave_one_out_kfold():
        # TODO: return the folds' indexes
        return []