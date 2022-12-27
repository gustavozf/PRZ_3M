import numpy as np

class KittlerCombinations:
    methods = {
        'prod': np.prod,
        'sum': np.sum,
        'max': np.amax,
        'min': np.amin,
        'median': np.median,
    }

    @staticmethod
    def combine(rule: str, classifiers: list):
        assert rule in KittlerCombinations.methods

        # get the predictions
        curr_pred = KittlerCombinations.methods[rule](classifiers, axis=0)

        
        pred_sum = np.sum(curr_pred, axis=1)
        pred_sum[pred_sum==0] = 1
        final_pred = (curr_pred.T / pred_sum).T

        return final_pred, np.argmax(final_pred, axis=1)