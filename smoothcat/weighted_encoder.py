import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from .base import TruncatedNormalEncoder

class WeightedTruncatedEncoder(TruncatedNormalEncoder):
    def __init__(self, weights=None, random_state=None):
        super().__init__(random_state=random_state)
        self.weights = weights  # dict 형태: {category: weight, ...}

    def fit(self, cat_series):
        if not isinstance(cat_series, pd.Series):
            raise ValueError("Input must be a pandas Series.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        value_counts = cat_series.value_counts(normalize=True).sort_values(ascending=False)

        if self.weights:
            # 기존 확률 대신 weights에 기반해 재조정
            valid_keys = [k for k in self.weights.keys() if k in value_counts.index]
            weight_values = np.array([self.weights[k] for k in valid_keys], dtype=float)
            weight_values /= weight_values.sum()
            self.categories_ = valid_keys
            probs = weight_values.cumsum()
        else:
            self.categories_ = value_counts.index.tolist()
            probs = value_counts.values.cumsum()

        self.probs_ = np.insert(probs, 0, 0.0)

def encode_category_with_weights(cat_series, weights=None, random_state=None):
    encoder = WeightedTruncatedEncoder(weights=weights, random_state=random_state)
    encoder.fit(cat_series)
    return encoder.transform(cat_series), encoder.categories_, encoder.probs_

def get_category_probabilities(cat_series, normalize=True):
    """
    범주형 Series의 레벨별 확률(또는 카운트)을 반환합니다.

    Parameters:
        cat_series : pandas Series
        normalize : True면 확률, False면 개수 반환

    Returns:
        pandas.Series : level 별 확률 또는 count (내림차순 정렬)
    """
    return cat_series.value_counts(normalize=normalize).sort_values(ascending=False)

