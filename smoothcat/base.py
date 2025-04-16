import numpy as np
import pandas as pd
from scipy.stats import truncnorm

'''
Example:

import pandas as pd

# 테스트 데이터
df = pd.DataFrame({'label': ['A', 'B', 'C', 'A', 'B', 'A']})

# for reproducibility
RANDOM_SEED = 42 

# 기본 인코딩
encoded1, cats1, probs1 = encode_category(df['label'], random_state=RANDOM_SEED)
print("Base Encoded:", encoded1)

# 가우시안 노이즈 추가 인코딩
encoded2, cats2, probs2 = encode_category_with_noise(df['label'], noise_var=0.04, random_state=RANDOM_SEED)
print("Noisy Encoded:", encoded2)
'''

class TruncatedNormalEncoder:
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.categories_ = None
        self.probs_ = None

    def fit(self, cat_series):
        if not isinstance(cat_series, pd.Series):
            raise ValueError("Input must be a pandas Series.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        value_counts = cat_series.value_counts(normalize=True).sort_values(ascending=False)
        self.categories_ = value_counts.index.tolist()
        probs = value_counts.values.cumsum()
        self.probs_ = np.insert(probs, 0, 0.0)

    def transform(self, cat_series):
        if self.categories_ is None or self.probs_ is None:
            raise RuntimeError("You must call fit() before transform().")

        encoded_vals = []
        for val in cat_series:
            if val not in self.categories_:
                encoded_vals.append(np.nan)
                continue

            idx = self.categories_.index(val)
            a, b = self.probs_[idx], self.probs_[idx + 1]
            mu = (a + b) / 2
            sigma = (b - a) / 6 if (b - a) > 0 else 1e-6

            sample = truncnorm.rvs((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
            encoded_vals.append(sample)

        return np.array(encoded_vals)

    def inverse_transform(self, encoded_vals):
        if self.categories_ is None or self.probs_ is None:
            raise RuntimeError("You must call fit() before inverse_transform().")

        decoded = []
        for val in encoded_vals:
            if val == 1.0:
                val -= 1e-8
            for i in range(len(self.categories_)):
                if self.probs_[i] <= val < self.probs_[i + 1]:
                    decoded.append(self.categories_[i])
                    break
            else:
                decoded.append(None)
        return decoded

def encode_category(cat_series, random_state=None):
    encoder = TruncatedNormalEncoder(random_state=random_state)
    encoder.fit(cat_series)
    return encoder.transform(cat_series), encoder.categories_, encoder.probs_
