import numpy as np
from scipy.stats import truncnorm
from .base import TruncatedNormalEncoder
'''
parameter
    - noise_var : std of gaussian noise
'''
class TruncatedNormalEncoderWithNoise(TruncatedNormalEncoder):
    def __init__(self, noise_var=0.0001, random_state=None):
        super().__init__(random_state=random_state)
        self.noise_std = np.sqrt(noise_var)

    def transform(self, cat_series):
        if self.categories_ is None or self.probs_ is None:
            raise RuntimeError("You must call fit() before transform().")

        if self.random_state is not None:
            np.random.seed(self.random_state)

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
            noisy_sample = np.clip(sample + np.random.normal(0, self.noise_std), 0.0, 1.0)
            encoded_vals.append(noisy_sample)

        return np.array(encoded_vals)

def encode_category_with_noise(cat_series, noise_var=0.0001, random_state=None):
    encoder = TruncatedNormalEncoderWithNoise(noise_var=noise_var, random_state=random_state)
    encoder.fit(cat_series)
    return encoder.transform(cat_series), encoder.categories_, encoder.probs_
