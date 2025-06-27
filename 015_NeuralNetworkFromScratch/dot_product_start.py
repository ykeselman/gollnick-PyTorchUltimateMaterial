# %% packages
import numpy as np
# %%
X = [0, 1]
w1 = [2, 3]
w2 = [0.4, 1.8]

# %% Question: which weight is more similar to input data X
print('w1' if np.dot(X, w1) > np.dot(X, w2) else 'w2')
