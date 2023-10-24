import numpy as np

def unfairness(data1, data2):
    """
    compute the unfairness of two populations
    """
    #scs.wasserstein_distance()
    probs = np.linspace(0.03, 0.97, num=97) # That way we avoid issues with some tail observations
    eqf1 = np.quantile(data1, probs)
    eqf2 = np.quantile(data2, probs)
    unfair_value = np.max(np.abs(eqf1-eqf2))
    return unfair_value