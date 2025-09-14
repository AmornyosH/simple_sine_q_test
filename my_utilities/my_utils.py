import numpy as np
import torch

def rangeConversion(input, new_min, new_max, old_min=-1, old_max=1) -> float:
    '''
    Range Conversion Method (Mapping from old range to a new range) \\
    This method will map the value in the defined old range (default = [-1, 1])
    in to a new defined range (no default). 
    Formula = new_min + ((input - old_min) * (new_max - new_min) / (old_max - old_min))
    
    Input Arguments: sample:Any, new_min, new_max, old_min=-1, old_max=1
    Return: sample in new range (sample \in [new_min, new_max])
    '''
    # Default of old range is [-1, 1]
    # Define parameters
    # old_length = (old_max - old_min) / 2  # length between old mid point and old marginal points.
    # old_new_diff = new_max - (old_max/old_length)  # the difference between old marginal point and new marginal point.
    # return tf.clip_by_value(new_min + ((input - old_min) * (new_max - new_min) / (old_max - old_min)), 
    #                         clip_value_min=new_min, 
    #                         clip_value_max=new_max)
    return torch.clip(new_min + ((input - old_min) * (new_max - new_min) / (old_max - old_min)), 
                            min=new_min, 
                            max=new_max)


def klDivergence(input_1, input_2):
    '''
    KL-Divergence Computation Method \\
    This method will return the Kullback-Leibler Divergence (D_KL) between input_1 and input_2.\\
    Remind that D_KL between input_1 and input_2 is not equal to D_KL between input_2 and input_1.

    Input Arguments: input_1:Any, input_2:Any (Numpy Format)
    Return: D_KL(input_1 || input_2) (Numpy Format)
    '''
    # old_policy = tf.gather_nd(old_policy, indices=slice_indices.astype(int))
    # return tf.reduce_mean(-tf.reduce_sum(old_policy * tf.math.log(new_policy/old_policy), axis=1)).numpy()
    kl = -np.sum(input_1 * np.log(input_2/input_1)) / len(input_1)
    return kl

# Sinusoidal Position Encoding (Same as the original Transformer.)
def sinPositionEncoding(seq_len, dim, N=10000):
    output = np.zeros([seq_len, dim], dtype=float)
    for k in range(seq_len):
        for i in np.arange(int(dim/2)):
            denominator = np.power(N, 2*i/dim)
            output[k, 2*i] = np.sin(k/denominator)
            output[k, 2*i+1] = np.cos(k/denominator)
    return output

# Positional encoding method (My implementation)
def sinusoidalEncoding(input, seq_len):
    t_space = np.linspace(start=-1, stop=1, num=seq_len, dtype=float)  # [-1, -0.5, 0, 0.5, 1]
    return np.sin(2 * np.pi * t_space[input-1])  # sin(2 * pi * t)