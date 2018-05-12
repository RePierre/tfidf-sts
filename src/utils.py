import math
import numpy as np

NUM_CATEGORIES = 6              # STS scores are between 0 and 5


def matrix_to_input(matrix):
    num_rows, num_columns = matrix.shape
    result = []
    for i in range(int(num_rows / 2)):
        t1 = matrix[2 * i]
        t2 = matrix[2 * i + 1]
        c = np.concatenate((t1, t2), axis=0)
        result.append(c)
    result = np.asarray(result)
    return result


def scores_to_categorical(scores):
    result = []
    for score in scores:
        a = np.zeros(NUM_CATEGORIES)
        fractional, integer = math.modf(score)
        integer = int(integer)
        if fractional == 0.:
            a[integer] = 1.
        else:
            a[integer + 1] = fractional
        result.append(a)
    result = np.asarray(result)
    return result
