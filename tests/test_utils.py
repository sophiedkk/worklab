from worklab.utils import find_nearest

import numpy as np


def test_find_nearest():
    array = np.arange(100)

    nearest_index = find_nearest(array, 25.3, index=True)
    nearest_value = find_nearest(array, 25.3, index=False)

    assert nearest_value == 25
    assert nearest_index == 25
