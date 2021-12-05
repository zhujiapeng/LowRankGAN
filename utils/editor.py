# python 3.7
"""Utility functions for image editing from latent space."""

import numpy as np

__all__ = [
    'parse_indices', 'manipulate_codes'
]


def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    If the input is a list or tuple, this function has no effect.

    The input can also be a string, which is either a comma separated list of
    numbers 'a, b, c', or a dash separated range 'a - c'. Space in the string will
    be ignored.

    Args:
      obj: The input object to parse indices from.
      min_val: If not `None`, this function will check that all indices are equal
        to or larger than this value. (default: None)
      max_val: If not `None`, this function will check that all indices are equal
        to or smaller than this field. (default: None)

    Returns:
      A list of integers.

    Raises:
      If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
          numbers = list(map(int, split.split('-')))
          if len(numbers) == 1:
              indices.append(numbers[0])
          elif len(numbers) == 2:
              indices.extend(list(range(numbers[0], numbers[1] + 1)))
    else:
        raise ValueError(f'Invalid type of input: {type(obj)}!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices


def manipulate_codes(latent_code,
                     boundary,
                     layer_index=None,
                     start_distance=-4.0,
                     end_distance=4.0,
                     steps=7):
    """Manipulates the given latent code with respect to a particular boundary.
    Basically, this function takes a latent code and a boundary as inputs, and
    outputs a collection of manipulated latent codes.
    Args:
        latent_code: The input latent code for manipulation.
        boundary: The semantic boundary as reference.
        layer_index: The layers to be manipulated.
        start_distance: The distance to the boundary where the manipulation starts.
          (default: -4.0)
        end_distance: The distance to the boundary where the manipulation ends.
          (default: 4.0)
        steps: Number of steps to move the latent code from start position to end
          position. (default: 7)
    Returns:
        The manipulated codes.
    """
    assert (len(latent_code.shape) == 3 and len(boundary.shape) == 3 and
            latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
            latent_code.shape[1] == boundary.shape[1])
    linspace = np.linspace(start_distance, end_distance, steps)
    linspace = linspace.reshape([-1, 1, 1]).astype(np.float32)
    inter_code = linspace * boundary
    is_manipulatable = np.zeros(inter_code.shape, dtype=bool)
    is_manipulatable[:, layer_index, :] = True
    mani_code = np.where(is_manipulatable, latent_code + inter_code, latent_code)
    return mani_code




