"""Misc procedural utilities for ecoshard usage."""
import numpy

def scrub_invalid_values(base_array, nodata, new_nodata):
    """Remove infinate or NaN values from array and replace with nodata.

    Args:
        base_array (numpy.ndarray): numpy array, could contain invalid values
        nodata (numpy.dtype): expected nodata value in base_array
        new_nodata (numpy.dtype): desired target nodata value.

    Returns:
        copy of base_array but invalid values and nodata are repalced with
            ``new_nodata``.

    """
    result = numpy.copy(base_array)
    invalid_mask = (
        ~numpy.isfinite(base_array) | numpy.isclose(result, nodata))
    result[invalid_mask] = new_nodata
    return result
