from typing import Generator, List, NamedTuple

import numpy as np


def find_end_first_consecutive_true(arr: np.ndarray) -> int:
    """Function to find the end position of the first consecutive sequence of True in an array."""
    if not arr[0]:
        return 0

    prog = np.cumsum(arr)
    if prog[-1] == len(arr):
        return len(arr)

    true_locs = np.where(prog[:-1:] == prog[1::])[0]

    return true_locs[0] + 1


def find_start_last_consecutive_true(arr: np.ndarray) -> int:
    """Function to find the start position of the last consecutive sequence of True in an array."""
    reverse = find_end_first_consecutive_true(arr[::-1])
    return len(arr) - reverse if reverse > 0 else -1


def group_consecutive_values(arr: np.ndarray, stepsize: int = 1) -> List[np.ndarray]:
    """Function to group consecutive values in an array."""
    return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)


class RepetitionTuple(NamedTuple):
    """Tuple to store information about a periodic sequence."""

    start: int
    end: int
    period: int
    times: int


def find_periodic_sequences(
    arr: np.ndarray, max_period: int, min_period: int = 1, mask_value: int = -1
) -> Generator[RepetitionTuple, None, None]:
    """Function to find periodic sequences in an array.

    This function sweeps through the array and checks for sequences of length
    [min_period, max_period] that repeat at least 3 times. To do so, it
    reshape the array into a matrix with `period` columns and checks if each
    row is equal to the previous row. Blocks of repeating rows indicates repeating
    sequences.

    Because there's no guarantee that the sequences start at the beginning of each
    row, it can only detect sequences that repeat at least 3 times. To account
    for the fact that sequences may not start at the beginning of each row (or
    end at the end of each row), we check the end of the previous row and the
    start of the next row to determine the actual start and end positions of the
    sequence.

    Args:
        arr (np.ndarray): The array to search for periodic sequences.
        max_period (int): The maximum period to check for.
        min_period (int, optional): The minimum period to check for. Defaults to 1.
        mask_value (int, optional): The value to use to pad the array. Defaults to -1.
    """
    # make sure the mask_value is not in the array
    if (arr == mask_value).sum() > 0:
        raise ValueError("`mask_value` is in the array")

    # no since we can only detect sequences that repeat at least 3 times,
    # there is no point in checking for periods greater than 1/3 of the length
    max_period = min(max_period, len(arr) // 3)

    for period in range(min_period, max_period + 1):
        # pad the array so that it can be reshaped into a matrix matching the period
        padded_arr = np.pad(arr, (0, period - (len(arr) % period)), constant_values=mask_value)
        shaped_arr = padded_arr.reshape(-1, period)

        # find rows that are equal to the previous  row; these are the possibly-periodic sequences
        is_equal_to_prev_row = shaped_arr == np.roll(shaped_arr, shift=1, axis=0)
        rows_with_period, *_ = np.where(is_equal_to_prev_row.all(axis=1))

        # no sequences found with this period
        if len(rows_with_period) == 0:
            continue

        # this finds the start and end positions of the sequences with period `period`
        where_true_consecutive = group_consecutive_values(rows_with_period)

        for sequence in where_true_consecutive:
            start_row = sequence[0]
            end_row = sequence[-1]

            # we check if any value at the end of the previous row is True, e.g.:
            #     [[False, False, True, True]
            #      [True, True, True, True]]
            # (in the case above, start offset is 2). If so, we subtract that from the
            # period to get the actual start offset.
            start_offset = find_start_last_consecutive_true(is_equal_to_prev_row[start_row - 1])
            start_offset = period - start_offset if start_offset > 0 else 0

            # same idea as above, we want to compute offset. Only difference is that
            # `find_end_first_consecutive_true` already returns the offset, so we don't
            # need to subtract from the period.
            end_offset = find_end_first_consecutive_true(is_equal_to_prev_row[end_row + 1])

            # because we are always comparing with preceding row in
            # `is_equal_to_prev_row`, we need to subtract 1 from the row number
            start_pos = (start_row - 1) * period - start_offset

            # note that the end position is exclusive
            end_pos = ((end_row + 1) * period) + end_offset

            out = RepetitionTuple(
                start=start_pos, end=end_pos, period=period, times=(end_pos - start_pos) // period
            )
            if out.times > 2:
                # cannot accurately determine the period of a sequence that repeats
                # less than 3 times with this algorithm
                yield out
