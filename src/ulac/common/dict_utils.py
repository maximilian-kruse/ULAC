import operator
from collections.abc import Iterable, Iterator
from functools import reduce


# ==================================================================================================
def create_empty_dict_from_keys(input_dict: dict) -> dict:
    def _create_empty_dict(d: dict) -> dict:
        if isinstance(d, dict):
            return {key: _create_empty_dict(value) for key, value in d.items()}
        return None

    return _create_empty_dict(input_dict)


# --------------------------------------------------------------------------------------------------
def get_dict_entry(key_sequence: Iterable[str], data_dict: dict) -> object:
    try:
        value = reduce(operator.getitem, key_sequence, data_dict)
    except KeyError as e:
        raise KeyError(f"Key sequence {key_sequence} not found") from e
    return value


# --------------------------------------------------------------------------------------------------
def set_dict_entry(key_sequence: Iterable[str], data_dict: dict, value: object) -> None:
    try:
        reduce(operator.getitem, key_sequence[:-1], data_dict)[key_sequence[-1]] = value
    except KeyError as e:
        raise KeyError(f"Key sequence {key_sequence} not found") from e


# --------------------------------------------------------------------------------------------------
def nested_dict_keys(
    d: dict[str, dict], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    for key, value in d.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from nested_dict_keys(value, path)
        else:
            yield path
