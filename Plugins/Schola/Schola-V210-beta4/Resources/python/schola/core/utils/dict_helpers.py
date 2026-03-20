# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Utility Functions and Classes for manipulating dictionaries.
"""
import itertools
from typing import TypeVar, Iterator, Dict, Callable, Tuple


V = TypeVar("V")
Y = TypeVar("Y")
K = TypeVar("K")
Z = TypeVar("Z")

# Define a recursive type for a nested dictionary and nested dictionary iterator
NestedIterator = Iterator[Tuple[K, V | "NestedIterator[K, V]"]]
NestedDict = Dict[K, V | "NestedDict[K, V]"]


def _flatten(
    _iterator: NestedIterator[str, V], prefix: str = ""
) -> NestedIterator[str, V]:
    for key, value in _iterator:
        new_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Iterator):
            yield from _flatten(value, new_prefix)
        else:
            yield (new_prefix, value)


def _kfilter(
    func: Callable[[K], bool], _iterator: NestedIterator[K, V], any_key: bool = True
) -> NestedIterator[K, V]:

    for key, value in _iterator:
        if isinstance(value, Iterator):
            if func(key) and any_key:  # some key was true which is good enough
                yield (key, value)
            elif func(key) or any_key:  # need to keep checking keys
                # evaluate here so that we don't output empty dictionaries
                output_ = [(k, v) for k, v in _kfilter(func, value, any_key)]
                if len(output_) > 0:
                    yield (key, iter(output_))
        else:
            if func(key):
                yield (key, value)


def _map(
    func: Callable[[V], Y], _iterator: NestedIterator[K, V]
) -> NestedIterator[K, Y]:
    yield from (
        (key, _map(func, value) if isinstance(value, Iterator) else func(value))
        for key, value in _iterator
    )


def _kmap(
    func: Callable[[K], Y], _iterator: NestedIterator[K, V]
) -> NestedIterator[Y, V]:
    yield from (
        (func(key), _kmap(func, value) if isinstance(value, Iterator) else value)
        for key, value in _iterator
    )


def _unflatten(
    _iterator: NestedIterator[str, V], flat_dict: Dict[str, Z], prefix: str = ""
) -> NestedIterator[str, Z]:
    for key, value in _iterator:
        flat_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Iterator):
            yield (key, _unflatten(value, flat_dict, flat_prefix))
        else:
            yield (key, flat_dict[flat_prefix])


def _leaves(_iterator: NestedIterator[K, V]) -> Iterator[Tuple[K, V]]:
    for key, value in _iterator:
        if isinstance(value, Iterator):
            yield from _leaves(value)
        else:
            yield (key, value)


class DIterator(NestedIterator[K, V]):
    """
    Wrapper around iterators through nested dictionaries.
    """

    def __init__(self, source_dict: NestedDict[K, V]):
        self.source_dict = source_dict
        self._base_iterator = self._make_iterator(self.source_dict)
        self._iterator = self._base_iterator

    @staticmethod
    def _make_iterator(_dict):
        for key, value in _dict.items():
            if isinstance(value, dict):
                yield (key, DIterator._make_iterator(value))
            else:
                yield (key, value)

    def map(self, func: Callable[[V], Y]) -> "DIterator[K,Y]":
        self._iterator = _map(func, self._iterator)
        return self

    def kmap(self, func: Callable[[K], Y]) -> "DIterator[Y,V]":
        self._iterator = _kmap(func, self._iterator)
        return self

    def kfilter(
        self, func: Callable[[K], bool], any_key: bool = True
    ) -> "DIterator[K,V]":
        self._iterator = _kfilter(func, self._iterator, any_key)
        return self

    def flatten(self, prefix: str = "") -> "DIterator[str,V]":
        self._iterator = _flatten(self._iterator, prefix)
        return self

    def unflatten(
        self, flat_dict: Dict[str, Z], prefix: str = ""
    ) -> "DIterator[str,Z]":
        self._iterator = _unflatten(self._iterator, flat_dict, prefix)
        return self

    def chain(self, other: "DIterator[K,V]") -> "DIterator[K,V]":
        """
        Chain two DIterators together. Note this function is not robust to reused intermediate keys e.g. dict(a=dict(b=2)) and dict(a=dict(c=3)) will cause errors
        """
        self._iterator = itertools.chain(self._iterator, other._iterator)
        return self

    @staticmethod
    def _to_dict(
        _iterator: NestedIterator[K, V], prune: bool = False
    ) -> NestedDict[K, V]:
        out = {}
        for key, value in _iterator:
            if isinstance(value, Iterator):
                processed_value = DIterator._to_dict(value, prune)
                if processed_value or not prune:
                    out[key] = processed_value
            else:
                out[key] = value
        return out

    def to_dict(self, prune: bool = True) -> NestedDict[K, V]:
        return self._to_dict(self._iterator, prune)

    def __iter__(self) -> NestedIterator[K, V]:
        return self.leaves()

    def __next__(self) -> Tuple[K, V | "NestedIterator[K, V]"]:
        return next(self._iterator)

    def leaves(self) -> Iterator[Tuple[K, V]]:
        yield from _leaves(self._iterator)

    def keys(self) -> Iterator[K]:
        yield from (key for key, _ in _leaves(self._iterator))

    def values(self) -> Iterator[V]:
        yield from (value for _, value in _leaves(self._iterator))


def map_dict(func: Callable[[V], Y], input_dict: NestedDict[K, V]) -> NestedDict[K, Y]:
    return DIterator(input_dict).map(func).to_dict()


def kfilter_dict(
    func: Callable[[K], bool], input_dict: NestedDict[K, V], any_key: bool = True
) -> NestedDict[K, V]:
    return DIterator(input_dict).kfilter(func, any_key).to_dict()


def flatten_dict(input_dict: NestedDict[K, V], prefix: str = "") -> Dict[str, V]:
    return DIterator(input_dict).flatten(prefix).to_dict()


def unflatten_dict(
    flat_dict: Dict[str, Y], input_dict: NestedDict[str, V], prefix: str = ""
) -> NestedDict[str, Y]:
    return DIterator(input_dict).unflatten(flat_dict, prefix).to_dict()


def flattened_key_iterator(
    input_dict: NestedDict[K, V], prefix: str = ""
) -> Iterator[str]:
    return DIterator(input_dict).flatten(prefix).keys()


def flattened_value_iterator(input_dict: NestedDict[K, V]) -> Iterator[V]:
    return DIterator(input_dict).values()
