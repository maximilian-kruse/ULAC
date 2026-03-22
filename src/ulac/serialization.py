import dataclasses
from functools import lru_cache
from typing import Any

import msgspec.msgpack
import numpy as np

from .construction import internal


def _np_array_enc_hook(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tobytes(), "dtype": str(obj.dtype), "shape": list(obj.shape)}
    raise TypeError(f"Unsupported type: {type(obj)}")


def _np_array_dec_hook(type: type, obj: Any) -> Any:
    if type is np.ndarray:
        return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(obj["shape"])
    raise TypeError(f"Unsupported type: {type}")


def _decode_recursively(data: dict, uac_submesh_attrs: frozenset[str]) -> dict | internal.UACSubmesh:
    if data.keys() == uac_submesh_attrs:
        return msgspec.convert(data, type=internal.UACSubmesh, dec_hook=_np_array_dec_hook)
    else:
            return {k: _decode_recursively(v, uac_submesh_attrs) for k, v in data.items()}


def encode_uac_submeshdata(data_dict: dict) -> bytes:
    return msgspec.msgpack.Encoder(enc_hook=_np_array_enc_hook).encode(data_dict)


def decode_uac_submeshdata(data: bytes) -> dict:
    raw_dict = msgspec.msgpack.decode(data)
    uac_submesh_attrs = frozenset(f.name for f in dataclasses.fields(internal.UACSubmesh))
    return _decode_recursively(raw_dict, uac_submesh_attrs)


def save_uac_submeshdata(path: str, submesh_dict: dict) -> None:
    with open(path, "wb") as f:
        f.write(encode_uac_submeshdata(submesh_dict))


def load_uac_submeshdata(path: str) -> dict[str, internal.UACSubmesh]:
    with open(path, "rb") as f:
        return decode_uac_submeshdata(f.read())
