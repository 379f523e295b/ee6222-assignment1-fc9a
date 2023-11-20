import typing

import tensorflow_datasets as tfds
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.utils import type_utils
from tensorflow_datasets.core.features import feature as feature_lib
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf

import bidict


class ClassLabel(tfds.features.Tensor):
    @staticmethod
    def _force_numpy(data):
        return (
            data if not isinstance(data, tf.Tensor) else
            data.numpy()
        )

    @staticmethod
    def _force_cast(a, dtype):
        a = tf.constant(a)
        if not a.dtype.is_numeric:
            return tf.strings.to_number(a, out_type=dtype)
        return tf.cast(a, dtype=dtype)

    def __init__(
        self,
        *,
        # TODO !!!!! doc !!!! {<encoded>: <decoded>}
        mapping: typing.Mapping,
        shape: utils.Shape = (),
        # TODO !!!!!
        dtype: type_utils.TfdsDType = str,
        ignore_errors: bool = True,
        **kwargs,
    ):
        self._init_args = dict(
            mapping=mapping,
            shape=shape,
            dtype=dtype,
            ignore_errors=ignore_errors,
            **kwargs
        )

        super().__init__(
            shape=shape, 
            dtype=(
                dtype.as_numpy_dtype()
                if isinstance(dtype, (tf.DType, tf.dtypes.DType))
                else
                dtype
            ), 
            **kwargs
        )

        self._mapping = bidict.bidict({
            self._force_numpy(
                self._force_cast(k, self.dtype)
            ): v
            for k, v in mapping.items()
        })
        self._ignore_errors = ignore_errors

    def to_json_content(self) -> type_utils.Json:
        return {
            **self._init_args,
            #'shape': feature_lib.to_shape_proto(self._init_args['shape']),
            'dtype': feature_lib.dtype_to_str(self._init_args['dtype']),
        }
    
    # TOOD !!!!!
    @classmethod
    def from_json_content(cls, value: type_utils.Json) -> 'ClassLabel':
        return cls(**{
            **value,
            #'shape': feature_lib.from_shape_proto(value['shape']),
            'dtype': feature_lib.dtype_from_str(value['dtype']),
        })
    
    def decode_example(self, data):
        try: return self._decode_example(data)
        except Exception as e: 
            if not self._ignore_errors:
                raise e
            return data

    def encode_example(self, data):
        try: return self._encode_example(data)
        except Exception as e: 
            if not self._ignore_errors:
                raise e
            return data
        
    def repr_html(self, data):
        return f'{data!r} ({self.decode_example(data)!r})'
    
    def _additional_repr_info(self):
        return {'num_classes': self.num_classes}

    def _decode_example(self, data):
        return self._mapping[
            self._force_numpy(data)
        ]

    def _encode_example(self, data):
        return self._mapping.inverse[
            self._force_numpy(data)
        ]

    @property
    def keys(self):
        return self._mapping.keys()

    @property
    def names(self):
        return self._mapping.values()
    
    @property
    def num_classes(self):
        return len(self._mapping)

    @property
    def mapping(self):
        return self._mapping


__all__ = [
    ClassLabel
]