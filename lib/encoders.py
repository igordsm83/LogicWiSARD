import numpy as np
import struct
from numba import njit
import numbers
from sklearn.neighbors import KNeighborsClassifier

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return None
    

class ThermometerEncoder(object):
    def __init__(self, minimum, maximum, resolution):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution
        
    def __repr__(self):
        return f"ThermometerEncoder(minimum={self.minimum}, maximum={self.maximum}, resolution={self.resolution})"
    
    def encode(self, X):
        X = np.asarray(X)

        if X.ndim == 0:
            f = lambda i: X > self.minimum + i*(self.maximum - self.minimum)/self.resolution
        elif X.ndim == 1:
            f = lambda i, j: X[j] > self.minimum + i*(self.maximum - self.minimum)/self.resolution
        else:
            f = lambda i, j, k: X[k, j] > self.minimum + i*(self.maximum - self.minimum)/self.resolution 
        return  np.fromfunction(
            f,
            (self.resolution, *reversed(X.shape)),
            dtype=int
        ).astype(int)
    
    def decode(self, pattern):
        pattern = np.asarray(pattern)
        
        # TODO: Check if pattern is at least a vector
        # TODO: Check if pattern length or number of rows is equal to resolution
        # TODO: Check if pattern is a binary array
        if pattern.ndim == 1:
            # TODO: Test np.count_nonzero
            popcount = np.sum(pattern)
            
            return self.minimum + popcount*(self.maximum - self.minimum)/self.resolution
        
        return np.asarray([self.decode(pattern[...,i]) for i in range(pattern.shape[-1])])
    
    
class CircularThermometerEncoder(object):
    def __init__(self, minimum, maximum, resolution, wrap=True):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution
        self.block_len = np.floor(self.resolution/2)
        self.wrap = wrap
        self.max_shift = resolution if wrap else resolution - self.block_len
        
    def __repr__(self):
        return f"CircularThermometerEncoder(minimum={self.minimum}, maximum={self.maximum}, resolution={self.resolution}), wrap={self.wrap}"
    
    def encode(self, X):
        X = np.asarray(X)

        if X.ndim == 0:
            if X < self.minimum or X > self.maximum:
                raise ValueError(
                    f"Encoded values should be in the range [{self.minimum}, {self.maximum}]. Value given: {X}")

            base_pattern = np.fromfunction(lambda i: i < self.block_len, (self.resolution,)).astype(np.uint8)
            shift = int(np.abs(self.minimum-X)/(self.maximum-self.minimum)*self.max_shift)

            return np.roll(base_pattern, shift)

        return np.stack([self.encode(v) for v in X], axis=X.ndim)
    
    def decode(self, pattern):
        pattern = np.asarray(pattern)
        
        # TODO: Check if pattern is at least a vector
        # TODO: Check if pattern length or number of rows is equal to resolution
        # TODO: Check if pattern is a binary array
        if pattern.ndim == 1:
            first_0 = index(pattern, 0)[0]
            first_1 = index(pattern, 1)[0]

            if first_0 > first_1:
                shift = (first_0 - self.block_len) % self.resolution
            else:
                shift = first_1

            if shift > self.max_shift:
                raise ValueError(
                    "Input pattern wraps around. Consider using a encoder with wrap enabled")
            
            return self.minimum + shift*(self.maximum - self.minimum)/self.max_shift
        
        return np.asarray([self.decode(pattern[...,i]) for i in range(pattern.shape[-1])])


class FloatBinaryEncoder(object):
    def __init__(self, double=True):
        self.double = double

    def __repr__(self):
        return f"FloatBinaryEncoder(double={self.double})"
    
    def encode(self, X):
        X = np.asarray(X)

        if X.ndim == 0:
            bitstring = ''.join(
                bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!d' if self.double else '!f', X)
            )
            return np.asarray([int(c) for c in bitstring])

        return np.stack([self.encode(v) for v in X], axis=X.ndim)
    

# Encode neighborhood information of a point to be used as suffix by CodeWord
class NeighborsSuffix(object):
    def __init__(self, X, y, k=5, repeats=1):
        self.repeats  = repeats
        self.nn_model = KNeighborsClassifier(k)
        self.nn_model.fit(X, y)
    
    def __call__(self, X):
        return np.repeat(
            np.apply_along_axis(
                lambda y: np.array([1,1,0]) if y[0] == -1 else np.array([0,1,1]),
                1,
                self.nn_model.predict(np.atleast_2d(X)).reshape(-1, 1)
            ),
            self.repeats,
            axis=1
        )
    
    def __len__(self):
        return 3*self.repeats


class MockNeighborsSuffix(object):
    def __init__(self, repeats=1):
        self.repeats = repeats
        
    def __call__(self, X):
        X = np.ascontiguousarray(X)
        
        depth = 1 if X.ndim == 1 else X.shape[0]
        
        return np.squeeze(
            np.repeat(
                np.tile(
                    np.asarray([0,1,0]),
                    (depth,1)
                ),
                self.repeats,
                axis=1
            )
        )
    
    def __len__(self):
        return 3*self.repeats
    
class Morph(object):
    # Make this argument optional. Fill it in when flatten is first called
    # If inflate is called and original_shape is not known, throw exception
    def __init__(self, original_shape=None):
        self.original_shape = original_shape
        
    def flatten(self, X, column_major=True):
        X = np.asarray(X)
        
        if self.original_shape is None:
            self.original_shape = X.shape
        
        order = 'F' if column_major else 'C'

        if X.ndim < 2:
            return X
        elif X.ndim == 2:
            return X.ravel(order=order)

        return np.asarray([X[:, :, i].ravel(order=order) for i in range(X.shape[2])])
    
    def inflate(self, X):
        if self.original_shape is None:
            raise AttributeError("Cannot inflate without knowing the original shape")
            
        X = np.asarray(X)
        
        if X.ndim == 1:
            return np.reshape(X, self.original_shape[:2], order='F')
        elif X.ndim == 2:
            return np.stack([self.inflate(v) for v in X], axis=2)
        
        raise ValueError('Dimension mismatch')


class CodeWord(object):
    def __init__(self, encoder, morpher=Morph(), prefix=None, suffix=None):
        self.encoder = encoder
        self.morpher = morpher # A default Morph could be created from the encoder (right?) Could delay its creation until we have the first pattern
        self.prefix  = prefix
        self.suffix  = suffix
    
    def _resolve_affix(self, X, affix):
        if affix is None:
            return np.empty((1,1), dtype=int)
        elif callable(affix):
            return np.atleast_2d(np.asarray(affix(X)))
        return np.atleast_2d(np.ascontiguousarray(affix))
    
    def _affix_len(self, affix):
        if affix is None:
            return 0
        elif isinstance(affix, numbers.Number):
            return 1
        return len(affix)
        
    def _remove_affixes(self, pattern):
        if pattern.ndim == 1:
            return pattern[self._affix_len(self.prefix):len(pattern)-self._affix_len(self.suffix)]
        else:
            return pattern[:, self._affix_len(self.prefix):pattern.shape[1]-self._affix_len(self.suffix)]
    
    def encode(self, X):
        components = []
        self.prefix is not None and components.append(self._resolve_affix(X, self.prefix))
        components.append(
            np.atleast_2d(
                self.morpher.flatten(
                    self.encoder.encode(X)
                )
            )
        )
        self.suffix is not None and components.append(self._resolve_affix(X, self.suffix))
        
        return np.squeeze(np.concatenate(components, axis=1))
    
    def decode(self, pattern):
        return self.encoder.decode(
            self.morpher.inflate(
                self._remove_affixes(pattern)
            )
        )


def flatten(X, column_major=True):
    X = np.asarray(X)
    order = 'F' if column_major else 'C'

    if X.ndim < 2:
        return X
    elif X.ndim == 2:
        return X.ravel(order=order)

    return np.asarray([X[:, :, i].ravel(order=order) for i in range(X.shape[2])])

       
def process_input(X, encoder):
    return flatten(encoder.encode(X), column_major=True).tolist()
