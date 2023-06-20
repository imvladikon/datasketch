import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t

from datasketch.chashfunc import sha1_hash32


# The size of a hash value in number of bytes
cdef int hashvalue_byte_size = len(bytes(np.int64(42).data))

# http://en.wikipedia.org/wiki/Mersenne_prime
cdef uint64_t _MERSENNE_PRIME = np.uint64((1 << 61) - 1)
cdef uint64_t _MAX_HASH = np.uint64((1 << 32) - 1)
cdef uint64_t _HASH_RANGE = (1 << 32)


cdef extern from "limits.h":
    int INT_MAX


cdef class CMinHash:
    cdef int seed
    cdef uint64_t num_perm
    cdef object hashfunc
    cdef object _hashvalues
    cdef object _permutations

    def __init__(self,
                 uint64_t num_perm=128,
                 int seed=1,
                 object hashfunc=sha1_hash32,
                 np.ndarray[uint64_t] hashvalues=None,
                 np.ndarray[uint64_t, ndim=2] permutations=None):
        # cdef np.ndarray[uint64_t] hashvalues
        # cdef np.ndarray[uint64_t, ndim=2] permutations

        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm > _HASH_RANGE:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError("Cannot have more than %d number of permutation functions" % _HASH_RANGE)
        self.seed = seed
        self.num_perm = num_perm
        # Check the hash function.
        # if not callable(hashfunc):
        #     raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc
        # Initialize hash values
        if hashvalues is not None:
            self._hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self._hashvalues = self._init_hashvalues(num_perm)
        # Initialize permutation function parameters
        if permutations is not None:
            self._permutations = permutations
        else:
            self._permutations = self._init_permutations(num_perm)
        if len(self) != len(self._permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    @property
    def permutations(self):
        return self._permutations

    @property
    def hashvalues(self):
        return self._hashvalues

    cdef np.ndarray[uint64_t] _init_hashvalues(self, int num_perm):
        return np.ones(num_perm, dtype=np.uint64) * _MAX_HASH

    cdef np.ndarray[np.npy_uint64, ndim=2] _init_permutations(self, int num_perm):
        gen = np.random.RandomState(self.seed)
        return np.array([
            (gen.randint(1, _MERSENNE_PRIME, dtype=np.uint64), gen.randint(0, _MERSENNE_PRIME, dtype=np.uint64))
            for _ in range(num_perm)
        ], dtype=np.uint64).T

    cdef np.ndarray[np.npy_uint64] _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)

    def update(self, v):
        cdef uint64_t hv = self.hashfunc(v)
        cdef np.ndarray[np.npy_uint64] a = self.permutations[0, :]
        cdef np.ndarray[np.npy_uint64] b = self.permutations[1, :]
        cdef np.ndarray[np.npy_uint64] phv = np.bitwise_and((a * hv + b) % _MERSENNE_PRIME, _MAX_HASH)
        self._hashvalues = np.minimum(phv, self.hashvalues)

    def update_batch(self, v):
        cdef np.ndarray[np.npy_uint64, ndim = 2] hv = np.array([self.hashfunc(_b) for _b in v], dtype=np.uint64)
        cdef np.ndarray[np.npy_uint64] a = self.permutations[0, :]
        cdef np.ndarray[np.npy_uint64] b = self.permutations[1, :]
        cdef np.ndarray[np.npy_uint64] phv = np.bitwise_and(((hv * np.tile(a, (len(hv), 1)).T).T + b) % _MERSENNE_PRIME, _MAX_HASH)
        self._hashvalues = np.vstack([phv, self.hashvalues]).min(axis=0)

    def jaccard(self, other):
        if other.seed != self.seed:
            raise ValueError("Cannot compute Jaccard given MinHash with different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with different numbers of permutation functions")
        return float(np.count_nonzero(self.hashvalues == other.hashvalues)) / float(len(self))

    def count(self):
        cdef uint64_t k = len(self)
        return float(k) / np.sum(self.hashvalues / float(_MAX_HASH)) - 1.0

    def merge(self, other):
        if other.seed != self.seed:
            raise ValueError("Cannot merge MinHash with different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot merge MinHash with different numbers of permutation functions")
        self._hashvalues = np.minimum(other.hashvalues, self.hashvalues)

    def digest(self):
        return np.copy(self.hashvalues)

    cpdef bint is_empty(self):
        cdef bint result
        result = not np.any(self.hashvalues != _MAX_HASH)
        return result

    cpdef void clear(self):
        self._hashvalues = self._init_hashvalues(len(self))

    def copy(self):
        return CMinHash(seed=self.seed,
                        hashfunc=self.hashfunc,
                        hashvalues=self.digest(),
                        permutations=self.permutations)

    def __len__(self):
        return len(self.hashvalues)

    def __eq__(self, CMinHash other):
        cdef bint result
        result = (type(self) is type(other) and
                  self.seed == other.seed and
                  np.array_equal(self.hashvalues, other.hashvalues))
        return result

    @classmethod
    def union(cls, *mhs):
        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(mhs[0])
        seed = mhs[0].seed
        if any((seed != m.seed or num_perm != len(m)) for m in mhs):
            raise ValueError("The unioning MinHash must have the same seed and number of permutation functions")
        hashvalues = np.minimum.reduce([m.hashvalues for m in mhs])
        permutations = mhs[0].permutations
        return cls(num_perm=num_perm, seed=seed, hashvalues=hashvalues, permutations=permutations)

    @classmethod
    def bulk(cls, b, **minhash_kwargs):
        return list(cls.generator(b, **minhash_kwargs))

    @classmethod
    def generator(cls, b, **minhash_kwargs):
        m = cls(**minhash_kwargs)
        for _b in b:
            _m = m.copy()
            _m.update_batch(_b)
            yield _m
