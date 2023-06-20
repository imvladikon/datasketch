#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import struct
cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    cdef bytes digest = hashlib.sha1(data).digest()[:4]
    return struct.unpack('<I', digest)[0]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sha1_hash64(data):
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    cdef bytes digest = hashlib.sha1(data).digest()[:8]
    return struct.unpack('<Q', digest)[0]
