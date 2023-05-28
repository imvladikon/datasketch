cdef class CMinHashLSHForest:
    cdef int l
    cdef int k
    cdef list hashtables
    cdef list hashranges
    cdef dict keys
    cdef list sorted_hashtables

    def __init__(self, int num_perm=128, int l=8):
        if l <= 0 or num_perm <= 0:
            raise ValueError("num_perm and l must be positive")
        if l > num_perm:
            raise ValueError("l cannot be greater than num_perm")
        self.l = l
        self.k = num_perm / l
        self.hashtables = [dict() for _ in range(self.l)]
        self.hashranges = [(i * self.k, (i + 1) * self.k) for i in range(self.l)]
        self.keys = dict()
        # This is the sorted array implementation for the prefix trees
        self.sorted_hashtables = [[] for _ in range(self.l)]

    def add(self, key, minhash):
        cdef int start, end
        cdef list H
        cdef list keys
        cdef bytes h

        if len(minhash) < self.k * self.l:
            raise ValueError("The num_perm of MinHash out of range")
        if key in self.keys:
            raise ValueError("The given key has already been added")

        cdef int i = 0
        keys = []
        for start, end in self.hashranges:
            h = self._H(minhash.hashvalues[start:end])
            keys.append(h)
            if h not in self.hashtables[i]:
                self.hashtables[i][h] = [key]
            else:
                self.hashtables[i][h].append(key)
            i += 1

    def index(self):
        cdef int i
        cdef int len_hashtables = len(self.hashtables)
        for i in range(len_hashtables):
            self.sorted_hashtables[i] = [H for H in self.hashtables[i].keys()]
            self.sorted_hashtables[i].sort()

    def _query(self, minhash, int r, int b):
        cdef int prefix_size, i, j
        cdef list hps
        cdef list ht
        cdef dict hashtable
        cdef bytes hp

        if r > self.k or r <= 0 or b > self.l or b <= 0:
            raise ValueError("parameter outside range")

        hps = [self._H(minhash.hashvalues[start:start + r]) for start, _ in self.hashranges]
        prefix_size = len(hps[0])
        cdef int len_hps = len(hps)
        for i in range(len_hps):
            ht = self.sorted_hashtables[i]
            hp = hps[i]
            hashtable = self.hashtables[i]
            i = self._binary_search(len(ht), hp, ht, prefix_size)
            if i < len(ht) and ht[i][:prefix_size] == hp:
                j = i
                while j < len(ht) and ht[j][:prefix_size] == hp:
                    for key in hashtable[ht[j]]:
                        yield key
                    j += 1

    def query(self, minhash, int k):
        cdef set results
        cdef int r
        cdef list keys
        if k <= 0:
            raise ValueError("k must be positive")
        if len(minhash) < self.k * self.l:
            raise ValueError("The num_perm of MinHash out of range")
        results = set()
        r = self.k
        while r > 0:
            keys = list(self._query(minhash, r, self.l))
            for key in keys:
                results.add(key)
                if len(results) >= k:
                    return list(results)
            r -= 1
        return list(results)

    def _binary_search(self, int n, bytes hp, list arr, int prefix_size):
        cdef int i = 0
        cdef int j = n
        cdef int h
        while i < j:
            h = i + (j - i) / 2
            if not arr[h][:prefix_size] >= hp:
                i = h + 1
            else:
                j = h
        return i

    def is_empty(self):
        cdef list t
        return any(len(t) == 0 for t in self.sorted_hashtables)

    def _H(self, hs):
        return bytes(hs.byteswap().data)

    def __contains__(self, key):
        return key in self.keys
