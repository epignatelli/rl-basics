class Table():
    def __init__(self):
        self._dict = dict()

    def get(self, s, a, default=0.):
        key = self._make_key(s, a)
        return self._dict.get(key)

    def update(self, s, a, q):
        key = self._make_key(s, a)
        self._dict[key] = q
        return

    def _make_key(self, s, a):
        return (str(s), str(a))
