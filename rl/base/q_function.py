class QTable(dict):
    def get(self, s, a, default=0.):
        key = self._make_key(s, a)
        if key in self:
            return self[key]
        else:
            return default

    def update(self, s, a, q):
        key = self._make_key(s, a)
        self[key] = q
        return True

    def _make_key(self, s, a):
        return (str(s), str(a))
