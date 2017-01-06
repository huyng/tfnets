class model(dict):
    """
    Creates a dict that one can access using dot notation
    """
    __getattr__ = dict.__getitem__

    def __setattr__(self, attr, value):
        self[attr] = value
        self['last'] = self[attr]
