import collections

PAD = 0 # xxpad
UNK = 1 # xxunk
BOS = 2 # xxbos
EOS = 3 # xxeos

class Vocab():
    """
    Contain the correspondance between numbers and tokens and numericalize.
    """

    def __init__(self, itos):
        self.itos = itos
        # set default to unk
        self.stoi = collections.defaultdict(lambda: UNK,
                                            {v: k for k, v in enumerate(self.itos)})

    def numericalize(self, t):
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums, sep=' '):
        "Convert a list of `nums` to their tokens."
        return sep.join([self.itos[i] for i in nums])

    def __getstate__(self):
        return {'itos': self.itos}

    def __setstate__(self, state: dict):
        self.itos = state['itos']
        self.stoi = collections.defaultdict(
            int, {v: k for k, v in enumerate(self.itos)})

    def numericalize_all(self, tokens):
        "Convert a list of sentences of tokens to their ids."
        return [self.numericalize(t) for t in tokens]
