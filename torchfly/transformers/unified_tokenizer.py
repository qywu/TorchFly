import sys
import json
import logging
import os
import regex as re
from functools import lru_cache

from ..utils.file_utils import http_get
from .base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab.json',
    'merges_file': 'merges.txt',
}

CACHE_DIR = ".cache"


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.
    
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    # we only support python 3
    _chr = chr
    bs = list(range(ord("!"),
                    ord("~") + 1)) + list(
                        range(ord("¡"),
                              ord("¬") + 1)
                    ) + list(range(ord("®"),
                                   ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class UnifiedTokenizer(BaseTokenizer):
    """
    RoBERTa BPE tokenizer, derived from the GPT-2 tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
          the absence of a space at the beginning of a string: `tokenizer.decode(tokenizer.encode("Hello")) = " Hello"`
    """

    #TODO: write a C++ version to speed up?

    def __init__(self, vocab_file=None, merges_file=None, errors='replace'):
        super(UnifiedTokenizer, self).__init__(
            max_len=512, special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
        )

        if vocab_file is None or merges_file is None:
            vocab_file, merges_file = self.load_from_cache()

        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, encoding='utf-8') as f:
            bpe_data = f.read().split('\n')[1:-1]
            bpe_merges = [tuple(merge.split()) for merge in bpe_data]

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def load_from_cache(self):
        vocab_path = os.path.join(
            os.getenv("HOME"), ".cache", "torchfly", "tokenizers",
            "roberta-vocab.json"
        )
        merges_path = os.path.join(
            os.getenv("HOME"), ".cache", "torchfly", "tokenizers",
            "roberta-merges.txt"
        )

        if not os.path.exists(vocab_path):
            # create the folder
            logger.warning("Downloading roberta-vocab.json")
            os.makedirs(
                os.path.join(
                    os.getenv("HOME"), ".cache", "torchfly", "tokenizers"
                ),
                exist_ok=True
            )
            http_get(
                "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
                vocab_path
            )

        if not os.path.exists(merges_path):
            # create the folder
            logger.warning("Downloading roberta-merges.json")
            os.makedirs(
                os.path.join(
                    os.getenv("HOME"), ".cache", "torchfly", "tokenizers"
                ),
                exist_ok=True
            )
            http_get(
                "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
                merges_path
            )

        return vocab_path, merges_path

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf'))
            )
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i +
                                                                   1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text, add_prefix_space=False):
        """ Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space toto get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
        if add_prefix_space:
            text = ' ' + text

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(
                    self.byte_encoder[ord(b)] for b in token
                )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            else:
                token = ''.join(
                    self.byte_encoder[b] for b in token.encode('utf-8')
                )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' ')
            )
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        # if not found return 50256
        return self.encoder.get(token, 50256)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c]
                          for c in text]).decode('utf-8', errors=self.errors)
        return text