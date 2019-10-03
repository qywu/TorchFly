import sys
import json
import logging
import os
import regex as re
from abc import ABC, abstractmethod, abstractproperty
from functools import lru_cache
        
logger = logging.getLogger(__name__)

class BaseTokenizer(ABC):
    def __init__(self, max_len=1024, special_tokens=None):
        self.max_len = max_len
        self.special_tokens = special_tokens

    @abstractproperty
    def vocab_size(self):
        """ Size of the base vocabulary (without the added tokens) """
        raise NotImplementedError

    @abstractmethod
    def _tokenize(self, text):
        raise NotImplementedError

    @abstractmethod
    def _convert_token_to_id(self, token):
        raise NotImplementedError

    @abstractmethod
    def _convert_id_to_token(self, index):
        raise NotImplementedError

    @abstractmethod
    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size

    def tokenize(self, text):
        if self.special_tokens is None:
            return self._tokenize(text)
        else:
            # sepcial tokens are assumed registered
            pattern = '(' + "|".join(self.special_tokens) + ')'
            raw_tokens = re.split(pattern, text)
            processed_tokens = []
            for raw_token in raw_tokens:
                if len(raw_token) > 0:
                    if raw_token in self.special_tokens:
                        processed_tokens.append(raw_token)
                    else:
                        processed_tokens.extend(self._tokenize(raw_token))
            return processed_tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = [self._convert_token_to_id(token) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self._convert_id_to_token(i) for i in ids]
        text = self.convert_tokens_to_string(tokens)
        return text
        