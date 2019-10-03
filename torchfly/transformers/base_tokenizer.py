import sys
import json
import logging
import os
import regex as re
from abc import ABC, abstractmethod, abstractproperty
from functools import lru_cache
        
logger = logging.getLogger(__name__)

class BaseTokenizer(ABC):
    def __init__(self, max_len=1024):
        self.max_len = max_len

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
        return self._tokenize(text)

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = [self._convert_token_to_id(token) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self._convert_id_to_token(i) for i in ids]
        text = self.convert_tokens_to_string(tokens)
        return text
        