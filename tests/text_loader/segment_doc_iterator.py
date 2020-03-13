import os
import numpy as np
import logging

from segment_doc_loader import SegmentDocLoader

logger = logging.getLogger(__name__)

class SegmentDocIterator:
    def __init__(self, processed_docs):
        self.processed_docs = processed_docs
        self.total_num_docs = len(processed_docs)

    def __iter__ (self):
        # shuffle the indices
        indices = np.arange(self.total_num_docs)
        np.random.shuffle(indices)

        for doc_index in indices:
            # randomly sample a document
            doc = self.processed_docs[doc_index]

            for i, segment in enumerate(doc):
                # output if the segment is the start of the document
                yield segment, i==0


class SegmentDocBatchIterator:
    def __init__(self, tokenizer, corpus_path:str, batch_size:int, max_seq_length:int, rank:int = 0):
        """
        Args:
            corpus_path: directory path to store the corpus sectors
            rank: for distributed learning.
        """ 
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.current_sector_id = rank
        self.corpus_loader = SegmentDocLoader(tokenizer, 
                                          max_seq_length=max_seq_length, 
                                          corpus_path=corpus_path)
        self.total_num_sectors = len(os.listdir(corpus_path))
        
        # process the data and save it into cache
    
    def __iter__(self):
        iterators = self.create_corpus_iterators(self.current_sector_id)
        
        while True:
            try:
                # TODO: extend it with Ray
                batch = [next(iterators[i]) for i in range(self.batch_size)]
                yield batch
                
            except StopIteration:
                # after the iterator finishes, load the next sector
                # update self.current_sector_id
                self.current_sector_id = (self.current_sector_id + 1) % self.total_num_sectors
                iterators = self.create_corpus_iterators(self.current_sector_id)
                
    def create_corpus_iterators(self, corpus_sector_id):
        processed_docs = self.corpus_loader.load_sector(self.current_sector_id)
        iterators = [iter(SegmentDocIterator(processed_docs)) for i in range(self.batch_size)]
        return iterators