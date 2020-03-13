#!/usr/bin/env python3
#
import os, sys, json
import argparse
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


import pdb


class Preprocess(object):
    """docstring for Preprocess"""
    def __init__(self, args):
        super(Preprocess, self).__init__()
        self.data_path = args.data_path
        self.mode = args.mode
        self.data_dir = os.path.join(self.data_path, self.mode)


    def sentenize(self):
        """
        split text into sentences
        """
        self.processed_data = {}

        for file_name in tqdm(os.listdir(self.data_dir)):
            if file_name.endswith('.txt'):
            # # if this is a text file
                file_path = os.path.join(self.data_dir, file_name)
                idx = -1

                with open(file_path) as file:
                    for line in file.readlines():
                        if line:
                        # # if this is not an empty line
                            for sent in sent_tokenize(line):
                                idx += 1
                                sent_id = '{:08d}_{:08d}'.format(int(file_name.split('.')[0]), idx)
                                self.processed_data[sent_id] = sent


    def save_file(self):
        """
        save processed data into json file
        """
        self.target_file_path = os.path.join(self.data_dir, 'processed.json')
        with open(self.target_file_path, 'w') as target_file:
            json.dump(self.processed_data, target_file, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data_path', default='/home/wuqy1203/pg19/deepmind-gutenberg/')
    args = parser.parse_args()

    preprocess = Preprocess(args)
    preprocess.sentenize()
    preprocess.save_file()


if __name__ == '__main__':
    main()
