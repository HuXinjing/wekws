#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Jing Du(thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import re


def split_mixed_label(input_str):
    tokens = []
    s = input_str
    while len(s) > 0:
        match = re.match(r'[A-Za-z!?,<>_()\']+', s)
        if match is not None:
            word = match.group(0)
        else:
            word = s[0:1]
        tokens.append(word)
        s = s.replace(word, '', 1).strip(' ')
    return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('text_file', help='text file')
    parser.add_argument('duration_file', help='duration file')
    parser.add_argument('output_file', help='output list file')
    args = parser.parse_args()

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    duration_table = {}
    with open(args.duration_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            duration_table[arr[0]] = float(arr[1])

    with open(args.text_file, 'r', encoding='utf8') as fin, \
         open(args.output_file, 'w', encoding='utf8') as fout:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            if len(arr) < 2:
                txt = '<SILENCE>'
            else:
                txt = ' '.join(split_mixed_label(arr[1]))
            assert key in wav_table
            wav = wav_table[key]
            assert key in duration_table
            duration = duration_table[key]
            line = dict(key=key, txt=txt, duration=duration, wav=wav)

            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + '\n')


def read_token(token_file):
    """Read token file and return a dictionary mapping token to index."""
    token_table = {}
    with open(token_file, 'r', encoding='utf8') as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) >= 2:
                token = arr[0]
                idx = int(arr[1])
                token_table[token] = idx
    return token_table


def read_lexicon(lexicon_file):
    """Read lexicon file and return a dictionary mapping word to token list."""
    lexicon_table = {}
    with open(lexicon_file, 'r', encoding='utf8') as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) >= 2:
                word = arr[0]
                tokens = arr[1:]
                lexicon_table[word] = tokens
    return lexicon_table


def query_token_set(keyword, token_table, lexicon_table):
    """
    Query token set for a keyword.
    
    Args:
        keyword: The keyword string
        token_table: Dictionary mapping token to index
        lexicon_table: Dictionary mapping word to token list
    
    Returns:
        strs: List of token strings
        indexes: List of token indices
    """
    strs = []
    indexes = []
    
    # Split keyword into characters/words
    # For Chinese, each character is a word
    # For English, split by space or use each character
    if ' ' in keyword:
        words = keyword.split()
    else:
        # For Chinese or single word, treat each character as a word
        words = list(keyword)
    
    for word in words:
        if word in lexicon_table:
            # Word found in lexicon
            tokens = lexicon_table[word]
            for token in tokens:
                if token in token_table:
                    strs.append(token)
                    indexes.append(token_table[token])
        elif word in token_table:
            # Word is directly a token
            strs.append(word)
            indexes.append(token_table[word])
        else:
            # Try to find character by character
            for char in word:
                if char in token_table:
                    strs.append(char)
                    indexes.append(token_table[char])
                elif char in lexicon_table:
                    tokens = lexicon_table[char]
                    for token in tokens:
                        if token in token_table:
                            strs.append(token)
                            indexes.append(token_table[token])
    
    return strs, indexes
