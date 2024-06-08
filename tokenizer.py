# Copyright (C) Ronsor Labs. All rights reserved.
#
# The license of this software is specified in the LICENSE file at the root of
# this repository.

import ast, os


class WorldTokenizer:
  def __init__(self, vocab_file=None, eos_token=None, eos_token_id=0):
    self.eos_token = eos_token
    self.eos_token_id = eos_token_id

    self.i2t = {}
    self.t2i = {}
    self.trie = {}

    if vocab_file is None:
      vocab_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab', 'rwkv_vocab_v20230424.txt')

    with open(vocab_file, 'r', encoding='utf-8') as fp:
      for line in fp:
        spc_left, spc_right = line.index(' '), line.rindex(' ')

        id = int(line[:spc_left])
        token = ast.literal_eval(line[spc_left:spc_right])
        if isinstance(token, str):
          token = bytes(token, 'utf-8')

        token_len = int(line[spc_right:])
        assert token_len == len(token)

        self.add_token(token, id)

    if eos_token_id not in self.i2t and eos_token is not None:
      self.add_token(bytes(eos_token, 'utf-8'), eos_token_id)

  def add_token(self, token, id):
    assert isinstance(token, bytes)

    self.i2t[id] = token
    self.t2i[token] = id
    self._trie_insert(token, id)

  def _trie_insert(self, token, id):
    node = self.trie
    for byte in token:
      if byte not in node:
        node[byte] = {}

      node = node[byte]
    node[-1] = id

  def _trie_find_longest(self, seq, n=0):
    node = self.trie
    ret = None, None
    while seq[n] in node:
      node = node[seq[n]]
      n += 1

      if -1 in node:
        ret = n, node[-1]

      if n == len(seq):
        break

    return ret

  def encode_bytes_iter(self, x):
    n = 0
    while n < len(x):
      n, id = self._trie_find_longest(x, n)
      assert id is not None
      yield id

  def encode_bytes(self, x):
    return list(self.encode_bytes_iter(x))

  def encode_iter(self, x):
    return self.encode_bytes_iter(bytes(x, 'utf-8'))

  def encode(self, x):
    return list(self.encode_iter(x))

  def decode_bytes(self, x):
    return b''.join([self.i2t[i] for i in x])

  def decode(self, x):
    return self.decode_bytes(x).decode('utf-8', errors='ignore')
