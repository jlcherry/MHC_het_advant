#  ===========================================================================
# 
#                             PUBLIC DOMAIN NOTICE
#                         National Library of Medicine
# 
#   This software/database is a "United States Government Work" under the
#   terms of the United States Copyright Act.  It was written as part of
#   the author's official duties as a United States Government employee and
#   thus cannot be copyrighted.  This software/database is freely available
#   to the public for use. The National Library of Medicine and the U.S.
#   Government have not placed any restriction on its use or reproduction.
# 
#   Although all reasonable efforts have been taken to ensure the accuracy
#   and reliability of the software and data, the NLM and the U.S.
#   Government do not and cannot warrant the performance or results that
#   may be obtained by using this software or data. The NLM and the U.S.
#   Government disclaim all warranties, express or implied, including
#   warranties of performance, merchantability or fitness for any particular
#   purpose.
# 
#   Please cite the author in any work or product based on this material.
# 
#  ===========================================================================
# 
#  Authors:  Joshua L. Cherry
# 
#  File Description: Bitstring model for MHC evolution
# 


import numpy as np

all_ones17 = 2**17-1  # use this to guarantee enough bits

def MaxZeroRun(n):
    s = bin(n)[2:]
    tmp = s.split('1')
    return max(len(x) for x in tmp)

max_zero_run = [MaxZeroRun(i) for i in range(2**17)]


def LongestMatch(n1, n2):
    return max_zero_run[(n1+65536)^n2]


log9 = np.log(9)

def _Eff(a, p, v):
    '''
    for a single pathogen
    '''
    return 1 - np.prod([1 - 1 / (1 + np.exp(log9*(v - LongestMatch(a, pep))))
                        for pep in p])


_D_cache = dict()  # actually 1-D
_prod_cache = {}
_last_v = None
def _EffFast(a, p, v):
    '''
    for a single pathogen
    '''
    if v not in _D_cache:
        _D_cache[v] = [1 - 1 / (1 + np.exp(log9*(v - i))) for i in range(17)]
    D = _D_cache[v]
    lms = tuple(sorted([LongestMatch(a, pep) for pep in p]))
    global _last_v
    global _prod_cache
    if _last_v is None or v != _last_v:
        _prod_cache = {}
        _last_v = v
    if lms not in _prod_cache:
        _prod_cache[lms] = 1 - np.prod([D[lm] for lm in lms])
    return _prod_cache[lms]

         
def _EffFastPenalty(a, p, v, pen):
    '''
    for a single pathogen
    '''
    if v not in _D_cache:
        _D_cache[v] = [1 - 1 / (1 + np.exp(log9*(v - i))) for i in range(17)]
    D = _D_cache[v]
    lms = tuple(sorted([LongestMatch(a, pep) for pep in p]))
    global _last_v
    global _prod_cache
    if _last_v is None or v != _last_v:
        _prod_cache = {}
        _last_v = v
    if (lms, pen) not in _prod_cache:
        _prod_cache[lms, pen] = 1 - np.prod([1 - pen*(1-D[lm]) for lm in lms])
    return _prod_cache[lms, pen]


def EffBS(a, P, v):
    '''
    P consists of sequences of bitstrings
    '''
    return [_EffFast(a, p, v) for p in P]


def EffBSPenalty(a, P, v, pen):
    '''
    P consists of sequences of bitstrings
    '''
    return [_EffFastPenalty(a, p, v, pen) for p in P]


def EffBSFast(a, P, v):
    '''
    P consists of pre-computed efficiencies for all MHC alleles.
    N.B.: v is ignored
    '''
    return P[a]


def EffectiveNumberPeps(v):
    '''
    A measure of diversity of peptides presented.
    Detection probabilities for all peptides are divided by their sum,
    and the result is treated as a probability distribution.
    Effective number of peptides is the reciprocal of the
    sum of squares of these probabilities.
    '''
    d = np.array([_EffFast(0, (p,), v) for p in range(65536)])
    d = d / d.sum()
    return 1 / sum(d**2)
