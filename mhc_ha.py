#!/usr/bin/env python
#
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
#  File Description: Simulation of MHC haplotype evolution
# 


import numpy as np
from scipy.special import gamma


def GetT(h, m=None):
    '''
    The m vertices of a generalized tetrahedron in h-dimensional space.
    '''
    if m is None:
        m = h + 1  # E.g., in 3 dimensions we have
                   # 3+1 == 4 corners of a tetrahedron

    amb = -1 / np.sqrt(2)
    b = 1 / (np.sqrt(2) * (m + np.sqrt(m)))
    c = 1 / (np.sqrt(2) * np.sqrt(m))

    t = np.eye(m-1) * amb + b
    t = np.hstack([t, c*np.ones((h, 1))])
    
    if h > m - 1:
        raise RuntimeError()

    return t


def EffGaussian(allele, pathogens, vir):
    return np.exp(-vir**2/2*((pathogens - allele)**2).sum(axis=0))


def _Cond(haps, pathogens, vir):
    effs = [Eff(allele, pathogens, vir) for allele in sum(haps, ())]
    mns = np.stack(effs).mean(axis=0)
    mns[0] *= c_max
    c = np.prod(mns)
    return c

Cond = _Cond


def CondFatalOverpresentation(haps, pathogens, vir):
    '''
    Presentation of too wide a variety of peptides is fatal.
    Meant for bitstring model with evolvable breadth.
    '''
    h1, h2 = haps
    a1, = h1; a2, = h2
    if (a1 > 65535 or a2 > 65535) and (a1 & 65535) != (a2 & 65535):
        return 0.
    return _Cond(haps, pathogens, vir)


def CondPenalizeOverpresentation(haps, pathogens, vir):
    '''
    Presentation of too wide a variety of peptides has a cost.
    Meant for bitstring model with evolvable breadth.
    pathogens should be a length-two sequence, with first element
    not including the penalty and second incuding it
    '''
    h1, h2 = haps
    a1, = h1; a2, = h2
    if (a1 > 65535 or a2 > 65535) and (a1 & 65535) != (a2 & 65535):
        return _Cond(haps, pathogens[1], vir)
    return _Cond(haps, pathogens[0], vir)


def CondLowExp(haps, pathogens, vir):
    '''
    alleles other than first in hap are weakly expressed
    '''
    le = 0.02  # relative expression level of lowly expressed
    exp_tot = (1-le)*len(haps) + le*sum(len(hap) for hap in haps)
    wts = sum(((1.,) + (len(hap)-1)*(le,) for hap in haps), ())
    wts = [w / exp_tot for w in wts]
    effs = [wts[i]*Eff(allele, pathogens, vir)
            for i, allele in enumerate(sum(haps, ()))]
    mns = np.stack(effs).sum(axis=0)
    mns[0] *= c_max
    c = np.prod(mns)
    return c


def Survival(haps, pathogens, vir, K):
    cond = Cond(haps, pathogens, vir)
    rv = cond / (cond + K)
    return rv


def Het(counts):
    '''
    Heterozygosity from allele/haplotype counts or frequencies
    '''
    counts = np.array(counts)
    freqs = counts / counts.sum()
    return 1 - (freqs**2).sum()


# Model specification, parameter values, and some set-up
bs = False         # bitstring model? (otherwise Gaussian)
N = 10**5          # population size
mu_del = 1e-7      # rate of deletion, per locus, in expanded haplos
mu_combine = 1e-9  # rate of combining alleles of both parents
K = 1.             # half-saturation for survival as function of condition
npep = 5           # number of peptides per pathogen
if not bs:
    Eff = EffGaussian
    mu = 5e-7
    v = 20.
    v_alt = None  # should always be None for Gaussian
    npath = 8  # number of pathogens
    P = GetT(npath - 1)
    deltamu = 0.03
    sigmamu = deltamu*gamma((npath-1)/2)/gamma(npath/2)/np.sqrt(2)
else:
    import bitstring
    npath = 100
    v = 9.
    v_alt = None  # alternative value of v that alleles can have
    P_bs = [[np.random.randint(65536) for j in range(npep)]
            for i in range(npath)]
    nbits = 16
    P = [np.array(bitstring.EffBS(a, P_bs, v)) for a in range(65536)]
    Eff = bitstring.EffBSFast
    mu = 5e-6

# Initialize population
if not bs:
    center = np.zeros((npath-1, 1))  # allele at origin
    haplos = [(center,)]
else:
    haplo_init = (np.random.randint(65536),)
    haplos = [haplo_init]
haplo_counts = [2*N]

# Set c_max
if not bs:
    c_max = 1 / np.prod(Eff(center, P, v))
else:
    c_max = 1 / max([np.prod(Eff(a, P, v)) for a in range(65536)])
    
# Set-up to allow mutation to change v
if bs and v_alt is not None:
    # Alter breadth of peptides presented
    ratio = (sum([bitstring._EffFast(i, [0], v) for i in range(65536)])
             / sum([bitstring._EffFast(i, [0], v_alt) for i in range(65536)]))
    # extra bit (highest order) determines allele-specific v
    P = P + [np.array(bitstring.EffBSPenalty(a, P_bs, v_alt, ratio))
             for a in range(65536)]
    nbits += 1  # for mutation
    if False:
        op_pen = 0.5  # decrease of detection probability due to
                      # overpresentation.  1 corresponds to no decrease;
                      # smaller numbers mean larger decreases.
        P2 = [np.array(bitstring.EffBSPenalty(a, P_bs, v, op_pen))
              for a in range(65536)]
        P2 += [np.array(bitstring.EffBSPenalty(a, P_bs, v_alt, op_pen*ratio))
               for a in range(65536)]
        P = (P, P2)
        Cond = CondPenalizeOverpresentation


extinct = []    # for first interation
res = []        # record of state at intervals
S = None
for gen in range(1_000_001):
    # diploid genotype frequencies
    geno_freqs = np.outer(haplo_counts, haplo_counts) / (2*N)**2

    # selection
    if S is None:
        S = np.zeros(geno_freqs.shape)
        for i in range(len(haplo_counts)):
            for j in range(i, len(haplo_counts)):
                s = Survival((haplos[i], haplos[j]), P, v, K)
                S[i, j] = s; S[j, i] = s
    geno_freqs *= S
    s_mean = geno_freqs.sum()
    geno_freqs /= s_mean
    haplo_freqs = geno_freqs.sum(axis=0)

    # Record/report state of population at intervals
    if ((gen < 1000 and gen % 10 == 0)
        or (gen < 10000 and gen % 100 == 0)
        or gen % 1000 == 0):
        
        tmp = (gen, len(haplos), 1 / (1 - Het(haplo_counts)), s_mean)
        if v_alt is None:
            mn_genes = sum([len(haplos[i])*haplo_counts[i]
                            for i in range(len(haplos))]) / (2*N)
            tmp += (mn_genes,)
        else:
            frac_v_alt = sum([haplo_counts[i] * (haplos[i][0] >= 2**16)
                              for i in range(len(haplos))]) / (2*N)
            tmp += (frac_v_alt,)
        res.append(tmp)
        print(*tmp)

    # drift / unequal xover
    ncomb = np.random.binomial(2*N, mu_combine)
    haplo_counts = np.random.multinomial(2*N - ncomb, haplo_freqs)
    for _comb in range(ncomb):
        # pick a post-selection diploid parent
        r = np.random.rand()
        s = 0.
        for i in range(len(geno_freqs)):
            for j in range(len(geno_freqs)):
                s += geno_freqs[i, j]
                if s >= r:
                    break
            if s >= r:
                break
        haplo_counts = list(haplo_counts)
        haplo_counts.append(1)
        if not bs:
            haplo_new = [a.copy() for a in haplos[i] + haplos[j]]
        else:
            haplo_new = haplos[i] + haplos[j]
        haplos.append(tuple(haplo_new))        

    extinct = [i for i, n in enumerate(haplo_counts) if n == 0]
    if extinct:
        haplo_counts = [a for i, a in enumerate(haplo_counts)
                        if i not in extinct]
        haplos = [a for i, a in enumerate(haplos) if i not in extinct]
        S = np.delete(S, extinct, axis=0)
        S = np.delete(S, extinct, axis=1)

    # mutation
    tot_alleles = sum(haplo_counts[i]*len(haplos[i])
                      for i in range(len(haplos)))
    nmuts = np.random.poisson(mu*tot_alleles)
    for i in range(nmuts):
        # Pick one haplo instance
        r = np.random.rand() * tot_alleles
        s = 0
        for idx, n in enumerate(haplo_counts):
            s += n*len(haplos[idx])
            if s >= r:
                break
        haplo_counts[idx] -= 1
        haplo_counts = list(haplo_counts)
        haplo_counts.append(1)
        if not bs:
            haplo_new = [a.copy() for a in haplos[idx]]
            gene_idx = np.random.randint(len(haplo_new))  # choose gene in haplo
            delta = sigmamu*np.random.randn(*haplos[idx][0].shape)
            haplo_new[gene_idx] += delta
        else:
            haplo_new = [a for a in haplos[idx]]
            gene_idx = np.random.randint(len(haplo_new))  # choose gene in haplo
            haplo_new[gene_idx] ^= 2**np.random.randint(nbits)  # xor one bit
        haplos.append(tuple(haplo_new))

    # deletion
    ndels = np.random.poisson(mu_del*tot_alleles)
    for i in range(ndels):
        # Pick one haplo instance
        r = np.random.rand() * tot_alleles
        s = 0
        for idx, n in enumerate(haplo_counts):
            s += n*len(haplos[idx])
            if s >= r:
                break
        if len(haplos[idx]) == 1:
            # don't delete if just one
            ndels -= 1
            continue
        haplo_counts[idx] -= 1
        haplo_counts = list(haplo_counts)
        haplo_counts.append(1)
        if not bs:
            haplo_new = [a.copy() for a in haplos[idx]]
        else:
            haplo_new = [a for a in haplos[idx]]
        gene_idx = np.random.randint(len(haplo_new))  # choose gene in haplo
        del haplo_new[gene_idx]
        haplos.append(tuple(haplo_new))

    # update S (survival matrix) if ther are any new haplos
    new_haps = nmuts + ndels + ncomb
    if new_haps:
        S = np.pad(S, (0, new_haps))
        for i in range(len(haplos)):
            for j in range(len(haplos)-new_haps, len(haplos)):
                s = Survival((haplos[i], haplos[j]), P, v, K)
                S[i, j] = s; S[j, i] = s
