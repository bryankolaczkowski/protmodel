################################################################################
# library for encoding amino-acid data
#
# (C) 2020 Bryan Kolaczkowski, University of Florida, Gainesville, FL USA
# Released under GNU General Public License (GPL)
# bryank@ufl.edu
################################################################################

import io
import collections
import numpy as np

################################################################################
# BEG DEFINE GLOBALS

## numpy data type to use for floats
FLOATTYPE = np.float32

## character indicating missing amino-acid residue
MISSINGAA = '.'

## secondary-structure code to integer mapping
#  must be consistent with biopython's Bio.PDB.DSSM
#  note that this is 'Q8' mapping of secondary structures
#   H 	Alpha helix (4-12)
#   B 	Isolated beta-bridge residue
#   E 	Strand
#   G 	3-10 helix
#   I 	Pi helix
#   T 	Turn
#   S 	Bend
#   - 	None
SSTRUCT_CODES  = ['H','B','E','G','I','T','S','-']
SSTRUCT_TO_INT = {}
for i in range(len(SSTRUCT_CODES)):
    SSTRUCT_TO_INT[SSTRUCT_CODES[i]] = i

## JTT's amino-acid exchange probabilities at P(0.01)
#  used to calculate amino-acid exchangabilities as a way
#  of assigning 'similarities' among amino-acid residues
JTT_P01_DATA="""#
# P(0.01), amino acid exchange data generated from SWISSPROT Release 22.0
# Ref. Jones D.T., Taylor W.R. and Thornton J.M. (1992) CABIOS 8:275-282
# A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A 0.98754 0.00030 0.00023 0.00042 0.00011 0.00023 0.00065 0.00130 0.00006 0.00020 0.00028 0.00021 0.00013 0.00006 0.00098 0.00257 0.00275 0.00001 0.00003 0.00194
R 0.00044 0.98974 0.00019 0.00008 0.00022 0.00125 0.00018 0.00099 0.00075 0.00012 0.00035 0.00376 0.00010 0.00002 0.00037 0.00069 0.00037 0.00018 0.00006 0.00012
N 0.00042 0.00023 0.98720 0.00269 0.00007 0.00035 0.00036 0.00059 0.00089 0.00025 0.00011 0.00153 0.00007 0.00004 0.00008 0.00342 0.00135 0.00001 0.00022 0.00011
D 0.00062 0.00008 0.00223 0.98954 0.00002 0.00020 0.00470 0.00095 0.00025 0.00006 0.00006 0.00015 0.00004 0.00002 0.00008 0.00041 0.00023 0.00001 0.00015 0.00020
C 0.00043 0.00058 0.00015 0.00005 0.99432 0.00004 0.00003 0.00043 0.00016 0.00009 0.00021 0.00004 0.00007 0.00031 0.00007 0.00152 0.00025 0.00016 0.00067 0.00041
Q 0.00044 0.00159 0.00037 0.00025 0.00002 0.98955 0.00198 0.00019 0.00136 0.00005 0.00066 0.00170 0.00010 0.00002 0.00083 0.00037 0.00030 0.00003 0.00008 0.00013
E 0.00080 0.00015 0.00025 0.00392 0.00001 0.00130 0.99055 0.00087 0.00006 0.00006 0.00009 0.00105 0.00004 0.00002 0.00009 0.00021 0.00019 0.00001 0.00002 0.00029
G 0.00136 0.00070 0.00035 0.00067 0.00012 0.00011 0.00074 0.99350 0.00005 0.00003 0.00006 0.00016 0.00003 0.00002 0.00013 0.00137 0.00020 0.00008 0.00003 0.00031
H 0.00021 0.00168 0.00165 0.00057 0.00014 0.00241 0.00016 0.00017 0.98864 0.00009 0.00051 0.00027 0.00008 0.00016 0.00058 0.00050 0.00027 0.00001 0.00182 0.00008
I 0.00029 0.00011 0.00020 0.00006 0.00003 0.00004 0.00007 0.00004 0.00004 0.98729 0.00209 0.00012 0.00113 0.00035 0.00005 0.00027 0.00142 0.00001 0.00010 0.00627
L 0.00023 0.00019 0.00005 0.00004 0.00005 0.00029 0.00006 0.00005 0.00013 0.00122 0.99330 0.00008 0.00092 0.00099 0.00052 0.00040 0.00015 0.00007 0.00008 0.00118
K 0.00027 0.00331 0.00111 0.00014 0.00001 0.00118 0.00111 0.00020 0.00011 0.00011 0.00013 0.99100 0.00015 0.00002 0.00011 0.00032 0.00060 0.00001 0.00003 0.00009
M 0.00042 0.00023 0.00013 0.00008 0.00006 0.00018 0.00011 0.00011 0.00007 0.00255 0.00354 0.00038 0.98818 0.00017 0.00008 0.00020 0.00131 0.00003 0.00006 0.00212
F 0.00011 0.00003 0.00004 0.00002 0.00015 0.00002 0.00003 0.00004 0.00009 0.00047 0.00227 0.00002 0.00010 0.99360 0.00009 0.00063 0.00007 0.00008 0.00171 0.00041
P 0.00148 0.00038 0.00007 0.00008 0.00003 0.00067 0.00011 0.00018 0.00026 0.00006 0.00093 0.00012 0.00004 0.00007 0.99270 0.00194 0.00069 0.00001 0.00003 0.00015
S 0.00287 0.00052 0.00212 0.00031 0.00044 0.00022 0.00018 0.00146 0.00017 0.00021 0.00054 0.00027 0.00007 0.00037 0.00144 0.98556 0.00276 0.00005 0.00020 0.00025
T 0.00360 0.00033 0.00098 0.00020 0.00008 0.00021 0.00020 0.00024 0.00011 0.00131 0.00024 0.00060 0.00053 0.00005 0.00060 0.00324 0.98665 0.00002 0.00007 0.00074
W 0.00007 0.00065 0.00003 0.00002 0.00023 0.00008 0.00006 0.00040 0.00002 0.00005 0.00048 0.00006 0.00006 0.00021 0.00003 0.00024 0.00007 0.99686 0.00023 0.00017
Y 0.00008 0.00010 0.00030 0.00024 0.00041 0.00010 0.00004 0.00006 0.00130 0.00017 0.00022 0.00005 0.00004 0.00214 0.00005 0.00043 0.00012 0.00010 0.99392 0.00011
V 0.00226 0.00009 0.00007 0.00016 0.00012 0.00008 0.00027 0.00034 0.00003 0.00511 0.00165 0.00008 0.00076 0.00025 0.00012 0.00026 0.00066 0.00004 0.00005 0.98761
"""

## 'reduced' amino-acid properties derived from the AAindex database
#   used to calculate amino-acid distances in multidimensional 'property space'
#   Original data are here: ftp://ftp.genome.jp/pub/db/community/aaindex/
#   There were 566 properties in the original AAindex database at time of
#   download (May, 2020). We reduced this to the featuers shown here by PCA;
#   these 11 features explain >90% of the variance in normalized
#   (mean=0,variance=1) properties found in AAindex
AANDX_RED_DATA = """#
# amino acid properties generated from AAindex file aaindex1 Feb/13/2017
# Ref. Kawashima, S., Pokarowski, P., Pokarowska, M., Kolinski, A.,
#      Katayama, T., and Kanehisa, M.; AAindex: amino acid index
#      database, progress report 2008. Nucleic Acids Res. 36, D202-D205
#      (2008). [PMID:17998252]
#  PC01    PC02    PC03    PC04    PC05    PC06    PC07    PC08    PC09    PC10    PC11
A -0.0086 -0.1348  0.4616 -0.0693  0.1245 -0.0974  0.1758  0.0425  0.2240  0.1468 -0.0912
R  0.1397  0.3740 -0.0290 -0.0032 -0.4022  0.3185  0.1816  0.3274 -0.1455  0.4340 -0.2502
N  0.2467 -0.0020 -0.1151  0.2287 -0.0322 -0.1215 -0.1339 -0.3356 -0.0033 -0.2867 -0.1238
D  0.3026  0.0614 -0.0023  0.0811  0.2997 -0.1711 -0.4602  0.0907 -0.1318  0.0992 -0.0255
C -0.1503 -0.1747 -0.1930  0.4813  0.4372  0.4583  0.1800  0.2741 -0.0785 -0.2256 -0.1828
Q  0.1270  0.2193  0.0536 -0.0103  0.0349  0.1008  0.0496 -0.0220  0.0287 -0.0725  0.4633
E  0.2014  0.2729  0.3149 -0.1269  0.3462 -0.1152 -0.2567  0.2072 -0.1719  0.0386 -0.0749
G  0.2582 -0.5223  0.0341  0.2440 -0.2662 -0.4457  0.2783  0.2544 -0.2370  0.1021  0.0928
H  0.0023  0.2077 -0.1324  0.1457  0.0366 -0.0474  0.2001 -0.5542 -0.2003  0.2052 -0.2896
I -0.3345 -0.1388  0.0757 -0.1287 -0.1306  0.0775 -0.2226  0.0080 -0.2496 -0.1335  0.1711
L -0.2815 -0.0861  0.3156 -0.2161 -0.0557 -0.0854  0.0425 -0.0703  0.1102 -0.2310 -0.4548
K  0.1992  0.3118  0.1614 -0.0439 -0.2640  0.0542  0.2554  0.0503  0.0733 -0.6013  0.1052
M -0.2663  0.1253  0.0926  0.0394  0.2740 -0.0860  0.3623 -0.2100 -0.1066  0.1995  0.4918
F -0.3107  0.0119 -0.1117 -0.0762 -0.0329 -0.1889 -0.0443 -0.1427 -0.0783  0.1215 -0.1622
P  0.2705 -0.2761 -0.4103 -0.7141  0.2161  0.1550  0.2039 -0.0110 -0.0497 -0.0300 -0.0256
S  0.2019 -0.1961  0.0362  0.1371 -0.0814  0.1391 -0.0797 -0.2065  0.3989  0.0274 -0.0091
T  0.0746 -0.1420  0.0049  0.0502 -0.1324  0.2717 -0.2266 -0.1473  0.4486  0.2869  0.1665
W -0.2810  0.2159 -0.3641  0.0102  0.0718 -0.3854  0.0207  0.3707  0.5100  0.0159 -0.0143
Y -0.1292  0.0776 -0.3624  0.0262 -0.2818 -0.1042 -0.2738 -0.0024 -0.1705 -0.1251  0.1127
V -0.2620 -0.2049  0.1697 -0.0551 -0.1616  0.2734 -0.2523  0.0767 -0.1707  0.0286  0.1008
"""

# END DEFINE GLOBALS
################################################################################
################################################################################
# BEG CLASS DEFINITIONS

# END CLASS DEFINITIONS
################################################################################
################################################################################
# BEG HELPER FUNCTIONS

## res_to_feature_helper
#  converts a feature_string to aa_to_aadist dictionary
def _res_to_feature_helper(feature_string, normalize=True, average=True):
    """convert a feature string to amino-acid -> distance dictionary"""

    # final return dictionary #
    aa_to_aadist = {}

    # number of features #
    feature_count = 0

    handle = io.StringIO(feature_string)
    for line in handle:
        if line[0] == '#':
            continue
        linearr = line.split()
        aa = linearr[0]
        aa_to_aadist[aa] = np.asarray([ float(x) for x in linearr[1:] ],
                                      dtype=FLOATTYPE)
        feature_count = aa_to_aadist[aa].size
    handle.close()

    # add entreis for selenocysteine (U = C)
    # and pyrrolysine (O = K)
    aa_to_aadist['U'] = aa_to_aadist['C']
    aa_to_aadist['O'] = aa_to_aadist['K']

    # add special entries for ambiguous amino-acids:
    #  X = any amino-acid
    #  B = D or N
    #  J = I or L
    #  Z = E or Q
    if average:
        aa_to_aadist['X'] = np.full(feature_count, 1.0/feature_count,
                                    dtype=FLOATTYPE)
    else:
        aa_to_aadist['X'] = np.full(feature_count, 0.0, dtype=FLOATTYPE)
    aa_to_aadist['B'] = np.mean([ aa_to_aadist['D'], aa_to_aadist['N'] ],
                                axis=0)
    aa_to_aadist['J'] = np.mean([ aa_to_aadist['I'], aa_to_aadist['L'] ],
                                axis=0)
    aa_to_aadist['Z'] = np.mean([ aa_to_aadist['E'], aa_to_aadist['Q'] ],
                                axis=0)

    # renormalize all amino-acid probability distributions to sum to 1.0
    if normalize:
        for aa in aa_to_aadist.keys():
            aa_to_aadist[aa] = aa_to_aadist[aa] / aa_to_aadist[aa].sum()

    # add special entry for 'missing' data "."
    # this will be an all-zero entry, so we can do sliding windows
    # over the first few real amino acids
    aa_to_aadist[MISSINGAA] = np.full(feature_count, 0.0, dtype=FLOATTYPE)

    return aa_to_aadist

# END HELPER FUNCTIONS
################################################################################
################################################################################
# BEG LIBRARY FUNCTIONS

## JTTP01dist
#  parse JTT's amino-acid exchange probabilities
#  (at P(0.01)) This is just to give us a rough estimate
#  of exchangability at each amino-acid. Similar to what
#  others have done using PSI-BLAST to generate variability
#  at a specific site, but we're not relying on evolutionary
#  relatedness to generate this 'variation' in the
#  amino-acid at each site.
def JTTP01dist():
    """Amino-acid exchangability 'distances' from JTT model at P(0.01)"""
    return _res_to_feature_helper(JTT_P01_DATA,
                                  normalize=True,
                                  average=True)

## AANDXred
#  parse a 'reduced' set of amino-acid features derived
#  from the AAIndex1 database of amino-acid properties
#  This is just a rough estimate of an amino-acid's location
#  in 'property space'. Amino-acid residues that are
#  'similar' to one another should be closer together in
#  this space, and vice-versa.
def AANDXred():
    """Amino-acid biochemical feature 'distances' from AAindex database"""
    return _res_to_feature_helper(AANDX_RED_DATA,
                                  normalize=False,
                                  average=False)

# END LIBRARY FUNCTIONS
################################################################################
################################################################################
# BEG LIBRARY VARIABLES

# map strings to functions in this module, so user can select a function
# to encode amino-acid features at runtime or from the command-line
AAENCODING_FNS = collections.OrderedDict()
AAENCODING_FNS['AANDXred']   = AANDXred
AAENCODING_FNS['JTTP01dist'] = JTTP01dist

# END LIBRARY VARIABLES
################################################################################
