#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

fname = 'aaindex1.rawproperties.txt'

# read data file #
res_ids  = []
raw_data = []
with open(fname, 'r') as handle:
    line = handle.readline()
    while line:
        linearr = line.split()
        if linearr[0] == 'I':
            if not res_ids:
                aasplits = linearr[1:]
                for aasplit in aasplits:
                    res = aasplit.split('/')[0]
                    res_ids.append(res)
                for aasplit in aasplits:
                    res = aasplit.split('/')[1]
                    res_ids.append(res)
            dsarr = handle.readline().split()
            dsarr.extend(handle.readline().split())
            if 'NA' not in dsarr:
                data = [ float(x) for x in dsarr ]
                raw_data.append(data)
            handle.readline()
            line = handle.readline()

# scale data to mean 0, var 1 #
scaled_data = []
scaler = StandardScaler()
for rd in raw_data:
    rda = np.array(rd).reshape((-1,1))
    scaler.fit(rda)
    nd = scaler.transform(rda).reshape((1,-1))
    scaled_data.append(nd[0])

# do PCA #
pca_cutoff = 0.90
pca = PCA(pca_cutoff)
pca.fit(scaled_data)

for i in range(len(res_ids)):
    sys.stdout.write(res_ids[i])
    row = pca.components_[:,i]
    for component in row:
        sys.stdout.write(' {:.4f}'.format(round(component,4)))
    sys.stdout.write('\n')
