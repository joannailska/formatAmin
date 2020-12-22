#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:31:48 2019

Script formatting the output of Stacks/genotypes (genotypes only, remember to remove the ID lines!) to fit pipeline resulting in Amin's GRM. 
The formatting includes filtering based on Mendelian inconsistencies and missingness (for now, remove markers missing in >50%). 
Specify whether Bonferroni correction is to be applied as first argument (0-no,1-yes)
Only bi-allelic markers are retained. 

5 Nov 2019

Adding filtering of the trees based on the connectedness to the rest of the population. For each tree, the number of SNPs in common with other trees is calculated. Then, they are sorted on the number of times that particular tree had less than desired number of SNPs. The top tree is removed, and counting is repeated, until no pairwise counts have less than desired threshold. 

@author: jilska2
"""

import sys as sys
import numpy as np
import pandas as pd
pd.set_option('precision', 3)
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from numba import njit
import argparse

def Mendel_chiSq_Bonferroni(row,p,n_snps):
    '''
    Function performing a chi-square test on the observed genotypes,
    given the cross type and its expected segregation probabilities.
    Given the number of SNPs, performs Bonferroni correction on the requested p-value
    '''
    exp={
    '<abxcd>':{'ac':0.25,'ad':0.25,'bc':0.25,'bd':0.25},
    '<efxeg>':{'ee':0.25, 'ef':0.25, 'eg':0.25, 'fg':0.25},
    '<hkxhk>':{'hh':0.25,'hk':0.5,'kk':0.25},
    '<lmxll>':{'ll':0.5,'lm':0.5},
    '<nnxnp>':{'nn':0.5,'np':0.5}}

    ## Extract the cross type
    cross=row['cross']
    
    ## Extract the genotypes
    genotypes=row.iloc[2:]
    
    ## Remove missing values
    test_row=genotypes.dropna().reset_index()
    test_row.columns=['ID','genotype']
    
    ## Count non-missing genotypes
    nonmis=test_row.shape[0]
    
    ## Count observed genotypes and put into a dictionary
    obs=test_row['genotype'].value_counts().to_dict()

    ## Remove observed genotypes not possible for a given cross.
    obs_keys_invalid=set(obs.keys())-set(exp[cross].keys())

    len(obs_keys_invalid)
    if len(obs_keys_invalid)>0:
        for x in obs_keys_invalid:
            del obs[x]

    ## Calculate the expected and observed values per genotype
    ch=[]

    for g in exp[cross].keys():
        ## Get the expected number scaled to the number of non-missing genotypes
        e=exp[cross][g]*nonmis
        try:
            ## Calculate the deviation
            ch.append((obs[g]-e)**2/e)
        except KeyError:
            ## If an expected genotype was not observed
            ch.append((0-e)**2/e)

    ## Sum the genotype deviations to get chi square for the cross. 
    chi=sum(ch)

    ## Calculate the number of degrees of freedom, i.e. the number of expected genotypes
    ## possible in a given cross - 1. 
    n=len(exp[cross].keys())-1

    # Find the critical value for 95% confidence*
    m = 1 - (p/n_snps)
    crit = stats.chi2.ppf(q = m, df=n)

    if chi>crit:
        outcome=0   # reject
    else:
        outcome=1   # accept
    
    return outcome

def unconnected_tree(df, thr):
    '''
    Function sorting the individuals by the number of times they pairwise comparison with their siblings fell below the threshold number of SNPs.
    Needs a matrix of the numbers of SNPs non-missing between pairs of sibs. 
    
    Returns an individual with the highest number of pairs below threshold.
    '''
    ids = df.index
    
    ## Convert to numpy array
    a = df.values
    
    ## Remove diagonals
    m = a.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = a.strides
    out = strided(a.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)
    ## Transpose so that the each individual is in a column
    out = out.T
    d = pd.DataFrame(data=out,    # values
                  columns=ids)

    ## Count for how many pairings an individual falls below the threshold in the number of SNPs used for calculation
    c_below = pd.DataFrame(columns=['ID','count_below'])
    c_below['ID']=d.columns
    c_below['count_below']=( d < thr ).sum(axis=0).reset_index()[0]

    ## Sort values, starting with an individual which fell below threshold most times
    cdf = c_below.sort_values(by='count_below', ascending=False)
    
    ## Count the number of individuals which fall below threshold at least once
    n = cdf.loc[cdf['count_below']!=0].shape[0]
    
    try:
        ## Print the top individual
        tree = cdf.iloc[0,0]
    except IndexError:
        tree='None'
    
    return tree, n


parser = argparse.ArgumentParser()
parser.add_argument("--p", default=0.05, help="Specify p-value to be used for chi2 test, default 0.05")
parser.add_argument("--miss", default=0.5, help="Specify missingness threshold, default 0.5")
parser.add_argument("--fam", default=1, help="Specify family, default family 1")
parser.add_argument("--inF", default='segMark_fam1.txt', help="Specify the input genotype file")
parser.add_argument("--Bonf", default=1, help="Specify whether to perform Bonferroni correction (default, 1) or not (0)")
parser.add_argument("--connect", default=1, help="Specify whether to remove unconnected individuals (default yes (1)")
parser.add_argument("--nSNP", default=500, help="Specify the number of SNPs required to accept connectedness level, default 500 (min)")


args = parser.parse_args()

p=float(vars(args)['p'])
mis=float(vars(args)['miss'])
f=int(vars(args)['fam'])
inFile=vars(args)['inF']
Bonf=vars(args)['Bonf']
connect=vars(args)['connect']
snps_req=int(vars(args)['nSNP'])

print("Family: ", f)       
print("Input file: ", inFile)    
print("Missing threshold: ",mis)
print("Probability threshold for chi-square test: ", p)
print("Bonferroni correction (1 - yes, 0 - no): ", Bonf)
print("Filtering unconnected individuals (1 - yes, 0 - no): ", connect)
print("Minimum number of SNPs required for accepted connectedness level: ", snps_req)

idsfile=inFile.replace("Mark_","MarkIDs_")
        
## Read in the IDs of the genotyped individuals
with open(idsfile) as sfil:
        ids=[x.replace('_trimAdapt.fil','') for x in sfil.read().splitlines()]
        
        
## Read in the full genotypes, using the IDs as column headers
df=pd.read_table(inFile, names=ids, na_values='--', skiprows=5)

## Relabel upper-case genotypes to lowercase
df = df.applymap(lambda s:s.lower() if type(s) == str else s)

start_Ntree=df.shape[1]-2
start_Nsnp=df.shape[0]
        
## Remove tri- and four-allelic loci
df=df.loc[df['cross']!='<efxeg>'] 
erem=df.shape[0]
df=df.loc[df['cross']!='<abxcd>']
Nsnp=df.shape[0]
        

print("Starting with {} markers and {} trees".format(start_Nsnp,start_Ntree))
print("*********************************************")
print("Removed:")
print("{} tri-allelic markers (<efxeg>)".format(start_Nsnp-erem))
print("{} four-allelic markers (<abxcd>)".format(erem-Nsnp))

## Filter out Mendelian inconsistencies
sel = df.copy()
if Bonf == '0':
    bn=1
else:
    bn=Nsnp
    
sel['Mendel']=sel.apply(Mendel_chiSq_Bonferroni, args=(p,bn), axis=1)
filtM = sel.loc[sel['Mendel']==1]
filtM.drop(columns=['Mendel'], inplace=True)
mend_Nsnp = filtM.shape[0]

print("SNPs removed due to Mendelian inconsistencies: ", Nsnp-mend_Nsnp)
Nsnp = mend_Nsnp

## Filter out markers missing in more than threshold
filt = filtM[filtM.isnull().sum(axis=1) < mis*start_Ntree]
mend_Nsnp = filt.shape[0]
print("SNPs removed due to missingness > {} : {}".format(mis*start_Ntree, Nsnp-mend_Nsnp))
Nsnp = mend_Nsnp

## Filter out trees with poor connectedness to the rest of the population
dfn = filt.drop(columns=['cross']).set_index('marker')
Nsnp = dfn.shape[0]

## Remove individuals which have < than the required number of SNPs. 
## If their non-missing SNPs are < threshold, there is not chance they will have more in a pairwise comparison!
n_itself=pd.Series(Nsnp - dfn.isnull().sum(axis=0), index = dfn.columns).reset_index()
removed=list(n_itself.loc[n_itself[0]<snps_req]['index'])

print ("Number of trees removed as their number of non-missing genotypes is lower than threshold: ", len(removed))

df2 = dfn.drop(removed, axis=1)

dfn = df2.T

# Define (numba optimized) distance function that counts the number of markers that are both not NaN
@njit
def dist(a, b):
    return ((~a) & (~b)).sum()

# Use scipy.spatial.distance.pdist to calculate all pairwise distances using our distance function and display values as a square matrix
# Note we pass df.isnull() to pdist() so that the distance function receives True for every NaN
# Annoyingly, pdist assumes the distance between an individual and itself is 0, and therefore doesn't bother to calculate it 
c = squareform(pdist(dfn.isnull(), dist))
ids = dfn.index
counts = pd.DataFrame(data=c,    # values
              index=ids,    # 1st column as index
              columns=ids)

counts.head()

top, n = unconnected_tree(counts, snps_req)
# print("Removing the top unconnected individual: ",top)

## As long as some individual(s) fall below threshold, re-do the calculation
while n>0:
    removed.append(top)

    ## Remove the top ID
    q = counts.drop(columns=[top]).drop([top])

    counts = q

    ## Re-do the calculation
    top, n = unconnected_tree(counts, snps_req)
    
print("In total, {} trees were responsible for pairwise comparisons with < {} SNPs in common.\nThe genotypes of these trees will be removed.".format(len(removed), snps_req))

filt=filt.drop(columns=removed)
removed.sort()

rem = open("unconnected_fam{}.txt".format(f), 'w')
rem.write("\n".join(removed))
rem.close()

## Final number of SNPs used
print("*********************************************")
print("Final numbers retained for further analysis:")
print("Retained {} bi-allelic SNPs for further analysis".format(Nsnp))
print("Retained {} trees for further analysis".format(filt.shape[1]))
print("*********************************************")
        
## Formating
## Change marker label to a string
form = filt.copy()
form['marker'] = 'M_' + form['marker'].astype(str)

## Add parental genotypes
form['P1']=form['cross'].str.split("x", n = 1, expand = True)[0].str.replace('<','')
form['P2']=form['cross'].str.split("x", n = 1, expand = True)[1].str.replace('>','')

## Reorder columns, so that P1 and P2 are before the offspring
cols = list(form)
cols.insert(1,cols.pop(cols.index('P1')))
cols.insert(2,cols.pop(cols.index('P2')))
form = form.loc[:,cols]

## Drop cross column
form.drop(columns=['cross'], inplace=True)

## Save to a file
outname=inFile.replace("segMark","gtpAmin").replace("_corrected","")
form.to_csv(outname, index=False, sep=' ', na_rep='--')
print("Genotypes saved in {}".format(outname))
        
## Write out a pedigree file
outped = outname.replace(".txt","_ped.txt")
ped = pd.DataFrame(columns=['tree','sire','dam'])
ped['tree']=list(form)[1:]
ped['sire']='P1'
ped['dam']='P2'
ped.loc[ped['tree']=='P1',['sire','dam']]='0'
ped.loc[ped['tree']=='P2',['sire','dam']]='0'

ped.to_csv(outped, index=False, header=False, sep=' ', na_rep='')
print("Pedigree saved in {}".format(outped))
