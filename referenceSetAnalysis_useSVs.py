#!/usr/bin/env python 
""" 
This module is for getting summary analyses on adcl, edpl, richness, eveness, Shannon-index alpha diversity and Bray-curtis beta-diversity 
"""
import argparse
from Bio import SeqIO, Entrez, pairwise2
from Bio.SeqRecord import SeqRecord
import re, time
import os, sys, glob
import pandas as pd
import numpy as np
import random
import uuid
import numpy as np
import pandas as pd
from skbio.tree import TreeNode
from skbio import read
from skbio.stats.distance import DistanceMatrix
import matplotlib as mpl
from scipy import stats
from ast import literal_eval
import sqlite3

""" set up fonts here before importing matplotlib.pylab """
parms = {'font.family': 'serif', 'font.serif': 'Palatino', 'svg.fonttype': 'none'}
mpl.rcParams.update(parms)
from matplotlib import pyplot as pl
import seaborn as sbs
sbs.set(context='talk', style='darkgrid', palette='deep', font='sans-serif')
#pearson R-squared
def pr2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

mock_seqtab=pd.read_csv("/mnt/z/ResearchHome/ClusterHome/ralsalla/CuratedRef_4th/CC11.map.SeqTable.csv",index_col=0)
df_sv_list_names=['mock','RDP_10398','RDP_5224','RDP_1017','RDP_92','RDP_12']
taxaFiles=["/mnt/z/ResearchHome/ClusterHome/ralsalla/CuratedRef_2nd/with_SVs_fasta_RDP/AbundanceOfTaxIdInSamples_primaryTaxid.csv"]+[d+"/analysis/oneRankEachSV_keepBest.csv" for d in df_sv_list_names[1:]]
nDlists=len(df_sv_list_names)


def percentageCorrect(mock_seqtab, taxaFiles,df_sv_list_names, withMultiplicity=True):
    mock=mock_seqtab.copy()
    #not all tax_ids in the mock are primary
    translate={415850:1463164,195041:45634,592977:1680, 796939:796937, 41791:126333}
    mock.ncbi_tax_id.replace(translate, inplace=True)
    mock.rename(columns={'sourceSeq':'colind'}, inplace=True)
    mock = mock[['community', 'colind','organism', 'ncbi_tax_id', 'multiplicity']]
    mock_gpbySample = mock.groupby('community', as_index=True)
    sample_ids=[list(mock_gpbySample)[i][0] for i in range(len(list(mock_gpbySample)))]
    df_pcorrect=pd.DataFrame(index=sample_ids)
    for i, f in enumerate(taxaFiles[1:],1):
        temp = pd.read_csv(f, index_col=0) #multisample assignment 
        for s in  sample_ids:
            merged=mock_gpbySample.get_group(s).merge(temp[['tax_id', 'tax_name', 'colind']+[s]], how='left', on='colind') #merge on SVs=sourceSeq
            merged.loc[:,"iscorrect"]=(merged['ncbi_tax_id'].astype(str)==merged['tax_id'].astype(str)).astype(int)
            if withMultiplicity:
                df_pcorrect.loc[s,df_sv_list_names[i]]=((merged['iscorrect']*merged['multiplicity'])/merged['multiplicity'].sum()).sum()*100.0 #percentage correct
            else:
                df_pcorrect.loc[s,df_sv_list_names[i]]=(merged['iscorrect']).mean()*100.0 #percentage correct
    return sample_ids, df_pcorrect

sample_ids, df_pcorrect=percentageCorrect(mock_seqtab, taxaFiles,df_sv_list_names)


def alphaDiversity(sample_ids, df_sv_list_names):
    df_rooted_qd_0=pd.DataFrame(index=sample_ids)
    df_rooted_qd_1=pd.DataFrame(index=sample_ids)
    df_rooted_qd_2=pd.DataFrame(index=sample_ids)
    df_phylo_entropy=pd.DataFrame(index=sample_ids)
    for n in df_sv_list_names[1:]:
        for s in sample_ids:
            temp=pd.read_csv(n+"/analysis/"+s+".alphaDiversity.csv")
            df_rooted_qd_0.loc[s,n]=temp['rooted_qd_0'].values
            df_rooted_qd_1.loc[s,n]=temp['rooted_qd_1'].values
            df_rooted_qd_2.loc[s,n]=temp['rooted_qd_2'].values
            df_phylo_entropy.loc[s,n]=temp['phylo_entropy'].values
    return df_rooted_qd_0, df_rooted_qd_1, df_rooted_qd_2, df_phylo_entropy

df_rooted_qd_0, df_rooted_qd_1, df_rooted_qd_2, df_phylo_entropy = alphaDiversity(sample_ids, df_sv_list_names)

def prepareFiles():
    df_sv_list=[]
    for i, f in enumerate(taxaFiles):
        if i==0: #mock
            mock=mock_seqtab.copy()
            sample_ids=mock['community'].unique()
            sample_ids=['CC11CM'+str(i) for i in range(sample_ids.shape[0])]
            #not all tax_ids in the mock are primary
            translate={415850:1463164,195041:45634,592977:1680, 796939:796937, 41791:126333}
            mock.ncbi_tax_id.replace(translate, inplace=True)
            mock.rename(columns={'sourceSeq':'colind','organism':'tax_name','ncbi_tax_id':'tax_id'}, inplace=True)
            sv_ids = mock.colind.unique()
            temp=pd.DataFrame(index=sv_ids,columns=['tax_id']+sample_ids)
            for s in sample_ids:
                mock_s = mock[mock.community==s]
                mock_s.set_index('colind', inplace=True)
                temp.loc[mock_s.index, 'tax_id']=mock_s['tax_id']
                temp.loc[mock_s.index, s]=mock_s['multiplicity']
            temp=temp.fillna(0)
            
        else: #analyzed using dada2/pplacer/RDP
            temp=pd.DataFrame(index=sv_ids,columns=['tax_id']+sample_ids)
            temp1=pd.read_csv(f)
            #drop rank, taxa_name and colind (SV index)
            temp1 = temp1.loc[:,['colind','tax_id']+sample_ids]
            temp1.set_index('colind',inplace=True)
            temp.loc[temp1.index,'tax_id']=temp1['tax_id']
            temp.loc[temp1.index,sample_ids]=temp1[sample_ids]
            temp=temp.fillna(0)

        ##very strange tax_id for the mock is not duplicated but when I set the index as tax_id for the mock the index and the mock becomes duplicated!
        df_sv_list.append(temp)
        #print(temp.head())
    return df_sv_list

df_sv_list=prepareFiles()



dircs=[i+"/" for i in df_sv_list_names[1:]]
prefix='CC11CM'
def get_taxa_adcl(directory, prefix):
    """ gather adcl and bestRankStats files under analysis directory and return
        a dataframe with adcl and tax_id/tax_name for each amplicon/SV that has both (successfully placed SV)"""

    samples = glob.glob(directory+"/analysis/"+prefix+"*.adcl.csv")

    df = pd.DataFrame({'adcl':[],'achieved_rank':[]})
    df_summary=pd.DataFrame(index=range(len(samples)))
    for i,f in enumerate(samples):
        s=os.path.basename(f).split(".adcl.csv")[0]
        #adcl
        adcl=pd.read_csv(directory+"analysis/"+s+".adcl.csv",header=None)
        adcl.columns=['name','adcl','multiplicity']
        #edpl
        edpl=pd.read_csv(directory+"analysis/"+s+".edpl.csv",header=None)
        edpl.columns=['name','edpl']
        #pplacer stats: richness of placements and min_distal_length
        pplacer_stats=pd.read_csv(directory+"analysis/"+s+"_pplaceStats.csv")
        pplacer_stats.columns=['name','placeRichness','min_distL']
        #best rank
        bestRank=pd.read_csv(directory+"/analysis/"+s+"_bestRankStats.csv", index_col=False)
        #name,rank,tax_id,tax_name,likelihood,achieved_rank,ranks_off
        bestRank.drop('index',inplace=True, axis=1)
        print("number of reads without adcl=",bestRank.shape[0]-adcl.shape[0])
        df_summary.loc[i,'N_tot']=bestRank.shape[0]
        df_summary.loc[i,'N_achieved']=bestRank[bestRank['ranks_off']==0].shape[0]
        df_summary.loc[i,'N_achievedGenus']=bestRank[bestRank['ranks_off']==1].shape[0]
        df_summary.loc[i,'N_achievedFamily']=bestRank[bestRank['ranks_off']==2].shape[0]
        df_summary.loc[i,'N_achievedOrder']=bestRank[bestRank['ranks_off']==3].shape[0]
        df_summary.loc[i,'N_off']=bestRank[bestRank['ranks_off']>0].shape[0]
        df_summary.loc[i,'N_missed'] = bestRank['ranks_off'].isnull().sum()
        df_summary.loc[i,'Avrlikelihood_achieved']=bestRank[bestRank['ranks_off']==0]['likelihood'].mean()
        df_summary.loc[i,'Avrlikelihood_achievedGenus']=bestRank[bestRank['ranks_off']==1]['likelihood'].mean()
        df_summary.loc[i,'Avr_rankOff']=bestRank[bestRank['ranks_off']>0]['ranks_off'].mean()
    
        merged=bestRank.merge(adcl, on='name', how='left')
        merged = merged.merge(edpl, on='name', how='left')
        merged = merged.merge(pplacer_stats, on='name', how='left')
        merged = merged[['name','tax_id','achieved_rank','adcl','edpl','placeRichness','min_distL']]
        merged.loc[:,'runDir']=directory
        
        df=df.append(merged)
    
    ranks=['species','genus','family','order','class','subphylum','phylum','superkingdom']
    values=[1,2,3,4,5,5.5,6,7]
    fig,ax=pl.subplots(figsize=(12,12))
    g= sbs.violinplot(x='achieved_rank',y='adcl', order=ranks, data=df, ax=ax)
    g.set_xticklabels(labels=ranks, rotation=20)
    #pl.savefig(directory+"analysis/violinpl_adcl_verRank.png")
    return df


def getUniqueSet(alltaxids):
    out=set(alltaxids[0])
    for l in alltaxids[1:]:
        out=out.union(l)
    return out
allsv=[df_sv_list[i].index.astype(str) for i in range(nDlists)]
allt=[df_sv_list[i].tax_id.astype(str) for i in range(nDlists)]
alls=[df_sv_list[i].columns.astype(str) for i in range(nDlists)]
alltaxa=getUniqueSet(allt)
allsvs=getUniqueSet(allsv)
allsamples=getUniqueSet(alls)

alln=[get_taxa_adcl(d, prefix).name for d  in dircs]
allnames=getUniqueSet(alln)


##""""""""" here we compute pplacer stats of assignment by sample """""##########
def plot_summary_bySV(dircs, prefix, allnames):
    df_adcl=pd.DataFrame(index=list(allnames))
    df_edpl=pd.DataFrame(index=list(allnames))
    df_prichness=pd.DataFrame(index=list(allnames))
    df_mindistL=pd.DataFrame(index=list(allnames))
    for i,d in enumerate(dircs):
        df = get_taxa_adcl(d, prefix)
        #adcl
        adcl = df[['name','adcl']]
        adcl = adcl.drop_duplicates('name')
        adcl = adcl.set_index("name")
        df_adcl = df_adcl.merge(adcl, how='left', left_index=True, right_index=True) 
        df_adcl.rename(columns={'adcl':d.split("/")[0]},inplace=True)
        #edpl
        edpl = df[['name','edpl']]
        edpl = edpl.drop_duplicates('name')
        edpl = edpl.set_index("name")
        df_edpl = df_edpl.merge(edpl, how='left', left_index=True, right_index=True) 
        df_edpl.rename(columns={'edpl':d.split("/")[0]},inplace=True)
        #prichness
        prichness = df[['name','placeRichness']]
        prichness = prichness.drop_duplicates('name')
        prichness = prichness.set_index("name")
        df_prichness = df_prichness.merge(prichness, how='left', left_index=True, right_index=True) 
        df_prichness.rename(columns={'placeRichness':d.split("/")[0]},inplace=True)
        #mindistL
        mindistL = df[['name','min_distL']]
        mindistL = mindistL.drop_duplicates('name')
        mindistL = mindistL.set_index("name")
        df_mindistL = df_mindistL.merge(mindistL, how='left', left_index=True, right_index=True) 
        df_mindistL.rename(columns={'min_distL':d.split("/")[0]},inplace=True)

    #adcl
    fig, ax = pl.subplots(figsize=(12,12))
    #show violinplot (letter-value plot) to view more quantiles and adequately view the distribution
    #more info at https://vita.had.co.nz/papers/letter-value-plot.pdf
    g= sbs.violinplot(data=df_adcl,ax=ax)
    ax.set_ylabel("adcl")
    #fig.savefig("violinpl_adcl_byRefPkg.png")
    #edpl
    fig, ax = pl.subplots(figsize=(12,12))
    #show violinplot (letter-value plot) to view more quantiles and adequately view the distribution
    #more info at https://vita.had.co.nz/papers/letter-value-plot.pdf
    g= sbs.violinplot(data=df_edpl,ax=ax)
    ax.set_ylabel("edpl")
    #fig.savefig("violinpl_edpl_byRefPkg.png")
    #prichness
    fig, ax = pl.subplots(figsize=(12,12))
    #show violinplot (letter-value plot) to view more quantiles and adequately view the distribution
    #more info at https://vita.had.co.nz/papers/letter-value-plot.pdf
    g= sbs.violinplot(data=df_prichness,ax=ax)
    ax.set_ylabel("Richness of placement")
    #fig.savefig("violinplot_prichness_byRefPkg.png")
    #mindistL
    fig, ax = pl.subplots(figsize=(12,12))
    #show violinplot (letter-value plot) to view more quantiles and adequately view the distribution
    #more info at https://vita.had.co.nz/papers/letter-value-plot.pdf
    g= sbs.violinplot(data=df_mindistL,ax=ax)
    ax.set_ylabel("Minimum distal length across placements")
    #fig.savefig("violinpl_mindistL_byRefPkg.png")

    return df_adcl, df_edpl, df_prichness, df_mindistL


df_SVs_adcl, df_SVs_edpl, df_SVs_prichness, df_SVs_mindistL = plot_summary_bySV(dircs,prefix,allnames)
#df_SVs_adcl.to_csv("adcl_bySV_allsamples.csv")
#df_SVs_edpl.to_csv("edpl_bySV_allsamples.csv")
#df_SVs_prichness.to_csv("prichness_bySV_allsamples.csv")
#df_SVs_mindistL.to_csv("mindistL_bySV_allsamples.csv")

def get_measure_bySample(df_measure_bySV, df_sv_list_names, mock_seqtab):
    #copy
    out=df_measure_bySV.copy()
    #make index a column named 'index' to be able to use apply
    out.reset_index(inplace=True)
    out.loc[:,'sample'] =out['index'].apply(lambda row: row.split("SCR")[0]) 
    out.loc[:,'seqID']=out['index'].apply(lambda row: row.split("_From")[0])

    mock = mock_seqtab[['community', 'sourceSeq', 'seqID','multiplicity']]
    mock.rename(columns={'community':'sample'}, inplace=True)
    merged = out.merge(mock, how='left', on=['sample','seqID'])
    out=merged[['sample']+df_sv_list_names[1:]]
    gpbySample=out.groupby('sample')
    out=gpbySample.describe(percentiles=[0.025,0.25,0.50,0.75,0.975])

    df_weightedAvr=pd.DataFrame(index=merged['sample'].unique(), columns=[(i,'WAvr') for i in df_sv_list_names[1:]])
    for s in merged['sample'].unique():
        merged_s = merged[merged['sample']==s]
        for l in df_sv_list_names[1:]:
            WAvr = merged_s.loc[:,l].values*merged_s.loc[:,'multiplicity'].values/merged_s.loc[:,'multiplicity'].sum()
            df_weightedAvr.loc[s,(l,'WAvr')]=WAvr.sum()

    out = out.merge(df_weightedAvr, how='left', left_index=True, right_index=True)
    return out 

df_sample_adcl=get_measure_bySample(df_SVs_adcl,df_sv_list_names, mock_seqtab)
df_sample_edpl=get_measure_bySample(df_SVs_edpl,df_sv_list_names, mock_seqtab)
df_sample_prichness=get_measure_bySample(df_SVs_prichness,df_sv_list_names, mock_seqtab)
df_sample_mindistL=get_measure_bySample(df_SVs_mindistL,df_sv_list_names, mock_seqtab)



#########################Using skbio functions##################################
from skbio.diversity import alpha_diversity
from skbio.diversity import beta_diversity
def diversity(df_sv_list):
    """ use skbio to compute different diversity metrics"""

    richness=pd.DataFrame(index=allsamples)
    shannon=pd.DataFrame(index=allsamples)
    bc_dm_list=[]
    for i, df in enumerate(df_sv_list):
        data=df.iloc[:,1:].T.values #columns are the SVs and rows are the samples
        ids=df.columns[1:] #ids should have the same order as the data rows
        #richness
        richness = richness.merge(pd.DataFrame(alpha_diversity("observed_otus", data, ids)), how="left", left_index=True, right_index=True)
        richness.rename(columns={0:df_sv_list_names[i]}, inplace=True)
        #shannon
        shannon = shannon.merge(pd.DataFrame(alpha_diversity("shannon", data, ids)), how="left", left_index=True, right_index=True)
        shannon.rename(columns={0:df_sv_list_names[i]}, inplace=True)
        #bray-curtis distance matrix:
        bc_dm = beta_diversity("braycurtis", data, ids)
        temp_bc=pd.DataFrame(index=bc_dm.ids, columns=bc_dm.ids)
        temp_bc.iloc[:,:]=bc_dm.data
        bc_dm_list.append(temp_bc)
    return richness, shannon, bc_dm_list

def diversity_analysis(wu_dm_list,bc_dm_list):
    from skbio.stats.distance import mantel
    #do the UniFrac and  Bray-Curtis distances correlate? 
    r, p_value, n = mantel(wu_dm_list[0],bc_dm_list[0])
    print("Mantel Correlation COEF=",r)
    print("At significance of 0.05, the p-value for the correlation is = ",p_value)
    #next perform principle coordinate analysis (PCoA) on the weighted UniFrac distance matrix:
    from skbio.stats.ordination import pcoa
    wu_pc = pcoa(wu_dm_list[0])
    #then you can inspect clustering of sample-to-sample distances by specific sample characteristics. Say we have
    # a sample meta data dataframe: sample_md with indices as samples and columns as features (characteristics) 
    # if we want to inspect clustering of wu distances by say feature1 then:
    #fig = wu_pc.plot(sample_md, 'feature1', axis_labels=('PC 1', 'PC 2', 'PC 3'), title='Samples colored by feature1', cmap='jet', s=50)
    #For further analysis see http://scikit-bio.org/docs/0.4.1/diversity.html


def comp_expected_computed(richness, shannon, bc_dm_list):
    #richness
    grid = sbs.PairGrid(data = richness, vars=richness.columns, size=1)
    grid = grid.map_lower(sbs.jointplot, kind="reg", stat_func=pr2)
    #shannon
    grid = sbs.PairGrid(data = shannon, vars=shannon.columns, size=1)
    grid = grid.map_lower(sbs.jointplot, kind="reg", stat_func=pr2)
    #bray-curtis
    for i, dm in enumerate(bc_dm_list):
        dd = dm.values #nsamples by nsamples 
        dd = np.tril(dd, k=-1) #get the distinct distances by taking the left triagonal, including  the diagonal zeros
        dd = dd[dd>0] #get rid of the diagonal zeros
        if i==0:
            df_bc = pd.DataFrame({df_sv_list_names[0]:dd})
        else:
            df_bc.loc[:,df_sv_list_names[i]]=dd
    grid = sbs.PairGrid(data = df_bc, vars=df_bc.columns, size=1)
    grid = grid.map_lower(sbs.jointplot, kind="reg", stat_func=pr2)



richness, shannon, bc_dm_list = diversity(df_sv_list)
comp_expected_computed(richness, shannon, bc_dm_list)

##""""""""" here we compute the accuracy of assignment by sample """""##########

#########################computing Bray-curtis and RMS using two taxa tables for each sample#########
#############the taxa tables would have the same tax_ids aligned in the index and 
############# the same samples aligned in columns 

def bray_curtis_distance(table1, table2, sample_id):
    """non phylogenetic diversity measure 
       table1 and table2 are pandas dataframes with exactly the same indices and columns 
       columns as samples and indices as taxids or SVs"""
    numerator = 0
    denominator = 0
    sample_counts1 = table1[sample_id]
    sample_counts2 = table2[sample_id]
    for sample_count1, sample_count2 in zip(sample_counts1, sample_counts2):
        numerator += abs(sample_count1 - sample_count2)
        denominator += sample_count1 + sample_count2
    return numerator / denominator

def sumsqr_distance(table1, table2, sample_id):
    """ the rms is defined as the square root of the sum of the squared
    difference between relative abundances in different tables for the same
    sample (e.g. different pipelines) divided by the number of taxa """
    numerator=0
    sample_counts1 = table1[sample_id]/table1[sample_id].sum() #relative abundance
    sample_counts2 = table2[sample_id]/table2[sample_id].sum() #relative abundance
    for sample_count1, sample_count2 in zip(sample_counts1, sample_counts2):
        numerator += (sample_count1 - sample_count2)**2
    return numerator 


def tables_to_distance(table1, table2, pairwise_distance_fn):
    """ returns a dataframe of samples as indices and one column specifying distance b/w
        the two tables data for each sample"""
    sample_ids = table1.columns
    assert np.all(table1.columns.values==table2.columns.values), "tables are not adequately formatted"
    assert np.all(table1.index.values==table2.index.values), "tables are not adequately formatted"
    num_samples = len(sample_ids)
    data = pd.DataFrame(index=sample_ids) 
    for i, sample_id in enumerate(sample_ids):
        data.loc[sample_id,'distance']=pairwise_distance_fn(table1, table2, sample_id)
    return data

def get_bc_rms(allsvs, df_sv_list, df_sv_list_names):
    """ distances between table1 taken as the mock and table2 taken as full, setA1, setB1, setC1
        one at a time. The taxa tables table1 and table2 would have the same tax_ids aligned 
        in the index and the same samples aligned in columns """
    mock=df_sv_list[0].iloc[:,1:] #to exclude tax_id column
    df_merge=pd.DataFrame(index=list(allsvs))
    mock_tab1=df_merge.merge(mock,how='left', left_index=True, right_index=True)
    mock_tab1=mock_tab1.fillna(0)
    df_bc_all=pd.DataFrame(index=mock_tab1.columns.values)
    df_rms_all=pd.DataFrame(index=mock_tab1.columns.values)
    for i, df in enumerate(df_sv_list[1:],1):
        ana_tab2=df_merge.merge(df.iloc[:,1:], how="left", left_index=True, right_index=True)
        ana_tab2=ana_tab2.fillna(0)
        tmp_bc=tables_to_distance(mock_tab1,ana_tab2, bray_curtis_distance)
        df_bc_all=df_bc_all.merge(tmp_bc, how="left", left_index=True, right_index=True) 
        df_bc_all.rename(columns={'distance':df_sv_list_names[i]}, inplace=True)
        tmp_ss=tables_to_distance(mock_tab1,ana_tab2,sumsqr_distance)
        tmp_ss=np.sqrt(tmp_ss/len(allsvs))
        df_rms_all = df_rms_all.merge(tmp_ss, how='left',left_index=True, right_index=True)
        df_rms_all.rename(columns={'distance':df_sv_list_names[i]}, inplace=True)

    #bc between analysis routes
    grid = sbs.PairGrid(data = df_bc_all, vars=df_bc_all.columns, size=1)
    grid = grid.map_lower(sbs.jointplot, kind="reg", stat_func=pr2)
    #rms between analysis routes
    grid = sbs.PairGrid(data = df_rms_all, vars=df_rms_all.columns, size=1)
    grid = grid.map_lower(sbs.jointplot, kind="reg", stat_func=pr2)
    #plot summary of rms per subset package:
    fig, ax = pl.subplots(figsize=(12,12))
    sbs.violinplot(data=df_rms_all, ax=ax)
    ax.set_ylabel("RMS")
    #fig.savefig("rms_dist_comp.png")

    return df_bc_all, df_rms_all

df_bc_all, df_rms_all = get_bc_rms(allsvs, df_sv_list, df_sv_list_names)

###calculating rank off from mock taxa:
##first I will save a dataframe of all unique taxa with a column named tax_id:
df=pd.DataFrame({"tax_id":list(alltaxa)})
#second save it as csv to be used by get_taxa_fromTaxaDB.py to get full taxonomies of all taxids:
#df.to_csv("uniqueTaxaAcrossMockAndPackages.csv")
#run python get_taxa_fromTaxaDB.py -db /home/ralsalla/set88analysis_2ndTry/curating_ncbi_ncbi/taxtastic_csvFs/taxonomy.db -l uniqueTaxaAcrossMockAndPackages.csv -o ALL_lineageIds.csv to get CSV file for all parents taxids for the taxids in uniqueTaxaAcrossMockAndPackages.csv
alltaxa_df = pd.read_csv("ALL_lineageIds.csv") #tax_id and all of known parents
alltaxa_df.set_index('tax_id', inplace=True)
alltaxa_df.index = alltaxa_df.index.astype(str)
#taxaDB="/home/ralsalla/set88analysis_2ndTry/curating_ncbi_ncbi/taxtastic_csvFs/taxonomy.db"
taxaDB="/mnt/z/ResearchHome/ClusterHome/ralsalla/set88analysis_2ndTry/curating_ncbi_ncbi/taxtastic_csvFs/taxonomy.db"

def get_tax_data(taxid):
    """once we have the taxid, we can fetch the record"""
    fitched=False
    while ~fitched:
        try:
            time.sleep(8)
            search = Entrez.efetch(id = taxid, db = "taxonomy", retmode = "xml")
            fitched=True
        except:
            time.sleep(8)
            fitched=False

    return Entrez.read(search)


def get_lineage_ids_fromdata(data, uprank):
    """once you have the data from get_tax_data fetch the lineage"""
    #uprank=['kingdom','phylum','class','order','family','genus','species']
    lineage_toparse = data[0]['LineageEx']
    lineage=dict()
    ids=dict()
    for l in lineage_toparse:
        for r in uprank:
            try:
                if l['Rank']==r:
                    lineage[r]=l['ScientificName']
                    ids[r]=l['TaxId']
            except:
                pass
    return lineage, ids

def get_lineage_ids(taxid, conn):
    """ This function gets the names and ids if all parents of the given id """

    query="SELECT nd.tax_id, nd.parent_id, nd.rank, na.tax_id, na.tax_name, na.name_class from nodes nd inner join names na on nd.tax_id=na.tax_id where na.name_class=='scientific name' AND na.tax_id==" + "'"+taxid+"'"

    df = pd.read_sql_query(query, conn)
    #print(df)

    df.columns=['tax_id', 'parent_id', 'rank', 'tax_id_drop', 'tax_name', 'name_class']
    df.drop("tax_id_drop",axis=1, inplace=True)
    rankorder=np.array(['no_rank','superkingdom','phylum','class','order','family','genus','species'])[::-1]
    #print(df['rank'])
    if not df['rank'].iloc[0].strip() in rankorder:
        rankorder=np.append(rankorder,df['rank'].iloc[0])
    rank_ind=np.where(df['rank'].iloc[0]==rankorder)[0][0]
#    if len(df.tax_name.iloc[0].split(" "))>=2:
#        lineage={rankorder[rank_ind]:" ".join(df.tax_name.iloc[0].split(" ")[1:])}
#    else:
#        lineage={rankorder[rank_ind]:df.tax_name.iloc[0]}

    lineage={rankorder[rank_ind]:df.tax_name.iloc[0]}
    ids={rankorder[rank_ind]: taxid}
    stop=False
    temp=df.copy()
    #print(lineage, ids)
    while not stop:
        parent_id=temp['parent_id'].iloc[0]
        if parent_id is None or parent_id=="" or parent_id=='0':
            stop=True
            break
        #print(parent_id)
        query="SELECT nd.tax_id, nd.parent_id, nd.rank, na.tax_id, na.tax_name, na.name_class from nodes nd inner join names na on nd.tax_id=na.tax_id where na.name_class=='scientific name' AND na.tax_id==" + "'"+parent_id+"'"
        temp = pd.read_sql_query(query, conn)
        temp.columns=['tax_id', 'parent_id', 'rank', 'tax_id_drop', 'tax_name', 'name_class']
        temp.drop("tax_id_drop",axis=1, inplace=True)
        lineage.update({temp['rank'].iloc[0]:temp['tax_name'].iloc[0]})

        ids.update({temp['rank'].iloc[0]: temp.tax_id.iloc[0]})



    return lineage, ids


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except ConnectionError as e:
        print(e)

    return None

def get_parent(taxid, taxaDB):
    parent_rank=False
    parent_taxid=False
    if taxid=='2':
        return parent_taxid, parent_rank
    #create connection:
    conn = create_connection(taxaDB)
    ranks=np.array(['superkingdom','phylum','subphylum','class','subclass','order','suborder', 'family','genus','species', 'subspecies'])
    try:
        lineage, ids = get_lineage_ids(str(taxid), conn)
    except:
        data_t=get_tax_data(str(taxid))
        lineage, ids = get_lineage_ids_fromdata(data_t, ranks) 

    if type(ids)==dict:
        allparents = ids
    elif type(temp[0])==dict:
        allparents = ids[0]
    else:
        print("Caught exception")
        sys.exit(1)

    rankorder=ranks[::-1]
    #handling when the taxid is itself among parent ids
    for r,i in allparents.items():
        if i==taxid: #and r != "superkingdom":
            taxid_rank=r
            ind_=np.where(rankorder==taxid_rank)[0][0]
            parent_rank=rankorder[ind_+1]


    for i, r in enumerate(rankorder):
        #handling when the taxid is itself among parent ids
        if parent_rank and taxid_rank==r:
            continue

        try:
            parent_taxid=allparents[r]
            parent_rank=r
            break
        except:
            pass
    return parent_taxid, parent_rank

def isSVInsample(svid, sample_id, table):
    """ table has indexes as the unique taxa and rows as samples
        if taxid exist in sample and has abundance>0 return True otherwise return False"""
    assert(table.index.name=="sv_id"),"the table has to have sv_id as its index and the index should be named sv_id"
    if np.any(table.index.isin([svid])):
        if table.loc[svid,sample_id]>0.0: #abundance is not zero
            return True
        else:
            return False
    else:
        return False

def istaxIDEqual(svid, sample_id, table1, table2):
    """this function implicitly assumes that svid exists in both tables for the sample_id and it checks if abundances is>0 in both 
       before comparing their tax_ids"""
    if table1.loc[svid, sample_id]>0 and table2.loc[svid, sample_id]>0:
        tax1 = table1.loc[svid, 'tax_id']
        tax2 = table2.loc[svid, 'tax_id']
    elif table1.loc[svid, sample_id]>0:
        tax1 = table1.loc[svid, 'tax_id']
        tax2=np.nan
    elif table2.loc[svid, sample_id]>0:
        tax1=np.nan
        tax2 = table2.loc[svid, 'tax_id']
    else:
        tax2=np.nan
        tax2=np.nan

    if tax1 == tax2:
        return True, tax1, tax2
    else:
        return False, tax1, tax2



def ranks_off(table1, table2, sv_id, sample_id, taxaDB):
    """A measure that calculates how many ranks is the SV in sample_id in table2 is off from 
       that in table1
       table1 and table2 are pandas dataframes with exactly the same indices and columns 
       columns as samples and indices as SVs"""
    ranks=['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    isInTable1=isSVInsample(sv_id, sample_id, table1)
    isInTable2=isSVInsample(sv_id, sample_id, table2)
    #SV has abundance in both
    if isInTable1 and isInTable2:
        isTaxaEq, tax1, tax2 = istaxIDEqual(sv_id, sample_id, table1, table2)
        if isTaxaEq: #SV in both and the corresponding tax_id is equal
            return 0
        else: #SV in both and the corresponding tax_ids differ 
            foundParent=False
            #try parents of tax1
            dummy_id = tax1
            output1=0
            while not foundParent and output1<7:
                print(dummy_id)
                parent_id, parent_rank =get_parent(dummy_id, taxaDB)
                if parent_id:
                    foundParent = parent_id==tax2   #istaxaInsample(parent_id, sample_id, table2)
                    dummy_id = parent_id
                    output1+=1
                else:
                    foundParent=True

            #try parents of tax1
            dummy_id = tax1
            output2=0
            while not foundParent and output2<7:
                print(dummy_id)
                parent_id, parent_rank =get_parent(dummy_id, taxaDB)
                if parent_id:
                    foundParent = parent_id==tax2   #istaxaInsample(parent_id, sample_id, table2)
                    dummy_id = parent_id
                    output2+=1
                else:
                    foundParent=True

            if output1>=output2:
                return output1
            else:
                return output2

    elif isInTable1: #not in table2
        isTaxaEq, tax1, tax2 = istaxIDEqual(sv_id, sample_id, table1, table2)
        parent_id, parent_rank =get_parent(tax1, taxaDB)
        if parent_rank in ranks:
            return ranks.index(parent_rank)+2 #plus 2 because the index starts from 0 and it is the parent
        elif parent_rank == "subspecies":
            return 7
        elif parent_rank == "subphylum":
            return 2
        


    elif isInTable2: #not in table1
        isTaxaEq, tax1, tax2 = istaxIDEqual(sv_id, sample_id, table1, table2)
        parent_id, parent_rank =get_parent(tax2, taxaDB)
        if parent_rank in ranks:
            return ranks.index(parent_rank)+2 #plus 2 because the index starts from 0 and it is the parent
        elif parent_rank == "subspecies":
            return 7
        elif parent_rank == "subphylum":
            return 2
    else: #not in both
        return 0



def get_ranksoff(allsvs, df_sv_list, df_sv_list_names, taxaDB):
    """ a measure that uses table1 taken as the mock and table2 taken as full, setA1, setB1, setC1
        one at a time. The taxa tables table1 and table2 would have the same tax_ids aligned 
        in the index and the same samples aligned in columns """
    mock=df_sv_list[0]
    df_merge=pd.DataFrame(index=list(allsvs))
    df_merge.index.name="sv_id"
    mock_tab1=df_merge.merge(mock,how='left', left_index=True, right_index=True)
    mock_tab1=mock_tab1.fillna(0)
    df_ranksoff_all=pd.DataFrame(index=mock_tab1.columns.values[1:])
    for i, df in enumerate(df_sv_list[1:],1):
        ana_tab2=df_merge.merge(df, how="left", left_index=True, right_index=True)
        ana_tab2=ana_tab2.fillna(0)
        dummy_df=pd.DataFrame(index=mock.index,columns=mock.columns.values)
        for sample_id in mock.columns.values[1:]: #exclude tax_id column
            for sv_id in mock.index.values:
                ranksOff =  ranks_off(mock_tab1, ana_tab2, sv_id, sample_id, taxaDB)
                RAM = mock.loc[sv_id, sample_id]/mock.loc[:, sample_id].sum() #mock relative abundance for taxa in this sample
                dummy_df.loc[sv_id,sample_id]=RAM*ranksOff #ranksOff times relative abundance in mock

            df_ranksoff_all.loc[sample_id,df_sv_list_names[i]]=dummy_df.loc[:,sample_id].sum()
            #df_ranksoff_all.to_csv("ranksOff_bySample.csv")
    fig, ax = pl.subplots(figsize=(12,12))
    sbs.violinplot(data=df_ranksoff_all, ax=ax)
    ax.set_ylabel("Ranks off")
    #fig.savefig("ranksOff_dist_comp.png")

    return df_ranksoff_all

df_ranksoff_all = get_ranksoff(allsvs, df_sv_list, df_sv_list_names, taxaDB)
#df_ranksoff_all.to_csv("ranksOff_bySample.csv")
#df_bc_all.to_csv("brayC_bySample.csv")
#df_rms_all.to_csv("rms_bySample.csv")

##"""""" here we explore how the accuracy of assignments vary with pplacer stats """"###### 
#accuracy measures: df_bc_all, df_rms_all, df_ranksoff_all <--y
#first align samples
samples=df_ranksoff_all.index.values
#accuracy: df_rankoff_all, df_rms_all
df_pcorrect = df_pcorrect.loc[samples,:]
df_rooted_qd_0 = df_rooted_qd_0.loc[samples,:]
df_rooted_qd_1 = df_rooted_qd_1.loc[samples,:]
df_rooted_qd_2 = df_rooted_qd_2.loc[samples,:]
df_phylo_entropy = df_phylo_entropy.loc[samples,:]
def plotViolin(df, ylbl, savepngF):
    fig, ax = pl.subplots(figsize=(12,12))
    sbs.violinplot(data=df, ax=ax)
    ax.set_ylabel(ylbl)
    #fig.savefig(savepngF)

plotViolin(df_pcorrect, "percentage correct", "violinpl_pcorrect_byRefPkg.png")
plotViolin(df_rooted_qd_0, "0D(T)", "violinpl_rooted_qd_0_byRefPkg.png")
plotViolin(df_rooted_qd_2, "2D(T)", "violinpl_rooted_qd_2_byRefPkg.png")
plotViolin(df_phylo_entropy, "phylogenetic entropy", "violinpl_phylo_entropy_byRefPkg.png")



#pplacer stats
df_sample_adcl = df_sample_adcl.loc[samples,:]
df_sample_edpl = df_sample_edpl.loc[samples,:]
df_sample_prichness = df_sample_prichness.loc[samples,:]
df_sample_mindistL = df_sample_mindistL.loc[samples,:]
def plot_scatter(df_x,df_y, xlabel, ylabel, df_sv_list_names):
    summary_stat=['mean','std','min','2.5%','25%','50%','75%','97.5%','max','WAvr']
    if xlabel=="adcl":
        summary_stat=['2.5%','50%', 'WAvr']
#    if xlabel=="prichness": #prichness min=q2.5%=q25%=1
#        summary_stat=['50%','75%','97.5%','max','WAvr']

    packages=df_sv_list_names[1:]
    for p in packages:
        y=df_y[p]
        for s in summary_stat:
            x=df_x[(p,s)]
            df_temp=pd.DataFrame({p+" "+s+" "+xlabel:x, p+" "+ylabel:y})
            df_temp = df_temp.astype(float)
            #fig, ax = pl.subplots(figsize=(12,12))
            #sbs.jointplot(x=x, y=y, kind='reg', stat_func=pr2, ax=ax)
            #fig.suptitle("accuracy in terms of "+ytitle+" for package '"+p+"' versus "+s+" of "+xlabel)
            #create a pair grid
            #grid = sbs.PairGrid(data=df_temp, vars=df_temp.columns, size=1)
            grid = sbs.PairGrid(data=df_temp, vars=df_temp.columns, x_vars=p+" "+s+" "+xlabel, y_vars=p+" "+ylabel, size=1)
            #map a joint plot to lower triangle
            grid = grid.map_lower(sbs.jointplot, kind="reg", stat_func=pr2)
