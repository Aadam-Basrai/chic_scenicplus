'''A script to create a search space for scenicplus using PCHIC data linking genes to regions, and 
overlapping this with ATAC consensus regions from scenicplus class, and the optionality to also include
scenicplus' normal search space function for a set distance'''

import pandas as pd 
import numpy as np
import ray 
import logging 
import os 
import pyranges as pr
import sys
from bx.intervals.intersection import Interval, IntervalTree
from ast import literal_eval
from tqdm import tqdm 

from .utils import extend_pyranges, extend_pyranges_with_limits, reduce_pyranges_with_limits_b
from .utils import calculate_distance_with_limits_join, reduce_pyranges_b, calculate_distance_join
from .utils import coord_to_region_names, region_names_to_coordinates, ASM_SYNONYMS, Groupby, flatten_list
from .scenicplus_class import SCENICPLUS



def get_pchic_search_space(SCENICPLUS_obj: SCENICPLUS,
                            species = None,
                            assembly = None,
                            biomart_host = 'http://www.ensembl.org',
                            inplace = True,
                            key_added = 'search_space',
                            pseudo_cell = False):
    """Get pchic_search_space:
    
    Parameters
    ----------
    SCENICPLUS_obj: SCENICPLUS
        a :class:`pr.SCENICPLUS`.
    pchic_countdata:
        a :class: `pd.DataFrame`
        dataframe must contain columns with the names: 'oeChr','oeStart','oeEnd','baitChr','baitStart','baitEnd','baitName'.
    inplace: bool, optional
        If set to True, store results into scplus_obj, otherwise return results.
    key_added: str, optional
        Key under which to add the results under scplus.uns.

    Return
    ------
    pd.DataFrame
        A data frame containing regions in the search space for each gene
    """

    #Create logger 
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('pchic_search_space')

    # GET GENE ANNOTATION AND CHROMSIZES
    if species is not None and assembly is not None:
        # Download gene annotation and chromsizes
        # 1. Download gene annotation from biomart
        import pybiomart as pbm
        dataset_name = '{}_gene_ensembl'.format(species)
        server = pbm.Server(host=biomart_host, use_cache=False)
        mart = server['ENSEMBL_MART_ENSEMBL']
        # check if biomart host is correct
        dataset_display_name = getattr(
            mart.datasets[dataset_name], 'display_name')
        if not (ASM_SYNONYMS[assembly] in dataset_display_name or assembly in dataset_display_name):
            print(
                f'\u001b[31m!! The provided assembly {assembly} does not match the biomart host ({dataset_display_name}).\n Please check biomart host parameter\u001b[0m\nFor more info see: https://m.ensembl.org/info/website/archives/assembly.html')
        # check wether dataset can be accessed.
        if dataset_name not in mart.list_datasets()['name'].to_numpy():
            raise Exception(
                '{} could not be found as a dataset in biomart. Check species name or consider manually providing gene annotations!')
        else:
            log.info(
                "Downloading gene annotation from biomart dataset: {}".format(dataset_name))
            dataset = mart[dataset_name]
            if 'external_gene_name' not in dataset.attributes.keys():
                external_gene_name_query = 'hgnc_symbol'
            else:
                external_gene_name_query = 'external_gene_name'
            if 'transcription_start_site' not in dataset.attributes.keys():
                transcription_start_site_query = 'transcript_start'
            else:
                transcription_start_site_query = 'transcription_start_site'
            annot = dataset.query(attributes=['chromosome_name', 'start_position', 'end_position',
                                  'strand', external_gene_name_query, transcription_start_site_query, 'transcript_biotype'])
            annot.columns = ['Chromosome', 'Start', 'End', 'Strand',
                             'Gene', 'Transcription_Start_Site', 'Transcript_type']
            annot['Chromosome'] = 'chr' + annot['Chromosome'].astype(str)
            annot = annot[annot.Transcript_type == 'protein_coding']
            if not any(['chr' in c for c in SCENICPLUS_obj.region_names]):
                annot.Chromosome = annot.Chromosome.str.replace('chr', '')
            
    #Loading Region names and formatting
    region_names = SCENICPLUS_obj.region_names
    region_names_df = pd.DataFrame(region_names, columns = ['region_name'])
    region_names_df[['chromosome','region']] = region_names_df['region_name'].str.split(':', n = 1, expand = True)

    region_names_df[['start','end']] = region_names_df['region'].str.split('-', n=1, expand=True)
    region_names_df['start']=region_names_df['start'].astype(int)
    region_names_df['end']=region_names_df['end'].astype(int)
    region_names_df=region_names_df[region_names_df['chromosome'].str.contains('chr')]
    region_names_df['chromosome']=region_names_df['chromosome'].str.strip('chr')

    #reordering region_names_df

    region_names_df.insert(0,'start',region_names_df.pop('start'))
    region_names_df.insert(1,'end',region_names_df.pop('end'))

    log.info('Cleaning PCHIC data')
    if species == 'hsapiens':
        chromosomes = ['1' ,'2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12','13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
    pchic_countdata = SCENICPLUS_obj.countdata_pchic
    pchic_cleaned = pchic_countdata.dropna(how = 'all') #change to dropping na on only a subset? 
    pchic_cleaned['chromosome']=pchic_cleaned['oeChr']
    if any(['chr' in c for c in pchic_cleaned['chromosome']]):
                pchic_cleaned['chromosome'] = pchic_cleaned['chromosome'].str.replace('chr', '')
    pchic_cleaned = pchic_cleaned[pchic_cleaned['chromosome'].isin(chromosomes)]
    
    pchic_cleaned['oeStart']=pchic_cleaned['oeStart'].astype(int)
    pchic_cleaned['oeEnd']=pchic_cleaned['oeEnd'].astype(int)

    #changing order or columns so oeStart and oeEnd are placed at the beginning: 
    pchic_cleaned.insert(0, 'oeStart',pchic_cleaned.pop('oeStart'))
    pchic_cleaned.insert(1,'oeEnd',pchic_cleaned.pop('oeEnd'))

    #matching pchic oeStart and oeEnd data to region_names created during consensus peak calling in earlier steps: 

    pchic_cleaned['region_name']=0 #will be used to fill in the region_name matched with scenic+ consensus peaks 
    if pseudo_cell == False:
        pchic_cleaned=pchic_cleaned[['oeStart','oeEnd','chromosome','region_name','baitName']]

    log.info('intersecting pchic data with region names')
    intersector = IntervalTree() #creates an empty dataset that will be filled by region_names for each chromosome in the for loops:
    for i in range(1,23): 
        intersector = IntervalTree()
        for j in region_names_df.index: 
            if str(region_names_df['chromosome'][j]) == str(i):
                intersector.insert(region_names_df['start'][j],region_names_df['end'][j],region_names_df['region_name'][j])
        for k in pchic_cleaned.index: 
            if str(pchic_cleaned['chromosome'][k])== str(i):
                pchic_cleaned['region_name'][k]= str(intersector.find(pchic_cleaned['oeStart'][k],pchic_cleaned['oeEnd'][k]))

    intersector = IntervalTree()
    for j in region_names_df.index:
        if str(region_names_df['chromosome'][j])== str('X'):
            intersector.insert(region_names_df['start'][j],region_names_df['end'][j],region_names_df['region_name'][j])
    for k in pchic_cleaned.index: 
        if str(pchic_cleaned['chromosome'][k])== str('X'):
                pchic_cleaned['region_name'][k]= str(intersector.find(pchic_cleaned['oeStart'][k],pchic_cleaned['oeEnd'][k]))

    intersector = IntervalTree()
    for j in region_names_df.index:
        if str(region_names_df['chromosome'][j])== str('Y'):
            intersector.insert(region_names_df['start'][j],region_names_df['end'][j],region_names_df['region_name'][j])
    for k in pchic_cleaned.index: 
            if str(pchic_cleaned['chromosome'][k])== str('Y'):
                pchic_cleaned['region_name'][k]= str(intersector.find(pchic_cleaned['oeStart'][k],pchic_cleaned['oeEnd'][k]))

    pchic_search_space = pchic_cleaned
    pchic_search_space.dropna(subset='region_name',inplace = True)
    if pseudo_cell == False:
        log.info('Exploding region names so each region name has its own row')
        pchic_search_space['region_name'] = pchic_search_space['region_name'].apply(literal_eval)
        pchic_search_space = pchic_cleaned.explode('region_name') 
    else:
        log.info('creating region_names from fragments')
        pchic_search_space.dropna(subset='region_name',inplace=True) #dropping any pchic contacts that don't overlap with any ATAC consensus peaks
        pchic_search_space = pchic_search_space.drop_duplicates(subset=['region_name','baitName']) 
        pchic_search_space['region_name'] = 'chr'+pchic_search_space['chromosome']+':'+pchic_search_space['oeStart'].astype(str)+'-'+pchic_search_space['oeEnd'].astype(str) + ';' + pchic_search_space['baitName']
        
    
    log.info('Calculating distances between regions and gene')
    annot_TSS = annot[['Gene','Transcription_Start_Site']]
    pchic_search_space=pchic_search_space.merge(annot_TSS,left_on = 'baitName',right_on = 'Gene',how = 'inner')
    pchic_search_space['avg_region'] = abs((pchic_search_space['oeStart']+pchic_search_space['oeEnd'])/2)
    pchic_search_space['Distance'] = abs(pchic_search_space['Transcription_Start_Site']-abs((pchic_search_space['avg_region'])))

    if 'Gene' not in pchic_search_space.columns:
        pchic_search_space = pchic_search_space.rename(columns = {'baitName':'Gene'})
    if 'Name' not in pchic_search_space.columns:
        pchic_search_space = pchic_search_space.rename(columns = {'region_name':'Name'})
    if pseudo_cell == False:
        pchic_search_space = pchic_search_space[['Name','Gene','Distance']]
    pchic_search_space.dropna(subset='Name',inplace=True)
    pchic_search_space = pchic_search_space.drop_duplicates(subset=['Name','Gene']) 
    pchic_search_space = pchic_search_space.reset_index()
    if inplace:
        SCENICPLUS_obj.uns[key_added] = pchic_search_space[['Name','Gene','Distance']]
    else:
        return pchic_search_space
    log.info('Done!')

def pchic_celltype_df(countdata, 
                     cell_name_dictionary):
    '''
    helper function to create pchic count dataframes per cell type'''
    peak_matrix = countdata

    pchic_celltype_dict={}
    for cellname in cell_name_dictionary.keys():
        columns = ('chromosome','oeStart','oeEnd','baitChr','baitStart','baitEnd','oeName','width','Name','Gene',*cell_name_dictionary[cellname])
        average = peak_matrix.loc[:,(*cell_name_dictionary[cellname],)]
        average['average'] = average.mean(axis=1)
        dataframe = peak_matrix.loc[:,columns]
        dataframe['average'] = average['average']
        dataframe = dataframe.reset_index()
        pchic_celltype_dict[cellname] = dataframe 
    return pchic_celltype_dict

def get_cell_labels(SCENICPLUS_obj,
                    cell_mapping):
    '''
    helper function to get celllabels for celltypes of interest'''

    metadata_cell = SCENICPLUS_obj.metadata_cell
    celllabels = {}
    for cellname in cell_mapping.keys():
        list = metadata_cell[metadata_cell['GEX_celltype'].isin(cell_mapping[cellname])].index
        celllabels[cellname]=list

    return celllabels

@ray.remote
def pseudo_single_cell_ray(cellname,countdata,cell_label,frac) -> list:
    return(pseudo_single_cell(cellname, countdata,cell_label,frac))

def pseudo_single_cell(cellname,
                        countdata,
                        cell_label,
                        frac=0.5) -> list:
    '''
    helper function to calculate a pseudo single cell matrix for a single cell type, given its bulk PcHiC.
    Carries out search space for each 
    '''
    random_state = 666
    pchic = countdata[['Name','average']]
    pchic.set_index(keys = 'Name',inplace = True)

    scPCHIC = []

    for i in range(0,len(cell_label)):
        random_state = random_state+1
        df_to_merge = pchic.T.sample(frac=frac,axis=1,random_state = random_state)
        df_to_merge['cell_label'] = cell_label[i]
        scPCHIC.append(df_to_merge)
    scPCHIC = pd.concat(scPCHIC,axis=0)
    scPCHIC.set_index(keys='cell_label',inplace = True)

    return scPCHIC,cellname

def calculate_scPCHIC(SCENICPLUS_obj: SCENICPLUS,
                            species,
                            assembly,
                            cell_name_dictionary,
                            cell_mapping,
                            biomart_host = 'http://www.ensembl.org',
                            n_cpu = 1,
                            frac = 0.5,
                            inplace = False):
    '''
    Parameters
    ----------
    SCENICPLUS_obj
    species
    assembly
    cell_name_dictionary 
        a :class: `Dict`
        A dictionary with the keys being cellnames, and the values being the column names for replicates
    cell_mapping
        a :class: `Dict` 
        A dictionary with the keys being cellnames matching the cellnames in cell_name_dictionary and the values being the names of cellname(s) in the ATAC-seq and RNA-seq data as a list like object. 
    '''

    #Create logger 
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('calculate scPCHIC')
    if cell_name_dictionary.keys() != cell_mapping.keys():
        raise Exception(
            'Please ensure cell_name_dictionary and cell_mapping have the same keys'
        )

    log.info('Getting countdata in search space format')
    countdata = get_pchic_search_space(SCENICPLUS_obj,
                                        species = species,
                                        assembly = assembly,
                                        biomart_host = biomart_host,
                                        pseudo_cell= True,
                                        inplace=False)
    SCENICPLUS_obj.uns['pchic_search_space'] = countdata[['Name','Gene','Distance']]

    log.info('creating celltype specific countdata')
    cell_countdata = pchic_celltype_df(countdata,
                                        cell_name_dictionary)
    
    log.info('getting cell_labels')
    cell_labels = get_cell_labels(SCENICPLUS_obj,
                                    cell_mapping)
    
    log.info('creating pseudo single cells for each celltype')

    ray.init(num_cpus=n_cpu)
    try:
        jobs = []

        for cellname in cell_countdata.keys():
            data = cell_countdata[cellname]
            labels = cell_labels[cellname]
            jobs.append(pseudo_single_cell_ray.remote(cellname,data,labels,frac))
        
        def to_iterator(obj_ids):
            while obj_ids:
                finished_ids,obj_ids = ray.wait(obj_ids)
                for finished_id in finished_ids:
                    yield ray.get(finished_id)
        scPCHIC_dict = {}
        for scPCHIC,cellname in tqdm(to_iterator(jobs),
                                    total=len(jobs),
                                    desc=f'Running using {n_cpu} cores',
                                    smoothing=0.1):
            scPCHIC_dict[cellname] = scPCHIC   
    except Exception as e:
        print(e)
    finally:
        ray.shutdown() 
    
    result_df = pd.concat(scPCHIC_dict[cellname]for cellname in scPCHIC_dict.keys())

    if inplace:
        if not len(result_df) == len(SCENICPLUS_obj.X_ACC.T):
            raise ValueError(
                'Please ensure that you use all celltypes when creating scPCHIC dataframe inplace'
            )
        else:
            SCENICPLUS_obj.X_PCHIC = result_df.T
    else:
        return result_df
