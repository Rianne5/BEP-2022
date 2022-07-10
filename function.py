import pandas as pd
import os
import arxiv

import urllib
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import tensorflow_hub as hub
import numpy as np 

import requests 
import time
import re


def search(Sentence):
    """
    Start functions for joining Anth and Arxiv
    Using clean title from Anthology (cleaned version)
    Search using Arxiv api
    If found, return Arxiv paper    
    """
    print('-- start sentence:', Sentence)
    que = pd.DataFrame(columns = ['title_Arxiv', 'pdf_url_Arxiv', 'published_Arxiv', 'result_Arxiv'])
    search = arxiv.Search(
        query = "%22"+str(Sentence)+"%22",
        max_results = 10,
        sort_by = arxiv.SortCriterion.SubmittedDate)    
    for result in search.results():
        
        que.loc[len(que.index)] = [result.title, result.pdf_url, result.published, result]
    return que


def run(Anth):
    """Run, for combining tables Anth and papers Arxiv"""
    i = 0
    strike = 0
#     t = 0
    for index, series in Anth.iterrows():
        #to make sure Arxiv does not ban me
        if i > 10000:
            return Anth
        #if paper is not found in ArXiv yet
        if series[['title_Arxiv']].isna().values[0] == True:
            try: 
                print('iteration: ',i)
                i+=1
                # search for the paper
                q = search(series['clean'])
                
                # set row to Not Found as placeholder. Used to makesure we do not run line twice
                Anth.loc[index,'title_Arxiv'] = 'NF'
                # for every found paper
                for q_i, q_s in q.iterrows():

                    strike =0 #may be wrong to do this
                    #check the location
                    location = Anth.loc[Anth['title']==q_s['title_Arxiv']]
                    if len(location) > 1:
                        print('Weird, multiple papers with the same name:',location)
                        next    
                    i_loc = location.index.values.astype(int)[0]
                    l = q_s.tolist()
                    Anth.loc[[i_loc],['title_Arxiv', 'pdf_url_Arxiv', 'published_Arxiv', 'result_Arxiv']] = l
                    strike =0

            #In case of error, Mostlikely caused by time out arxiv api
            except: 
                print('Might be error in: Line with iloc: IndexError: index 0 is out of bounds for axis 0 with size 0')                    
                strike +=1
                if strike ==3:
                    print('Assumption: Arxiv stopped')
                    return Anth
        
        # Arxiv_title is not NA. This means is is NF (not found) or found and title is given here.
        else: 
            next
            
    return Anth



def clean_title(clean):
    """
    Clean title from illigal instances like ?!'
    So it can be used to save to pdf
    """
    clean['title'] = clean.apply(lambda x: x['title'].replace('-', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('_', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace("\\", ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace("/", ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('?', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('!', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('+', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('#', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('%', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('{', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('}', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('*', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('$', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace(':', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('|', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('=', ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace("'", ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace("''", ' '),axis =1)
    clean['title'] = clean.apply(lambda x: x['title'].replace('"', ' '),axis =1)
    return clean

def clean_title2(clean):
    """
    Clean title from illigal instances like ?!'
    So it can be used to save to pdf
    Function to use apply
    """
    clean = clean.replace('-', ' ')
    clean = clean.replace('_', ' ')
    clean = clean.replace("\\", ' ')
    clean = clean.replace("/", ' ')
    clean = clean.replace('?', ' ')
    clean = clean.replace('!', ' ')
    clean = clean.replace('+', ' ')
    clean = clean.replace('#', ' ')
    clean = clean.replace('%', ' ')
    clean = clean.replace('{', ' ')
    clean = clean.replace('}', ' ')
    clean = clean.replace('*', ' ')
    clean = clean.replace('$', ' ')
    clean = clean.replace(':', ' ')
    clean = clean.replace('|', ' ')
    clean = clean.replace('=', ' ')
    clean = clean.replace("'", ' ')
    clean = clean.replace('"', ' ')
    return clean

def add_col(Anth):
    Anth['title_Arxiv'] = np.nan
    Anth['pdf_url_Arxiv'] = np.nan
    Anth['published_Arxiv'] = np.nan
    Anth['result_Arxiv'] = np.nan
    return Anth




def only_first(s):
    """
    Start functions for pdf to text
    Get only first version instead of another from arxiv.
    use apply function on the column to be changed."""
    return s.strip('12234567890')+'1'

def prepare_table(tab):
    if len(tab.columns)>18:
        tab = tab.dropna(axis = 0, subset=['pdf_url_Arxiv'])
        tab = tab[['title','year', 'month','url', 'title_Arxiv', 'pdf_url_Arxiv', 'published_Arxiv','result_Arxiv']]
        tab['pdf_url_Arxiv'] = tab['pdf_url_Arxiv'].apply(only_first)
        
    return tab


def clean_n(clean):
    """
    input df column
    Clean title from illigal instances like ?!'
    So it can be used to save to pdf
    Function to use apply
    """
    clean = clean.replace('-/n', '')
    clean = clean.replace('/n', ' ')
    clean = clean.replace('-\n', '')
    clean = clean.replace('\n', ' ')
    return clean

def pdf_similarity(embed, tab_old, result=None, max_iter=1, r_state = 2):
    """
    Wanted statistics?: number of words, similarity, number of pages, percentage of words exactly the same,
    in = table
    """ 
    # for each random row in table, replace=False all rows only sampled once
    sample = tab_old.sample(n=max_iter, replace=False, random_state=r_state)
    not_sample = tab_old.drop(sample.index)
    
    #download two papers from table
    loc_name_Arxiv = 'pdf sim/Arxiv.pdf'
    loc_name_Anth = 'pdf sim/Anth.pdf'
    
    #needs to be changed
    sample[['Succes', 'w_Anth', 'w_Arxiv', 'pages_Anth', 'pages_Arxiv', 'cosine', 'ref_Anth', 'ref_Arxiv', 'Jaccard' ]] =np.nan
    
    
    for index,series in sample.iterrows():
        sample.loc[[index],['Succes']]=0

#         if True:
        #open paper in jupiter/blocks
        try: #sometimes download not available. Then download works but does not give a good pdf so reading does not work
            urllib.request.urlretrieve(series['pdf_url_Arxiv'], loc_name_Arxiv)
            urllib.request.urlretrieve(series['url']+'.pdf', loc_name_Anth)
            print('start with two papers')
            doc_Arxiv = fitz.open(loc_name_Arxiv)
            doc_Anth = fitz.open(loc_name_Anth)

            b_Anth = get_blocks(doc_Anth)
            b_Arxiv = get_blocks(doc_Arxiv)
            
            #try removing sidebar from arxiv
            # if we DO find something
            if [i for i, x in enumerate(b_Arxiv) if x.find('arXiv:')>= 0] !=[]:
                i_sidebar = [i for i, x in enumerate(b_Arxiv) if x.find('arXiv:')>= 0][0]
                if len(b_Arxiv[i_sidebar])< 50: #dont remove references by accident, normal sidebar is length 41
                    x = b_Arxiv.pop(i_sidebar)
#                     print(x)
            
            #remove footnote from anthology (could/did not remove pagenumbers)
            if b_Anth[0][0:18] == 'Proceedings of the' and len(b_Anth[0])<150:
                b_Anth=b_Anth[1:]

#             elem_both_list = set()
            elem_both_list = set(b_Anth)&set(b_Arxiv)
            #sometimes same occurs twice in paper then remove this from elem_both_list
            dup = [x for x in elem_both_list if b_Anth.count(x)>1]
            dup = set(dup + [x for x in elem_both_list if b_Anth.count(x)>1])
            for el in dup:
                elem_both_list.remove(el)
                
        
            #number of citations
            if [i for i, x in enumerate(b_Anth) if x.find('References')>= 0] ==[]:
                ref_Anth='NF'
                ref_Arxiv='NF'
            elif [i for i, x in enumerate(b_Anth) if x.find('References')>= 0] ==[]:
                ref_Anth='NF'
                ref_Arxiv='NF'
            else:              
                start_ref_Anth = [i for i, x in enumerate(b_Anth) if x.find('References')>= 0][0]
                start_ref_Arxiv = [i for i, x in enumerate(b_Arxiv) if x.find('References')>= 0][0]
                
  
                amount_pattern = r'(?:[1][89][0-9]{2}[^0-9]|[2][0][012][0-9][^0-9])'
                amount_expr = re.compile(amount_pattern, re.IGNORECASE)
                l = []
                for i in b_Anth[start_ref_Anth:]:
                    l+=(amount_expr.findall(i))
                ref_Anth=len(l)
                l = []
                for i in b_Arxiv[start_ref_Anth:]:
                    l+=(amount_expr.findall(i))
                ref_Arxiv=len(l)
                
    
            #similarity
            cosine_avg, w_Anth, w_Arxiv, jacc = similarity(b_Anth,b_Arxiv,embedding=embed)
            
            # add statistics to table 'sample'
            sample.loc[[index],['cosine_avg', 'w_Anth', 'w_Arxiv', 'ref_Anth', 'ref_Arxiv', 'Jaccard']]= cosine_avg, w_Anth, w_Arxiv, ref_Anth, ref_Arxiv, jacc
            sample.loc[[index],['pages_Anth', 'pages_Arxiv']] = [doc_Anth.page_count, doc_Arxiv.page_count]
            
            sample.loc[[index],['Succes']]=1
            

        except: #error with downloading 
            print('Exception found some error')
            next 
              
        time.sleep(3)
        
    #end loop
    # join tables
    if type(result) ==pd.DataFrame:
        sample = sample.merge(result,how='outer')
    return not_sample, sample


def similarity(anth, arxiv, embedding):
#     print('start sim')
    anth = [clean_n(x) for x in anth]
    arxiv = [clean_n(x) for x in arxiv]
    
    w_anth = " ".join(anth).split()
    w_arxiv = " ".join(arxiv).split()
    
    set_Anth = set(w_anth)
    set_Arxiv = set(w_arxiv)
    
    # jacc = Intersection/Union
    jacc = len(set_Anth & set_Arxiv)/len(set_Anth | set_Arxiv)
    
    df = pd.DataFrame()
    df['Anth'] = [" ".join(list(x)) for x in np.array_split(np.array(w_anth),20)]
    df['Arxiv'] = [" ".join(list(x)) for x in np.array_split(np.array(w_arxiv),20)]
    
    df['words Anth'] = df['Anth'].apply(lambda x: len(x.split()))
    df['words Arxiv'] = df['Arxiv'].apply(lambda x: len(x.split()))
    
    df['cosine'] = df.apply(lambda x:cosine_similarity(embedding([x.Anth]),embedding([x.Arxiv])), axis=1)
    df['cosine'] = df['cosine'].astype(float)

#     display(df)
#     df.tolatex()

    #nr of words per paper
    nr_w_Anth = sum(df['words Anth'])
    nr_w_Arxiv = sum(df['words Arxiv'])

    cosine_avg = round(np.average(df['cosine']),6)
    
    return cosine_avg, nr_w_Anth, nr_w_Arxiv, jacc 

    
def get_blocks(doc):
    blocks = []
    for page in doc.pages():
        b_p = page.get_text('blocks')
        for b in b_p:
            if b[6] == 0: #only text not metadata pictures.
                blocks.append(b[4])
    return blocks