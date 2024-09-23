
import os

from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Union

import time
from pathlib import Path
import pandas as pd
import numpy as np
import collections
from nltk.corpus import wordnet as wn
import csv
import datetime
import logging
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', message='TypedStorage is deprecated.')


from sklearn.metrics import confusion_matrix, precision_score, average_precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings_transformer(sentences, tokenizer_in, model_in):

    
    # Tokenize sentences
    encoded_input = tokenizer_in(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model_in(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


    return sentence_embeddings



def CalculateEmbeddingsVocabs(DictList_syn_Virus_SEED, DictList_syn_Country_SEED, model_biobert, tokenizer_transformers_bio,
                                                    model_transformers_bio, model_All, EXPAND_DICT):
    
    DictList_syn_Embeddings_Virus_SEED = []
    for ix in range(len(DictList_syn_Virus_SEED)):

        row = DictList_syn_Virus_SEED[ix].copy()
        dynamicrow = row.copy()
        
        if EXPAND_DICT:
            for idx in range(len(row)):

                all_y = []
                all_y.append(str(row[idx]).strip())
                if "_" in str(row[idx]):
                    all_y.append(str(row[idx]).replace("_", " ").replace("  ", " ").strip())
                if "-" in str(row[idx]):
                    all_y.append(str(row[idx]).replace("-", " ").replace("  ", " ").strip())
                if "." in str(row[idx]):
                    all_y.append(str(row[idx]).replace(".", " ").replace("  ", " ").strip())
                if "," in str(row[idx]):
                    all_y.append(str(row[idx]).replace(",", " ").replace("  ", " ").strip())
                if ";" in str(row[idx]):
                    all_y.append(str(row[idx]).replace(";", " ").replace("  ", " ").strip())

                y_syn = wn.synonyms(str(row[idx]).strip()) + wn.synonyms(str(row[idx]).lower().strip())
                if len(y_syn) > 0:
                    newlist = []
                    for x in y_syn:
                        if len(newlist) > 0:
                            newlist = newlist + x
                        else:
                            newlist = x
                    y_syn = newlist
                    y_syn = list(dict.fromkeys(y_syn))

                for y in y_syn:
                    all_y.append(str(y).strip())
                    if "_" in y:
                        all_y.append(str(y).replace("_", " ").replace("  ", " ").strip())
                    if "-" in y:
                        all_y.append(str(y).replace("-", " ").replace("  ", " ").strip())
                    if "." in y:
                        all_y.append(str(y).replace(".", " ").replace("  ", " ").strip())
                    if ";" in y:
                        all_y.append(str(y).replace(";", " ").replace("  ", " ").strip())

                dynamicrow = dynamicrow + all_y
                dynamicrow = list(dict.fromkeys(dynamicrow))  # duplicates

            DictList_syn_Virus_SEED[ix] = dynamicrow

        
        if model_biobert:
            dynamicrow_list_Embeddings = np.array([np.mean(model_biobert.encode(dynamicrow), axis=0)])
        else:
            dynamicrow_list_Embeddings = np.array(
                [np.mean(get_embeddings_transformer(dynamicrow, tokenizer_transformers_bio,
                                                    model_transformers_bio).numpy(), axis=0)])

        if len(DictList_syn_Embeddings_Virus_SEED) > 0:
            DictList_syn_Embeddings_Virus_SEED = np.append(DictList_syn_Embeddings_Virus_SEED,
                                                           dynamicrow_list_Embeddings, axis=0)  # addedencode
        else:
            DictList_syn_Embeddings_Virus_SEED = dynamicrow_list_Embeddings.copy()

    DictList_syn_Embeddings_Country_SEED = []
    for ix in range(len(DictList_syn_Country_SEED)):

        row = DictList_syn_Country_SEED[ix].copy()
        dynamicrow = row.copy()
        
        if EXPAND_DICT:
            for idx in range(len(row)):

                all_y = []
                all_y.append(str(row[idx]).strip())
                if "_" in str(row[idx]):
                    all_y.append(str(row[idx]).replace("_", " ").replace("  ", " ").strip())
                if "-" in str(row[idx]):
                    all_y.append(str(row[idx]).replace("-", " ").replace("  ", " ").strip())
                if "." in str(row[idx]):
                    all_y.append(str(row[idx]).replace(".", " ").replace("  ", " ").strip())
                if "," in str(row[idx]):
                    all_y.append(str(row[idx]).replace(",", " ").replace("  ", " ").strip())
                if ";" in str(row[idx]):
                    all_y.append(str(row[idx]).replace(";", " ").replace("  ", " ").strip())

                y_syn = wn.synonyms(str(row[idx]).strip()) + wn.synonyms(str(row[idx]).lower().strip())
                if len(y_syn) > 0:
                    newlist = []
                    for x in y_syn:
                        if len(newlist) > 0:
                            newlist = newlist + x
                        else:
                            newlist = x
                    y_syn = newlist
                    y_syn = list(dict.fromkeys(y_syn))

                for y in y_syn:
                    all_y.append(str(y).strip())
                    if "_" in y:
                        all_y.append(str(y).replace("_", " ").replace("  ", " ").strip())
                    if "-" in y:
                        all_y.append(str(y).replace("-", " ").replace("  ", " ").strip())
                    if "." in y:
                        all_y.append(str(y).replace(".", " ").replace("  ", " ").strip())
                    if ";" in y:
                        all_y.append(str(y).replace(";", " ").replace("  ", " ").strip())

                dynamicrow = dynamicrow + all_y
                dynamicrow = list(dict.fromkeys(dynamicrow))  # duplicates

            DictList_syn_Country_SEED[ix] = dynamicrow

        
        dynamicrow_list_Embeddings = np.array([np.mean(model_All.encode(dynamicrow), axis=0)])

        if len(DictList_syn_Embeddings_Country_SEED) > 0:
            DictList_syn_Embeddings_Country_SEED = np.append(DictList_syn_Embeddings_Country_SEED,
                                                             dynamicrow_list_Embeddings, axis=0)  # addedencode
        else:
            DictList_syn_Embeddings_Country_SEED = dynamicrow_list_Embeddings.copy()
       
            
    return DictList_syn_Virus_SEED, DictList_syn_Country_SEED, DictList_syn_Embeddings_Virus_SEED, DictList_syn_Embeddings_Country_SEED
            

def attachCentroidsToDict(DictOVERALL,DictList_syn_SEED):
    
    CENTROIDS = []
    for xj in range(len(DictOVERALL)):

        Syn_at_xj = DictOVERALL[xj].copy()

        if (len(Syn_at_xj) > 2) and (
                xj > len(DictList_syn_SEED)):  # I do this only for the new added terms from the dict seed
            centroxj = most_frequent(Syn_at_xj)
        else:
            centroxj = Syn_at_xj[0]

       
        CENTROIDS.append(centroxj)

        listenNoDuplicates = list(dict.fromkeys(Syn_at_xj))  # remove duplicates
        DictOVERALL[xj] = listenNoDuplicates

    df_synOVERALL = pd.DataFrame(DictOVERALL, index=CENTROIDS)
    
    return df_synOVERALL



def attachToDicts(df,DictList_syn_Virus_SEED,DictList_syn_Country_SEED,DictList_syn_Embeddings_Virus_SEED,DictList_syn_Embeddings_Country_SEED,
                  COSINE_THRESHOLD_VIRUS,COSINE_THRESHOLD_COUNTRY,
                model_biobert,tokenizer_transformers_bio,model_transformers_bio,model_All,):
    
    
    # CREATE STEP DICT SYN:
    DictList_syn_Virus = DictList_syn_Virus_SEED.copy()
    DictList_syn_Embeddings_Virus = DictList_syn_Embeddings_Virus_SEED.copy()  # model_All.encode(DictList_syn_Virus)

    DictList_syn_Country = DictList_syn_Country_SEED.copy()
    DictList_syn_Embeddings_Country = DictList_syn_Embeddings_Country_SEED.copy()  # model_All.encode(DictList_syn_Country)

    ##############################################

    for colname in df.columns:

        if ("geoname EpiTator" in colname) or ("country_extracted" in colname) or (
                "disease EpiTator" in colname) or ("virus_extracted" in colname):

            print("\n... " + colname)

            if ("disease EpiTator" in colname) or ("virus_extracted" in colname):
                IsBio = True
            else:
                IsBio = False

            y_pred = df[colname].values

            y_pred = y_pred[~pd.isnull(y_pred)]  # remove nan
            y_pred = list(dict.fromkeys(y_pred))  # remove duplicates

            for idx in tqdm(range(len(y_pred))):
                # print(idx)

                all_y_pred_list = []
                all_y_pred_list.append(str(y_pred[idx]).strip())
                if "_" in str(y_pred[idx]):
                    all_y_pred_list.append(str(y_pred[idx]).replace("_", " ").replace("  ", " ").strip())
                if "-" in str(y_pred[idx]):
                    all_y_pred_list.append(str(y_pred[idx]).replace("-", " ").replace("  ", " ").strip())
                if "." in str(y_pred[idx]):
                    all_y_pred_list.append(str(y_pred[idx]).replace(".", " ").replace("  ", " ").strip())
                if "," in str(y_pred[idx]):
                    all_y_pred_list.append(str(y_pred[idx]).replace(",", " ").replace("  ", " ").strip())
                if ";" in str(y_pred[idx]):
                    all_y_pred_list.append(str(y_pred[idx]).replace(";", " ").replace("  ", " ").strip())

                
                # wn synset
                y_pred_syn = wn.synonyms(str(y_pred[idx]).strip()) + wn.synonyms(str(y_pred[idx]).lower().strip())
                if len(y_pred_syn) > 0:
                    newlist = []
                    for x in y_pred_syn:
                        if len(newlist) > 0:
                            newlist = newlist + x
                        else:
                            newlist = x
                    y_pred_syn = newlist
                    y_pred_syn = list(dict.fromkeys(y_pred_syn))
                for y in y_pred_syn:
                    all_y_pred_list.append(str(y).strip())
                    if "_" in y:
                        all_y_pred_list.append(str(y).replace("_", " ").replace("  ", " ").strip())
                    if "-" in y:
                        all_y_pred_list.append(str(y).replace("-", " ").replace("  ", " ").strip())
                    if "." in y:
                        all_y_pred_list.append(str(y).replace(".", " ").replace("  ", " ").strip())
                    if ";" in y:
                        all_y_pred_list.append(str(y).replace(";", " ").replace("  ", " ").strip())

                all_y_pred_list = list(dict.fromkeys(all_y_pred_list))  # remove duplicates

                #########

                

                intersect = set()
                FoundAtIy = -1
                all_y_pred_list_SET = set([x.lower() for x in all_y_pred_list])
                if IsBio:
                    for iy in range(len(DictList_syn_Virus)):
                        b = DictList_syn_Virus[iy]
                        b_set = set([x.lower() for x in b])
                        intersect = all_y_pred_list_SET.intersection(b_set)
                        if len(intersect) > 0:
                            FoundAtIy = iy
                            break
                else:
                    for iy in range(len(DictList_syn_Country)):
                        b = DictList_syn_Country[iy]
                        b_set = set([x.lower() for x in b])
                        intersect = all_y_pred_list_SET.intersection(b_set)
                        if len(intersect) > 0:
                            FoundAtIy = iy
                            break

                if FoundAtIy > -1:
                    if IsBio:
                        listenlarged = DictList_syn_Virus[FoundAtIy] + all_y_pred_list  # list(intersect)
                        DictList_syn_Virus[FoundAtIy] = listenlarged
                                                                                        model_transformers_bio).numpy()
                        if model_biobert:
                            DictList_syn_Embeddings_Virus[FoundAtIy] = np.array(
                                [np.mean(model_biobert.encode(DictList_syn_Virus[FoundAtIy]), axis=0)])
                        else:
                            DictList_syn_Embeddings_Virus[FoundAtIy] = np.array([np.mean(
                                get_embeddings_transformer(DictList_syn_Virus[FoundAtIy],
                                                            tokenizer_transformers_bio,
                                                            model_transformers_bio).numpy(), axis=0)])

                    else:
                        listenlarged = DictList_syn_Country[FoundAtIy] + all_y_pred_list  # list(intersect)
                        DictList_syn_Country[FoundAtIy] = listenlarged
                        DictList_syn_Embeddings_Country[FoundAtIy] = np.array(
                            [np.mean(model_All.encode(DictList_syn_Country[FoundAtIy]), axis=0)])

                else:
                    if IsBio == True:
                                                                                        model_transformers_bio).numpy()
                        if model_biobert:
                            all_y_pred_list_Embeddings = np.array(
                                [np.mean(model_biobert.encode(all_y_pred_list), axis=0)])
                        else:
                            all_y_pred_list_Embeddings = np.array(
                                [np.mean(get_embeddings_transformer(all_y_pred_list,
                                                                    tokenizer_transformers_bio,
                                                                    model_transformers_bio).numpy(), axis=0)])

                        cosine_sim_ALL = \
                        cosine_similarity(all_y_pred_list_Embeddings, DictList_syn_Embeddings_Virus)[
                            0]  # addedencode
                        maxcosine = max(cosine_sim_ALL)
                        idmaxcosine = -1
                        if maxcosine > COSINE_THRESHOLD_VIRUS:
                            idmaxcosine = np.where(cosine_sim_ALL == maxcosine)[0][0]
                            listenlarged = DictList_syn_Virus[idmaxcosine] + all_y_pred_list  # list(intersect)
                            DictList_syn_Virus[idmaxcosine] = listenlarged

                                                                                            model_transformers_bio).numpy()
                            if model_biobert:
                                DictList_syn_Embeddings_Virus[idmaxcosine] = np.array(
                                    [np.mean(model_biobert.encode(DictList_syn_Virus[idmaxcosine]), axis=0)])
                            else:
                                DictList_syn_Embeddings_Virus[idmaxcosine] = np.array([np.mean(
                                    get_embeddings_transformer(DictList_syn_Virus[idmaxcosine],
                                                                tokenizer_transformers_bio,
                                                                model_transformers_bio).numpy(), axis=0)])

                        if idmaxcosine < 0:  # it didn't find it, I need to add a new dictionary group
                            DictList_syn_Virus.append(all_y_pred_list)
                            DictList_syn_Embeddings_Virus = np.append(DictList_syn_Embeddings_Virus,
                                                                        all_y_pred_list_Embeddings,
                                                                        axis=0)  # addedencode

                    else:
                        
						all_y_pred_list_Embeddings = np.array([np.mean(model_All.encode(all_y_pred_list), axis=0)])

                        cosine_sim_ALL = \
                        cosine_similarity(all_y_pred_list_Embeddings, DictList_syn_Embeddings_Country)[
                            0]  # addedencode
                        maxcosine = max(cosine_sim_ALL)
                        idmaxcosine = -1
                        if maxcosine > COSINE_THRESHOLD_COUNTRY:
                            idmaxcosine = np.where(cosine_sim_ALL == maxcosine)[0][0]
                            listenlarged = DictList_syn_Country[idmaxcosine] + all_y_pred_list  # list(intersect)
                            DictList_syn_Country[idmaxcosine] = listenlarged
                            DictList_syn_Embeddings_Country[idmaxcosine] = np.array(
                                [np.mean(model_All.encode(DictList_syn_Country[idmaxcosine]), axis=0)])

                        if idmaxcosine < 0:  # it didn't find it, I need to add a new dictionary group
                            DictList_syn_Country.append(all_y_pred_list)
                            DictList_syn_Embeddings_Country = np.append(DictList_syn_Embeddings_Country,
                                                                        all_y_pred_list_Embeddings,
                                                                        axis=0)  # addedencode

        else:
            continue
        
        
    return DictList_syn_Virus, DictList_syn_Country, DictList_syn_Embeddings_Virus, DictList_syn_Embeddings_Country

            
            
            
            


def run_dictionaryPopulation(LIST_FILES, input_dir, COSINE_THRESHOLD_VIRUS, COSINE_THRESHOLD_COUNTRY, DictList_syn_Virus_SEED,DictList_syn_Country_SEED, model_biobert,tokenizer_transformers_bio, model_transformers_bio, model_All, ):



    EXPAND_DICT = True
    DictList_syn_Virus_SEED, DictList_syn_Country_SEED, DictList_syn_Embeddings_Virus_SEED, DictList_syn_Embeddings_Country_SEED = CalculateEmbeddingsVocabs(DictList_syn_Virus_SEED, DictList_syn_Country_SEED, model_biobert, tokenizer_transformers_bio,
                                                    model_transformers_bio, model_All, EXPAND_DICT)


    MODELS = []
    VIRUSES = []
    COUNTRIES = []
    DICTS = []

    
    DictOVERALL_Virus = []
    DictOVERALL_Country = []
    for input_filename in LIST_FILES:

        modelName = input_filename.replace(input_dir, "").replace("OutputAnnotatedTexts-", "").replace(
            "Extractions.csv", "").replace(".csv", "")

        MODELS.append(modelName)

        print("\nCOMPUTING = " + modelName)

        df = pd.read_csv(input_filename, sep=',', header=0, encoding='utf-8')

        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        ######      
        
        
        
        DictList_syn_Virus, DictList_syn_Country, DictList_syn_Embeddings_Virus, DictList_syn_Embeddings_Country = attachToDicts(df,DictList_syn_Virus_SEED,DictList_syn_Country_SEED,DictList_syn_Embeddings_Virus_SEED,DictList_syn_Embeddings_Country_SEED,
                  COSINE_THRESHOLD_VIRUS,COSINE_THRESHOLD_COUNTRY,
                model_biobert,tokenizer_transformers_bio,model_transformers_bio,model_All)
           
            


        if (len(DictOVERALL_Virus) <= 0):
            DictOVERALL_Virus = DictList_syn_Virus.copy()
            DictOVERALL_Embeddings_Virus = DictList_syn_Embeddings_Virus.copy()
        else:

            print("\n...Merging the Virus Dictionaries")
            for ix_syn in tqdm(range(len(DictList_syn_Virus))):

                xSyn_at_ix_syn = DictList_syn_Virus[ix_syn].copy()
                x_emb = DictList_syn_Embeddings_Virus[ix_syn].copy()


                intersect = set()
                FoundAtIy = -1
                all_y_pred_list_SET = set([x.lower() for x in xSyn_at_ix_syn])
                for iy in range(len(DictOVERALL_Virus)):
                    b = DictOVERALL_Virus[iy]
                    b_set = set([x.lower() for x in b])
                    intersect = all_y_pred_list_SET.intersection(b_set)
                    if len(intersect) > 0:
                        FoundAtIy = iy
                        break

                if FoundAtIy > -1:
                    listenlarged = DictOVERALL_Virus[FoundAtIy] + xSyn_at_ix_syn  # list(intersect)
                    DictOVERALL_Virus[FoundAtIy] = listenlarged

                    if model_biobert:
                        DictOVERALL_Embeddings_Virus[FoundAtIy] = np.array(
                            [np.mean(model_biobert.encode(DictOVERALL_Virus[FoundAtIy]), axis=0)])
                    else:
                        DictOVERALL_Embeddings_Virus[FoundAtIy] = np.array([np.mean(
                            get_embeddings_transformer(DictOVERALL_Virus[FoundAtIy], tokenizer_transformers_bio,
                                                       model_transformers_bio).numpy(), axis=0)])

                else:

                    cosine_sim_INTERSECT = cosine_similarity([x_emb], DictOVERALL_Embeddings_Virus)[0]

                    maxcosine_INTERSECT = max(cosine_sim_INTERSECT)
                    idmaxcosine_INTERSECT = -1
                    if maxcosine_INTERSECT > COSINE_THRESHOLD_VIRUS:
                        idmaxcosine_INTERSECT = np.where(cosine_sim_INTERSECT == maxcosine_INTERSECT)[0][0]
                        listMerged = DictOVERALL_Virus[idmaxcosine_INTERSECT] + xSyn_at_ix_syn
                        listMerged = list(dict.fromkeys(listMerged))  # remove duplicates
                        DictOVERALL_Virus[idmaxcosine_INTERSECT] = listMerged
                        
						if model_biobert:
                            DictOVERALL_Embeddings_Virus[idmaxcosine_INTERSECT] = np.array(
                                [np.mean(model_biobert.encode(DictOVERALL_Virus[idmaxcosine_INTERSECT]), axis=0)])
                        else:
                            DictOVERALL_Embeddings_Virus[idmaxcosine_INTERSECT] = np.array([np.mean(
                                get_embeddings_transformer(DictOVERALL_Virus[idmaxcosine_INTERSECT],
                                                           tokenizer_transformers_bio,
                                                           model_transformers_bio).numpy(), axis=0)])

                    if idmaxcosine_INTERSECT < 0:  # it didn't find it, I need to add a new dictionary group
                        DictOVERALL_Virus.append(xSyn_at_ix_syn)
                        DictOVERALL_Embeddings_Virus = np.append(DictOVERALL_Embeddings_Virus, [x_emb], axis=0)

        if (len(DictOVERALL_Country) <= 0):
            DictOVERALL_Country = DictList_syn_Country.copy()
            DictOVERALL_Embeddings_Country = DictList_syn_Embeddings_Country.copy()
        else:

            print("\n...Merging the Country Dictionaries")
            for ix_syn in tqdm(range(len(DictList_syn_Country))):

                xSyn_at_ix_syn = DictList_syn_Country[ix_syn].copy()
                x_emb = DictList_syn_Embeddings_Country[ix_syn].copy()

                intersect = set()
                FoundAtIy = -1
                all_y_pred_list_SET = set([x.lower() for x in xSyn_at_ix_syn])
                for iy in range(len(DictOVERALL_Country)):
                    b = DictOVERALL_Country[iy]
                    b_set = set([x.lower() for x in b])
                    intersect = all_y_pred_list_SET.intersection(b_set)
                    if len(intersect) > 0:
                        FoundAtIy = iy
                        break

                if FoundAtIy > -1:
                    listenlarged = DictOVERALL_Country[FoundAtIy] + xSyn_at_ix_syn  # list(intersect)
                    DictOVERALL_Country[FoundAtIy] = listenlarged

                    if model_biobert:
                        DictOVERALL_Embeddings_Country[FoundAtIy] = np.array(
                            [np.mean(model_biobert.encode(DictOVERALL_Country[FoundAtIy]), axis=0)])
                    else:
                        DictOVERALL_Embeddings_Country[FoundAtIy] = np.array([np.mean(
                            get_embeddings_transformer(DictOVERALL_Country[FoundAtIy], tokenizer_transformers_bio,
                                                       model_transformers_bio).numpy(), axis=0)])


                else:
                    cosine_sim_INTERSECT = cosine_similarity([x_emb], DictOVERALL_Embeddings_Country)[0]

                    maxcosine_INTERSECT = max(cosine_sim_INTERSECT)
                    idmaxcosine_INTERSECT = -1
                    if maxcosine_INTERSECT > COSINE_THRESHOLD_COUNTRY:
                        idmaxcosine_INTERSECT = np.where(cosine_sim_INTERSECT == maxcosine_INTERSECT)[0][0]
                        listMerged = DictOVERALL_Country[idmaxcosine_INTERSECT] + xSyn_at_ix_syn
                        DictOVERALL_Country[idmaxcosine_INTERSECT] = listMerged
                        DictOVERALL_Embeddings_Country[idmaxcosine_INTERSECT] = np.array(
                            [np.mean(model_All.encode(DictOVERALL_Country[idmaxcosine_INTERSECT]), axis=0)])

                    if idmaxcosine_INTERSECT < 0:  # it didn't find it, I need to add a new dictionary group
                        DictOVERALL_Country.append(xSyn_at_ix_syn)
                        DictOVERALL_Embeddings_Country = np.append(DictOVERALL_Embeddings_Country, [x_emb], axis=0)

            
        #################################################

    
    
    df_synOVERALL_Virus = attachCentroidsToDict(DictOVERALL_Virus,DictList_syn_Virus_SEED)
        
    
    
    df_synOVERALL_Country = attachCentroidsToDict(DictOVERALL_Country,DictList_syn_Country_SEED)
    

    return df_synOVERALL_Virus, df_synOVERALL_Country



def main():

    DEBUG = False

    COSINE_THRESHOLD_VIRUS = 0.7 #0.7 
    COSINE_THRESHOLD_COUNTRY = 0.8 #0.7
    
    input_specify = Path(__file__).parent.resolve() / Path("/eos/jeodpp/data/projects/ETOHA/DATA/etohaSurveillanceScraper/corpus_processed/SUMMARIZED/")

    input_dir = str(input_specify).strip() + "/"

    LIST_FILES = [        
    (input_dir + "OutputAnnotatedTexts-llama-3-70b-instruct.csv"),
    (input_dir + "OutputAnnotatedTexts-mistral-7b-openorca.csv"),
    (input_dir + "OutputAnnotatedTexts-zephyr-7b-beta.csv"),
        ]

    month_abbreviations = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    week_abbreviations = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    #print("Get embeddings for terms texts...")
    model_All = SentenceTransformer('all-mpnet-base-v2')  # https://www.sbert.net/docs/pretrained_models.html#model-overview

    model_biobert = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    
    tokenizer_transformers_bio = AutoTokenizer.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    model_transformers_bio = AutoModel.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    
    somesyn_filename_Virus = input_dir + "defsyn-Dictionary-Viruses_SEED.csv"
    df_syn_Virus_SEED = pd.read_csv(somesyn_filename_Virus, sep=',', header=None, encoding='latin1')
    DictList_syn_Virus_SEED = df_syn_Virus_SEED.stack().groupby(level=0).apply(list).tolist()

    somesyn_filename_Country = input_dir + "defsyn-Dictionary-Countries_SEED.csv"
    df_syn_Country_SEED = pd.read_csv(somesyn_filename_Country, sep=',', header=None, encoding='latin1')
    DictList_syn_Country_SEED = df_syn_Country_SEED.stack().groupby(level=0).apply(list).tolist()




    start = time.time()

    df_synOVERALL_Virus, df_synOVERALL_Country = run_dictionaryPopulation(LIST_FILES, input_dir, COSINE_THRESHOLD_VIRUS,
                                                                          COSINE_THRESHOLD_COUNTRY,
                                                                          DictList_syn_Virus_SEED,
                                                                          DictList_syn_Country_SEED, model_biobert,
                                                                          tokenizer_transformers_bio,
                                                                          model_transformers_bio, model_All)




    newdict_filename = somesyn_filename_Virus.replace("Dictionary", "Dictionary-NEW-utf8").replace("_SEED", "")
    df_synOVERALL_Virus.to_csv(newdict_filename, sep=',', index=True, index_label=['REPRESENTATIVE-LABEL'], encoding='utf-8')  # header=None,

    newdict_filename = somesyn_filename_Country.replace("Dictionary", "Dictionary-NEW-utf8").replace("_SEED", "")
    df_synOVERALL_Country.to_csv(newdict_filename, sep=',', index=True, index_label=['REPRESENTATIVE-LABEL'], encoding='utf-8') #header=None,

    #
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nOverall Computational Time... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #



    print("\nEnd Computations")





if __name__ == "__main__":
    main()
