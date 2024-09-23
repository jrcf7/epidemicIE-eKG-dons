
import os

from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Union

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import collections
from nltk.corpus import wordnet as wn
import evaluate
import csv
import datetime
import logging
from tqdm import tqdm
import time
import json
from dateutil.parser import isoparse
import requests

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


from llmquery import setup_openai, api_call_openai, model_list_openai, call_model, call_model_with_caching, process_list, setup_gptjrc, api_call_gptjrc, model_list_gptjrc, token_counter, encoding_getter

from epidemicExtractor_Deployment import run_epidemicExtractor

from llm_extraction_Dictionary_Deployment import run_dictionaryPopulation, attachCentroidsToDict, CalculateEmbeddingsVocabs, attachToDicts

from llm_extraction_majorityVoting_Deployment import run_EnsembleComputation


import warnings
warnings.filterwarnings('ignore', message='TypedStorage is deprecated.')


def Most_Common(lista):
    country = ""
    if lista:
        data = Counter(lista)
        ordered_c = data.most_common()
        country = ordered_c[0][0]
        max_freq = ordered_c[0][1]
        for j in range(0, len(ordered_c)):
            if ordered_c[j][1] < max_freq:
                break
            else:
                if ordered_c[j][0] == "COUNTRY.lower()":
                    country = ordered_c[j][0]
    
    return country


from collections import Counter


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def get_EIOS_APIs_token():

    import config
    url = "https://login.microsoftonline.com/f610c0b7-bd24-4b39-810b-3dc280afb590/oauth2/v2.0/token"

    querystring = {"grant_type": "application/x-www-form-urlencoded"}

    payload = "grant_type=client_credentials&client_id%20="+config.eios_client_id+"&client_secret="+config.eios_client_secret+"&scope="+config.eios_scope+"&undefined="
    headers = {
        'cache-control': "no-cache",
    }

    response = requests.request("POST", url, data=payload, headers=headers, params=querystring)


    data = response.json()
    actk = data["access_token"]

    return actk
    #

def getPromedArticles(startdate_promed_ts_string,enddate_promed_ts_string,EIOS_PromedBoard_ID,EIOS_APIs_TKN):
    # get promed
    url = "https://portal.who.int/eios/API/News/Service/GetBoardArticles"

    # querystring = {"timespan": "2023-06-01T00:00", "untilTimespan": "2023-06-02T00:00", "boardId": +EIOS_PromedBoard_ID, "limit": "300"}
    querystring = {"timespan": startdate_promed_ts_string, "untilTimespan": enddate_promed_ts_string,
                   "boardId": EIOS_PromedBoard_ID, "limit": "300"}

    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'Authorization': "Bearer " + EIOS_APIs_TKN,
        'cache-control': "no-cache",
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    #print(response.text)

    return response.json()
    #


def getDONSArticles(startdate_dons_ts_string,enddate_dons_ts_string,EIOS_WhoDonsBoard_ID,EIOS_APIs_TKN):
    # get whodons
    url = "https://portal.who.int/eios/API/News/Service/GetBoardArticles"

    querystring = {"timespan": startdate_dons_ts_string, "untilTimespan": enddate_dons_ts_string,
                   "boardId": EIOS_WhoDonsBoard_ID, "limit": "300"}

    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'Authorization': "Bearer " + EIOS_APIs_TKN,
        'cache-control': "no-cache",
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    #print(response.text)

    return response.json()
    #



def main(synk_location):

    print("SYS ARGV: ")
    print(sys.argv)
    print("\n")

    DEBUG = True

    EIOS_PromedBoard_ID = "14714"
    EIOS_WhoDonsBoard_ID = "14715"

    MAX_TOKENS_PROMPT = 4096  # for GPT3.5 - llama-3-70b-instruct - mistral-7b-openorca - zephyr-7b-beta

    TOKENS_TOLERANCE = 1200  # 800 was good for llama-3-70b-instruct  #5000 tokens are for the InContext examples

    USE_CACHE = True  # True #False

    DATE_IMPUTATION = True # False

    JSON_RECONSTRUCT = True  # True False

    # service_provider = "openai"
    # model_name = "gpt-3.5-turbo-16k"
    #
    #
    # service_provider = "dglc"
    # dglc available models: 'OA_SFT_Pythia_12B', 'JRC_RHLF_13B', 'OA_GPT3.5', 'OA_GPT3'
    # model_name = "gpt-3.5-turbo"  #OpenAI name
    # model_name = 'JRC_RHLF_13B'
    # model_name = "OA_SFT_Pythia_12B"   #EleutherAI-pythia-12b
    # model_name = "OA_GPT3"
    # model_name = "GPT@JRC_4"
    #
    #
    service_provider = "gptjrc"

    # temperature: temperature_value (0: precise, 1: creative)
    temperature_value = 0.01  # 0.1

    MODELS = ["llama-3-70b-instruct", "mistral-7b-openorca", "zephyr-7b-beta"]

    InContextExamples = []
    
    input_specify = Path(__file__).parent.resolve() / Path(synk_location)

    input_dir = str(input_specify).strip() + "/"

    month_abbreviations = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    week_abbreviations = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    ########## PARAMS FOR DICTIONARY

    COSINE_THRESHOLD_VIRUS = 0.7  # 0.7 # good for ALL-MPNET    #COSINE_THRESHOLD_VIRUS = 0.8 #0.7 #0.6
    COSINE_THRESHOLD_COUNTRY = 0.8  # 0.7

    try:
        
        tokenizer_transformers_bio = AutoTokenizer.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    except Exception as err:
        print(f"Error: An error occurred when loading tokenizer BioBERT: {err}")
        logging.error(f"Error: An error occurred when loading tokenizer BioBERT: {err}")


    try:
        model_transformers_bio = AutoModel.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        
    except Exception as err:
        print(f"Error: An error occurred when loading model_transformers_bio BioBERT: {err}")
        logging.error(f"Error: An error occurred when loading model_transformers_bio BioBERT: {err}")

    try:
        # print("Get embeddings for terms texts...")
        
        model_All = SentenceTransformer(
            'all-mpnet-base-v2')  # https://www.sbert.net/docs/pretrained_models.html#model-overview
    except Exception as err:
        print(f"Error: An error occurred when loading all-mpnet-base-v2: {err}")
        logging.error(f"Error: An error occurred when loading all-mpnet-base-v2 : {err}")

    try:
        model_biobert = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    except Exception as err:
        print(f"Error: An error occurred when loading all-mpnet-base-v2: {err}")
        logging.error(f"Error: An error occurred when loading all-mpnet-base-v2 : {err}")






    somesyn_filename_Virus = input_dir + "defsyn-Dictionary-NEW-utf8-Viruses.csv"   
    df_syn_Virus = pd.read_csv(somesyn_filename_Virus, sep=',', header=0, dtype=str, encoding='utf-8') 
    df_syn_Virus.drop(columns=df_syn_Virus.columns[0], axis=1, inplace=True) # remove first column that is the index
    DictList_syn_Virus = df_syn_Virus.stack().groupby(level=0).apply(list).tolist()

    somesyn_filename_Country = input_dir + "defsyn-Dictionary-NEW-utf8-Countries.csv"   
    df_syn_Country = pd.read_csv(somesyn_filename_Country, sep=',', header=0, dtype=str, encoding='utf-8')  
    df_syn_Country.drop(columns=df_syn_Country.columns[0], axis=1, inplace=True) # remove first column that is the index
    DictList_syn_Country = df_syn_Country.stack().groupby(level=0).apply(list).tolist()
    
    
    ##########

    # OpenAI ChatGPT API
    if service_provider == "openai":
        MyOpenAPIKey = ""
        fkeyname = "OpenAI-DigLifeAccount-APItoken.key"
        if os.path.exists(fkeyname):
            with open(fkeyname) as f:
                MyOpenAPIKey = f.read()
        setup_openai(key=MyOpenAPIKey)

    
    #### GPT@JRC API
    if service_provider == "gptjrc":
        key_gptjrc = ""
        fkeyname = "GPTJRC-APItoken.key"
        if os.path.exists(fkeyname):
            with open(fkeyname) as f:
                key_gptjrc = f.read()
        setup_gptjrc(key_gptjrc)

    
    ###########################################################################

    # Create new `pandas` methods which use `tqdm` progress
    # (can use tqdm_gui, optional kwargs, etc.)
    tqdm.pandas()

    #############################################################################################

    # save input texts
    ensemble_filename = input_dir + "OutputAnnotatedTexts-LLMs-ENSEMBLE.csv"
    df_Ensemble = pd.read_csv(ensemble_filename, sep=',', header=0, dtype=str, encoding='utf-8')

   
    #############################################################################################

    start = time.time()

    df_tofilter = df_Ensemble[df_Ensemble['fileid'].str.contains('TOTAL_ProMED')].drop(columns=[col for col in df_Ensemble if "date_cases_IMPUTED" not in col])
    maxdate_promed = pd.to_datetime(df_tofilter['date_cases_IMPUTED'],errors='coerce',utc=True).max(axis=0)
    if pd.isnull(maxdate_promed):
        df_tofilter = df_Ensemble[df_Ensemble['fileid'].str.contains('TOTAL_ProMED')].drop(columns=[col for col in df_Ensemble if "date_extracted" not in col])
        maxdate_promed = pd.to_datetime(df_tofilter['date_extracted'],errors='coerce',utc=True).max(axis=0)

    df_tofilter = df_Ensemble[df_Ensemble['fileid'].str.contains('who_dons')].drop(columns=[col for col in df_Ensemble if "date_cases_IMPUTED" not in col])
    maxdate_dons = pd.to_datetime(df_tofilter['date_cases_IMPUTED'],errors='coerce',utc=True).max(axis=0)
    if pd.isnull(maxdate_dons):
        df_tofilter = df_Ensemble[df_Ensemble['fileid'].str.contains('who_dons')].drop(columns=[col for col in df_Ensemble if "date_extracted" not in col])
        maxdate_dons = pd.to_datetime(df_tofilter['date_extracted'],errors='coerce',utc=True).max(axis=0)

    

    ###
    dataToPopulate = []
    ###

    dtnow = datetime.datetime.now(datetime.timezone.utc)
    tmsdtnow = pd.to_datetime(dtnow, errors='coerce', utc=True)

    startdate_promed = maxdate_promed + datetime.timedelta(days=1)
    enddate_promed = startdate_promed + datetime.timedelta(days=1)
    startdate_promed_ts_string = startdate_promed.strftime("%Y-%m-%dT%H:%M")
    enddate_promed_ts_string = enddate_promed.strftime("%Y-%m-%dT%H:%M")

    numbersOfDaysPromed = tmsdtnow - startdate_promed
    print("PROMED - Number of days required = " + str(numbersOfDaysPromed.days) + " - Period: " + startdate_promed.strftime("%d %b %Y") + " - " + tmsdtnow.strftime("%d %b %Y"))
    print("\n")

    while enddate_promed < tmsdtnow:

        if DEBUG:
            print("\nGetting PROMED full-texts via EIOS - " + startdate_promed.strftime("%d %b %Y") + " - " + enddate_promed.strftime("%d %b %Y"))

        EIOS_APIs_TKN = get_EIOS_APIs_token()

        promedArticlesJson=getPromedArticles(startdate_promed_ts_string, enddate_promed_ts_string, EIOS_PromedBoard_ID, EIOS_APIs_TKN)

        ### INPUT DATAFRAME : ###

        if promedArticlesJson and promedArticlesJson["count"]>0:

            for idYY, YY in enumerate(promedArticlesJson["result"]):
                #iiii = promedArticlesJson["result"][idYY]

                a=""
                if YY['eiosUrl'] and len(YY['eiosUrl'])>0:
                    a = "TOTAL_ProMED/"+YY['eiosUrl']
                elif YY['id'] and len(YY['id'])>0:
                    a = "TOTAL_ProMED/" + YY['id']
                elif YY['rssItemId'] and len(YY['rssItemId']) > 0:
                    a = "TOTAL_ProMED/" + YY['rssItemId']
                elif YY['link'] and len(YY['link']) > 0:
                    a = "TOTAL_ProMED/" + YY['link']

                b=""
                if YY['fullText'] and len(YY['fullText'])>0:
                    b = YY['fullText']

                c=""
                if DATE_IMPUTATION == True:
                    if YY['pubDate'] and len(YY['pubDate'])>0:
                        date_str = YY['pubDate'] #'2023-06-23T22:30:00.0000000Z'
                        #date_obj = isoparse(date_str)
                        date_obj = datetime.datetime.strptime(date_str[:-2] + 'Z', '%Y-%m-%dT%H:%M:%S.%fZ')
                        c = date_obj.strftime("%Y-%m-%d")
                    elif YY['processedOnDate'] and len(YY['processedOnDate']) > 0:
                        date_str = YY['processedOnDate']  # '2023-06-23T22:30:00.0000000Z'
                        # date_obj = isoparse(date_str)
                        date_obj = datetime.datetime.strptime(date_str[:-2] + 'Z', '%Y-%m-%dT%H:%M:%S.%fZ')
                        c = date_obj.strftime("%Y-%m-%d")

                if a and b and c:
                    dataToPopulate.append([a, b, c])


        startdate_promed = startdate_promed + datetime.timedelta(days=1)
        enddate_promed = startdate_promed + datetime.timedelta(days=1)
        startdate_promed_ts_string = startdate_promed.strftime("%Y-%m-%dT%H:%M")
        enddate_promed_ts_string = enddate_promed.strftime("%Y-%m-%dT%H:%M")

    # end for cycle promed


    ### GETTING DONs
    startdate_dons = maxdate_dons + datetime.timedelta(days=1)
    enddate_dons = startdate_dons + datetime.timedelta(days=1)
    startdate_dons_ts_string = startdate_dons.strftime("%Y-%m-%dT%H:%M")
    enddate_dons_ts_string = enddate_dons.strftime("%Y-%m-%dT%H:%M")

    numbersOfDaysDONs = tmsdtnow - startdate_dons
    print("\nDONS - Number of days required = " + str(numbersOfDaysDONs.days) + " - Period: " + startdate_dons.strftime("%d %b %Y") + " - " + tmsdtnow.strftime("%d %b %Y"))
    print("\n")

    while enddate_dons < tmsdtnow:

        if DEBUG:
            print("\nGETTING DONs full-texts via EIOS - " + startdate_dons.strftime("%d %b %Y") + " - " + enddate_dons.strftime(
                "%d %b %Y"))

        EIOS_APIs_TKN = get_EIOS_APIs_token()

        donsArticlesJson = getDONSArticles(startdate_dons_ts_string, enddate_dons_ts_string, EIOS_WhoDonsBoard_ID,EIOS_APIs_TKN)

        ### INPUT DATAFRAME : ###

        if donsArticlesJson and donsArticlesJson["count"] > 0:

            for idYY, YY in enumerate(donsArticlesJson["result"]):

                a = ""
                if YY['eiosUrl'] and len(YY['eiosUrl']) > 0:
                    a = "who_dons/" + YY['eiosUrl']
                elif YY['id'] and len(YY['id']) > 0:
                    a = "who_dons/" + YY['id']
                elif YY['rssItemId'] and len(YY['rssItemId']) > 0:
                    a = "who_dons/" + YY['rssItemId']
                elif YY['link'] and len(YY['link']) > 0:
                    a = "who_dons/" + YY['link']

                b = ""
                if YY['fullText'] and len(YY['fullText']) > 0:
                    b = YY['fullText']

                c = ""
                if DATE_IMPUTATION == True:
                    if YY['pubDate'] and len(YY['pubDate']) > 0:
                        date_str = YY['pubDate']  # '2023-06-23T22:30:00.0000000Z'
                        # date_obj = isoparse(date_str)
                        date_obj = datetime.datetime.strptime(date_str[:-2] + 'Z', '%Y-%m-%dT%H:%M:%S.%fZ')
                        c = date_obj.strftime("%Y-%m-%d")
                    elif YY['processedOnDate'] and len(YY['processedOnDate']) > 0:
                        date_str = YY['processedOnDate']  # '2023-06-23T22:30:00.0000000Z'
                        # date_obj = isoparse(date_str)
                        date_obj = datetime.datetime.strptime(date_str[:-2] + 'Z', '%Y-%m-%dT%H:%M:%S.%fZ')
                        c = date_obj.strftime("%Y-%m-%d")

                if a and b and c:
                    dataToPopulate.append([a, b, c])

        startdate_dons = startdate_dons + datetime.timedelta(days=1)
        enddate_dons = startdate_dons + datetime.timedelta(days=1)
        startdate_dons_ts_string = startdate_dons.strftime("%Y-%m-%dT%H:%M")
        enddate_dons_ts_string = enddate_dons.strftime("%Y-%m-%dT%H:%M")

    # end for cycle dons

    #
    endGettingFromEIOS = time.time()
    hours, rem = divmod(endGettingFromEIOS - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for getting the full-text articles from EIOS... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #


    if len(dataToPopulate)<=0:        
        print("\nNo data to insert for the period....exiting now....")
        #sys.exit()
        return
    

    df = pd.DataFrame(dataToPopulate, columns=['fileid', 'texts', 'date_cases_IMPUTED'])


    df['texts'] = df['texts'].replace(r'\(\'\\n\\n', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\(\"\\n\\n', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\\n\\n\',\)', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\\n\\n\",\)', ' ', regex=True)

    while df['texts'].str.contains(r'##\n').any():
        df['texts'] = df['texts'].str.replace(r"##\n", '. ', regex=True)
    while df['texts'].str.contains('###').any():
        df['texts'] = df['texts'].str.replace("###", ' ')
    while df['texts'].str.contains('##').any():
        df['texts'] = df['texts'].str.replace("##", ' ')
    while df['texts'].str.contains(' # ').any():
        df['texts'] = df['texts'].str.replace(" # ", ' ')
    while df['texts'].str.contains('--').any():
        df['texts'] = df['texts'].str.replace("--", '-')
    while df['texts'].str.contains(r'\\\\-').any():
        df['texts'] = df['texts'].str.replace(r"\\\\-", '.', regex=True)
    while df['texts'].str.contains(r'\*\*\n').any():
        df['texts'] = df['texts'].str.replace(r"\*\*\n", '. ', regex=True)
    while df['texts'].str.contains(r'\*\*\*').any():
        df['texts'] = df['texts'].str.replace(r"\*\*\*", ' ', regex=True)
    while df['texts'].str.contains(r'\*\*').any():
        df['texts'] = df['texts'].str.replace(r"\*\*", ' ', regex=True)
    while df['texts'].str.contains(r' \* ').any():
        df['texts'] = df['texts'].str.replace(r" \* ", ' ', regex=True)
    while df['texts'].str.contains(r'is a program of the\n\nInternational Society for Infectious Diseases').any():
        df['texts'] = df['texts'].replace(r'is a program of the\n\nInternational Society for Infectious Diseases',
                                          'is a program of the International Society for Infectious Diseases',
                                          regex=True)

    
    while df['texts'].str.contains(r' \*\.').any():
        df['texts'] = df['texts'].str.replace(r' \*\.', ' .', regex=True)
    while df['texts'].str.contains('  ').any():
        df['texts'] = df['texts'].str.replace("  ", ' ')
    while df['texts'].str.contains(r'\.\.').any():
        df['texts'] = df['texts'].str.replace(r'\.\.', '.', regex=True)
    while df['texts'].str.contains(r'\. \.').any():
        df['texts'] = df['texts'].str.replace(r'\. \.', '.', regex=True)

    df['texts'] = df['texts'].replace(r'\(\"\.', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\(\'\.', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\",\)', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\',\)', ' ', regex=True)

    df['texts'] = df['texts'].astype(str).str.strip()

    df['texts_length'] = df['texts'].astype(str).str.strip().str.len()

    ########################## END ALL PREPROCESSING


    dir_path_iteration_day = input_dir + tmsdtnow.strftime("%Y-%m-%d") + "/"  #"/path/to/your/directory"
    if not os.path.exists(dir_path_iteration_day):
        os.makedirs(dir_path_iteration_day)

    print("\n...RUNNING the Epidemic Extractor using the LLMs...")
    print("\n")

    LIST_FILES = []

    for model_name in MODELS:

        print("Model = " + model_name)

        if InContextExamples:
            for row in InContextExamples:
                for col in row:
                    nt = token_counter(col, model_name)
                    # print("\nNumber of Tokens in the example = " + str(nt))
                    ntotExamplesTokens = ntotExamplesTokens + nt
            #
            print("\nNumber of Tokens of the all examples in the json extraction = " + str(ntotExamplesTokens))
            TOKENS_TOLERANCE = TOKENS_TOLERANCE + ntotExamplesTokens
            print("\nUpdated TOKENS_TOLERANCE to " + str(TOKENS_TOLERANCE))

        ###########################################################################

        cache_name=""
        load_map_query_input_output = {}
        if USE_CACHE:
            # cache_prefix_fp: prefix of the file to which write content of cache after each call
            cache_prefix_fp = "LLMQUERYDEPLOY"
            cache_name = cache_prefix_fp + "___" + "__".join(
                [service_provider, model_name, str(temperature_value)]).replace(
                " ", "_") + ".json"

            if os.path.exists(cache_name):

                with open(cache_name) as f:
                    load_map_query_input_output = json.load(f)
                
            else:
                load_map_query_input_output = {}

        ###########################################################################

        ##################################################################################################

        date_imputation_inner=False
        ddf = run_epidemicExtractor(dir_path_iteration_day,model_name,df,date_imputation_inner,MAX_TOKENS_PROMPT,TOKENS_TOLERANCE,service_provider,USE_CACHE,load_map_query_input_output,cache_name,JSON_RECONSTRUCT,temperature_value,InContextExamples)

        # save
        output_texts_filename = str(dir_path_iteration_day) + "/OutputAnnotatedTexts-" + model_name + ".csv"

        ddf.to_csv(output_texts_filename, sep=',', header=True, index=True, encoding='utf-8')

        LIST_FILES.append(output_texts_filename)

        ##################################################################################################

    #
    endRunningEpidemicExtractor = time.time()
    hours, rem = divmod(endRunningEpidemicExtractor - endGettingFromEIOS, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for running the Epidemic Extractor using the LLMs... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #



    #### UPDATING DICTIONARY

    print("\n...UPDATING DICTIONARY...")
    print("\n")
    
    DictList_syn_Virus_SEED = DictList_syn_Virus.copy()
    DictList_syn_Country_SEED = DictList_syn_Country.copy()
    
    EXPAND_DICT = False
    DictList_syn_Virus, DictList_syn_Country, DictList_syn_Embeddings_Virus, DictList_syn_Embeddings_Country = CalculateEmbeddingsVocabs(DictList_syn_Virus, DictList_syn_Country, model_biobert, tokenizer_transformers_bio,
                                                    model_transformers_bio, model_All, EXPAND_DICT)

    #

    
    endCalculatingEmbeddingsVocas = time.time()
    hours, rem = divmod(endCalculatingEmbeddingsVocas - endRunningEpidemicExtractor, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for Calculating Embeddings of the Vocabs... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #

    for input_filename in LIST_FILES:

        print("\nCOMPUTING = " + input_filename)

        df_iterSample = pd.read_csv(input_filename, sep=',', header=0, encoding='utf-8')

        df_iterSample = df_iterSample.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        ######  

        DictList_syn_Virus, DictList_syn_Country, DictList_syn_Embeddings_Virus, DictList_syn_Embeddings_Country = attachToDicts(df_iterSample,DictList_syn_Virus,DictList_syn_Country,DictList_syn_Embeddings_Virus,DictList_syn_Embeddings_Country,
                    COSINE_THRESHOLD_VIRUS,COSINE_THRESHOLD_COUNTRY,
                    model_biobert,tokenizer_transformers_bio,model_transformers_bio,model_All)

               
    
    df_synOVERALL_Virus = attachCentroidsToDict(DictList_syn_Virus,DictList_syn_Virus_SEED)    
    df_synOVERALL_Country  = attachCentroidsToDict(DictList_syn_Country,DictList_syn_Country_SEED)
    
    
    
    newdict_filename = somesyn_filename_Virus  #.replace("Dictionary", "Dictionary-NEW-utf8").replace("_SEED", "")
    df_synOVERALL_Virus.to_csv(newdict_filename, sep=',', index=True, index_label=['REPRESENTATIVE-LABEL'],
                               encoding='utf-8')  # header=None,
    
    newdict_filename = somesyn_filename_Country  #.replace("Dictionary", "Dictionary-NEW-utf8").replace("_SEED", "")
    df_synOVERALL_Country.to_csv(newdict_filename, sep=',', index=True, index_label=['REPRESENTATIVE-LABEL'],
                                 encoding='utf-8')  # header=None,
    

    #
    endRunningDictPopulation = time.time()
    hours, rem = divmod(endRunningDictPopulation - endCalculatingEmbeddingsVocas, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for running the Dictionary Population pipeline... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #




    ######### RUNNUNING ENSEMBLE for the UPDATE

    print("\n...RUNNING ENSEMBLE for the UPDATE...")
    print("\n")

        
    
    df_syn_Virus = pd.read_csv(somesyn_filename_Virus, sep=',', header=0, dtype=str, encoding='utf-8')
    DictList_syn_Virus = df_syn_Virus.stack().groupby(level=0).apply(list).tolist()

    df_syn_Country = pd.read_csv(somesyn_filename_Country, sep=',', header=0, dtype=str, encoding='utf-8')
    DictList_syn_Country = df_syn_Country.stack().groupby(level=0).apply(list).tolist()
        

    df_ensemble_iter = run_EnsembleComputation(LIST_FILES, dir_path_iteration_day, week_abbreviations, month_abbreviations, DictList_syn_Virus, DictList_syn_Country)

    #save iter ensemble
    output_texts_filename = dir_path_iteration_day + "OutputAnnotatedTexts-LLMs-ENSEMBLE-IterUpdate.csv"
    

    df_ensemble_iter.to_csv(output_texts_filename, sep=',', header=True, index=False, encoding='utf-8')

    ####concat and save the overall ensemble

    
    df_concat_ensemble = pd.concat([df_Ensemble, df_ensemble_iter], ignore_index=True)

    df_concat_ensemble = df_concat_ensemble.drop_duplicates()  # drop any eventual duplicate rows

    # save input texts
    output_texts_filename = input_dir + "OutputAnnotatedTexts-LLMs-ENSEMBLE.csv"
    df_concat_ensemble.to_csv(output_texts_filename, sep=',', header=True, index=False, encoding='utf-8')

    ######## df_ensemble_REDUCED:
    
    #if "/JRC-OpenData/" in synk_location:
    df_ensemble_REDUCED = df_concat_ensemble.copy()
    if "texts" in df_ensemble_REDUCED.columns:
        df_ensemble_REDUCED = df_ensemble_REDUCED.drop(columns=["texts"])
    if "texts_length" in df_ensemble_REDUCED.columns:
        df_ensemble_REDUCED = df_ensemble_REDUCED.drop(columns=["texts_length"])
    if "fileid" in df_ensemble_REDUCED.columns:
        df_ensemble_REDUCED['fileid'] = np.where(df_ensemble_REDUCED['fileid'].str.contains('ProMED', case=True),'P',
                                                 df_ensemble_REDUCED['fileid'])
        df_ensemble_REDUCED['fileid'] = np.where(df_ensemble_REDUCED['fileid'].str.contains('dons', case=True), 'D',
                                                 df_ensemble_REDUCED['fileid'])

    output_texts_filename = input_dir + "OutputAnnotatedTexts-LLMs-ENSEMBLE_reduced.csv"
    df_ensemble_REDUCED.to_csv(output_texts_filename, sep=',', header=True, index=False, encoding='utf-8')

    ########

    ######## df_ensemble_donsOnly:

    df_ensemble_donsOnly = df_concat_ensemble.copy()
    if "texts" in df_ensemble_donsOnly.columns:
        df_ensemble_donsOnly = df_ensemble_donsOnly.drop(columns=["texts"])
    if "texts_length" in df_ensemble_donsOnly.columns:
        df_ensemble_donsOnly = df_ensemble_donsOnly.drop(columns=["texts_length"])

    if "fileid" in df_ensemble_donsOnly.columns:
        df_ensemble_donsOnly = df_ensemble_donsOnly[~df_ensemble_donsOnly['fileid'].str.contains('ProMED', case=True)]
        df_ensemble_donsOnly['fileid'] = df_ensemble_donsOnly['fileid'].str.replace("who_dons/", '')
        df_ensemble_donsOnly['fileid'] = df_ensemble_donsOnly['fileid'].str.replace(".pickle", '')
        df_ensemble_donsOnly['fileid'] = df_ensemble_donsOnly['fileid'].str.replace("/full-article", '')
        df_ensemble_donsOnly['fileid'] = df_ensemble_donsOnly['fileid'].str.replace("https://portal.who.int/eios#/items/", '')


    output_texts_filename = input_dir + "OutputAnnotatedTexts-LLMs-ENSEMBLE_whoDons.csv"
    df_ensemble_donsOnly.to_csv(output_texts_filename, sep=',', header=True, index=False, encoding='utf-8')

    ########




    #
    endRunningEnsemble = time.time()
    hours, rem = divmod(endRunningEnsemble - endRunningDictPopulation, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for running the Ensemble pipeline... {:0>2}:{:0>2}:{:05.2f}\n".format(
        int(hours), int(minutes), seconds))
    #


    ######### ######### ######### ######### #########

    #
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nOverall Computational Time... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #

    print("\nEnd Computations")





if __name__ == "__main__":

    synk_location = "/eos/jeodpp/data/projects/ETOHA/DATA/etohaSurveillanceScraper/corpus_processed/SUMMARIZED/"
    if (("/jupyter/" in str(sys.argv)) == False) and len(sys.argv) > 1:
        synk_location = sys.argv[1]
    print("synk_location = " + str(synk_location) + "\n")

    main(synk_location)
