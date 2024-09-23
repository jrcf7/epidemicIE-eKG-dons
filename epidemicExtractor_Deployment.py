import os
import sys

import openai
import json
import time
from tqdm import tqdm

from functools import partial
import pandas as pd
import numpy as np
import ast

import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter  # https://python.langchain.com/docs/modules/data_connection/document_transformers/
from langchain.text_splitter import TokenTextSplitter

import logging

from llmquery import setup_openai, api_call_openai, model_list_openai, call_model, call_model_with_caching, process_list, setup_gptjrc, api_call_gptjrc, model_list_gptjrc, token_counter, encoding_getter

import re

from nltk.corpus.reader.api import CorpusReader
from pathlib import Path
from typing import Iterator, List, Optional
import pickle

import tiktoken

""" Epidemic Extractor

features:
- build in retry in case of error
- cache the results in case of crash
- call llm with a lambda

supported API:
- OpenAI
- GPT@JRC
"""



class PickledCorpusReader(CorpusReader):
    def __init__(
        self,
        root: Path = (
            #Path(__file__).parent.resolve() / Path("../data/corpus_processed/") #commented consose 20230601
                Path(__file__).parent.resolve() / Path("/eos/jeodpp/data/projects/ETOHA/DATA/etohaSurveillanceScraper/corpus_processed/")  # commented consose 20230601
        ),
        fileids=r".+\.pickle",
    ):
        """
        Initialize the corpus reader.

        Keyword Arguments:
            root -- Path of corpus root.
            fileids -- Regex pattern for documents.
        """
        CorpusReader.__init__(self, str(root), fileids)

    def get_fileids(self, ) -> List[str]:
        fileids = self.fileids()
        return fileids
    def docs(self, fileids: Optional[List[str]] = None,) -> Iterator[str]:
        """Returns the processes text of a pickled corpus document.

        Arguments:
            fileids -- Filenames of corpus documents.

        Yields:
            Generator of processed documents of corpus.
        """
        if fileids is None:
            fileids = self.fileids()

        for path, enc in self.abspaths(fileids, include_encoding=True):
            with open(path, "rb") as f:
                yield pickle.load(f)


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True

def json_reconstruction(myjson: str, attributes: list[str]):

    try:

        if not(myjson):
            myjson = '{\"virus\": \"\", \"country\": \"\", "date": \"\", \"cases\": \"\", \"deaths\": \"\"}'

        jsonloaded = json.loads(myjson)

        if jsonloaded:
            dictkeyslist = list(jsonloaded.keys())

            notExpectedAttributes = list(set(dictkeyslist) - set(attributes))
            if notExpectedAttributes:
                for notExp in notExpectedAttributes:
                    del jsonloaded[notExp]
                    myjson = json.dumps(jsonloaded)

    except ValueError as e:


        if myjson.lower().strip() == "json" or myjson.lower().strip() == "virus":

            myjson = '{\"virus\": \"\", \"country\": \"\", "date": \"\", \"cases\": \"\", \"deaths\": \"\"}'

        else:

            while "\\n" in myjson:
                myjson=myjson.replace("\\n"," ")
                myjson = myjson.strip()

            while "\\{" in myjson:
                myjson=myjson.replace("\\{","{")
                myjson = myjson.strip()

            while "\\}" in myjson:
                myjson=myjson.replace("\\}","}")
                myjson = myjson.strip()

            if "<" in myjson and ">" in myjson:
                myjson = myjson.split("<")[0].strip() + myjson.split(">")[1].strip()

            while "\t" in myjson:
                myjson=myjson.replace("\t"," ")
                myjson = myjson.strip()

            if "the answer is:" in myjson.lower() or "the answer is :" in myjson.lower():
                myjson = myjson.replace("the answer is:", " ")
                myjson = myjson.replace("The answer is:", " ")
                myjson = myjson.replace("the answer is :", " ")
                myjson = myjson.replace("The answer is :", " ")
                myjson = myjson.strip()

            if "sure!" in myjson.lower() :
                myjson = myjson.replace("sure!", " ")
                myjson = myjson.replace("Sure!", " ")
                myjson = myjson.replace("sure !", " ")
                myjson = myjson.replace("Sure !", " ")
                myjson = myjson.strip()

            if "the json object with the requested information" in myjson.lower() :
                myjson = myjson.replace("here is the json object with the requested information:", " ")
                myjson = myjson.replace("Here is the JSON object with the requested information:", " ")
                myjson = myjson.replace("here is the json object with the requested information :", " ")
                myjson = myjson.replace("Here's the JSON object with the requested information :", " ")
                myjson = myjson.replace("here's the json object with the requested information:", " ")
                myjson = myjson.replace("Here's the JSON object with the requested information:", " ")
                myjson = myjson.replace("here's the json object with the requested information :", " ")
                myjson = myjson.replace("Here's the JSON object with the requested information :", " ")
                myjson = myjson.strip()

            if "here is the extracted information in json format" in myjson.lower() :
                myjson = myjson.replace("here is the extracted information in json format:", " ")
                myjson = myjson.replace("Here is the extracted information in JSON format:", " ")
                myjson = myjson.replace("here is the extracted information in json format :", " ")
                myjson = myjson.replace("Here is the extracted information in JSON format :", " ")
                myjson = myjson.strip()

            if "here is the json object with the information you requested" in myjson.lower() :
                myjson = myjson.replace("here is the json object with the information you requested:", " ")
                myjson = myjson.replace("Here is the JSON object with the information you requested:", " ")
                myjson = myjson.replace("here is the json object with the information you requested :", " ")
                myjson = myjson.replace("Here is the JSON object with the information you requested :", " ")
                myjson = myjson.strip()

            if "here is the information extracted from the text, formatted as a json object with the requested keys" in myjson.lower() :
                myjson = myjson.replace("here is the information extracted from the text, formatted as a json object with the requested keys:", " ")
                myjson = myjson.replace("Here is the information extracted from the text, formatted as a JSON object with the requested keys:", " ")
                myjson = myjson.replace("here is the information extracted from the text, formatted as a json object with the requested keys :", " ")
                myjson = myjson.replace("Here is the information extracted from the text, formatted as a JSON object with the requested keys :", " ")
                myjson = myjson.strip()

            if "Explanation:" in myjson:
                myjson = myjson.split("Explanation:")[0]
                myjson = myjson.strip()

            if "Note:" in myjson:
                myjson = myjson.split("Note:")[0]
                myjson = myjson.strip()

            if "Note that the" in myjson:
                myjson = myjson.split("Note that the")[0]
                myjson = myjson.strip()

            if "Here's a brief explanation" in myjson:
                myjson = myjson.split("Here's a brief explanation")[0]
                myjson = myjson.strip()

            if "Here's a breakdown of the information" in myjson:
                myjson = myjson.split("Here's a breakdown of the information")[0]
                myjson = myjson.strip()

            if "in the following JSON object:" in myjson:
                myjson = myjson.split("in the following JSON object:")[1]
                myjson = myjson.strip()

            if "The information is presented in the format of a JSON object:" in myjson:
                myjson = myjson.split("The information is presented in the format of a JSON object:")[1]
                myjson = myjson.strip()

            if "The information is in the format you requested:" in myjson:
                myjson = myjson.split("The information is in the format you requested:")[1]
                myjson = myjson.strip()

            if "Here is the JSON object with the extracted information:" in myjson:
                myjson = myjson.split("Here is the JSON object with the extracted information:")[1]
                myjson = myjson.strip()

            if  "The information is not present" in myjson or "The information was found in the following places" in myjson:
                myjson = myjson.split("The information is not present")[0]
                myjson = myjson.split("The information was found in the following places")[0]
                myjson = myjson.strip()

            if "answer: {" in myjson.lower() or "answer:{" in myjson.lower():
                myjson = myjson.replace("Answer: {", "answer:{")
                myjson = myjson.replace("answer: {","answer:{")
                myjson = myjson.split("answer:")[1]
                myjson = myjson.strip()


            if "here is your solution:" in myjson.lower() or "here is your solution :" in myjson.lower() or "here is your solution." in myjson.lower() or "here is your solution ." in myjson.lower():
                myjson = myjson.replace("here is your solution:", " ")
                myjson = myjson.replace("here is your solution :", " ")
                myjson = myjson.replace("Here is your solution:", " ")
                myjson = myjson.replace("Here is your solution :", " ")
                myjson = myjson.replace("here is your solution.", " ")
                myjson = myjson.replace("here is your solution .", " ")
                myjson = myjson.replace("Here is your solution.", " ")
                myjson = myjson.replace("Here is your solution .", " ")
                myjson=myjson.strip()

            if "here is the solution:" in myjson.lower() or "here is the solution :" in myjson.lower():
                myjson = myjson.replace("here is the solution:", " ")
                myjson = myjson.replace("here is the solution :", " ")
                myjson = myjson.replace("Here is the solution:", " ")
                myjson = myjson.replace("Here is the solution :", " ")
                myjson=myjson.strip()

            if "here is the solution to your task:" in myjson.lower() or "here is the solution to your task :" in myjson.lower():
                myjson = myjson.replace("here is the solution to your task:", " ")
                myjson = myjson.replace("here is the solution to your task :", " ")
                myjson = myjson.replace("Here is the solution to your task:", " ")
                myjson = myjson.replace("Here is the solution to your task :", " ")
                myjson=myjson.strip()

            if "the solution to the problem is as follows:" in myjson.lower() or "the solution to the problem is as follows :" in myjson.lower():
                myjson = myjson.replace("the solution to the problem is as follows:", " ")
                myjson = myjson.replace("The solution to the problem is as follows:", " ")
                myjson = myjson.replace("the solution to the problem is as follows :", " ")
                myjson = myjson.replace("The solution to the problem is as follows :", " ")
                myjson=myjson.strip()

            if "the solution to the question is as follows:" in myjson.lower() or "the solution to the question is as follows :" in myjson.lower():
                myjson = myjson.replace("the solution to the question is as follows:", " ")
                myjson = myjson.replace("The solution to the question is as follows:", " ")
                myjson = myjson.replace("the solution to the question is as follows :", " ")
                myjson = myjson.replace("The solution to the question is as follows :", " ")
                myjson=myjson.strip()

            if "here is the json object:" in myjson.lower() or "here is the json object :" in myjson.lower():
                myjson = myjson.replace("Here is the JSON object:", " ")
                myjson = myjson.replace("Here is the JSON object :", " ")
                myjson = myjson.replace("here is the json object:", " ")
                myjson = myjson.replace("here is the json object :", " ")
                myjson=myjson.strip()

            if "here is the solution to the task:" in myjson.lower() or "here is the solution to the task :" in myjson.lower():
                myjson = myjson.replace("here is the solution to the task:", " ")
                myjson = myjson.replace("here is the solution to the task :", " ")
                myjson = myjson.replace("Here is the solution to the task:", " ")
                myjson = myjson.replace("Here is the solution to the task :", " ")
                myjson=myjson.strip()

            if "here is the json object for the given prompt:" in myjson.lower() or "here is the json object for the given prompt :" in myjson.lower():
                myjson = myjson.replace("Here is the JSON object for the given prompt:", " ")
                myjson = myjson.replace("Here is the JSON object for the given prompt :", " ")
                myjson = myjson.replace("here is the json object for the given prompt:", " ")
                myjson = myjson.replace("here is the json object for the given prompt :", " ")
                myjson=myjson.strip()

            if "JSON object with the following keys: virus, country, date, cases, deaths" in myjson:
                myjson = myjson.replace("JSON object with the following keys: virus, country, date, cases, deaths", " ")
                myjson = myjson.strip()

            if "The information you requested is below:" in myjson:
                myjson = myjson.replace("The information you requested is below:", " ")
                myjson = myjson.strip()

            if "Here is the information extracted from the text in the format requested:" in myjson:
                myjson = myjson.replace("Here is the information extracted from the text in the format requested:", " ")
                myjson = myjson.strip()

            if "Here's the information extracted from the text:" in myjson:
                myjson = myjson.replace("Here's the information extracted from the text:", " ")
                myjson = myjson.strip()

            if "JSON object with the following keys:" in myjson:
                myjson = myjson.replace("JSON object with the following keys:", " ")
                myjson = myjson.strip()

            if "*" in myjson:
                myjson = myjson.replace("*", "")
                myjson = myjson.strip()

            if "json{" or "json {" or "json:{" or "json: {" in myjson:
                myjson = myjson.replace("json{", "{")
                myjson = myjson.replace("json {", "{")
                myjson = myjson.replace("json:{", "{")
                myjson = myjson.replace("json: {", "{")
                myjson = myjson.strip()

            if "}." or "} ." in myjson:
                myjson = myjson.replace("}.", "}")
                myjson = myjson.replace("} .", "}")
                myjson = myjson.strip()

            commamissing = False
            if ("," in myjson) == False:
                commamissing = True

            for i, x in enumerate(attributes):

                insensitive_json = re.compile(re.escape(x), re.IGNORECASE)
                myjson = insensitive_json.sub(x, myjson)

                while "  " in myjson:
                    myjson = myjson.replace('  ', ' ')

                myjson = myjson.strip()
                if len(myjson)<=0:
                    break

                if ('\'' + x + '\'') in myjson:
                    myjson = myjson.replace('\'', '\"')

                if commamissing == True and i != 0:
                    myjson = myjson.replace(x, '\" , '+x)

                if (('\"' + x + '\"') in myjson) == False:
                    myjson = myjson.replace((x+":"), '\"' + x + '\":')
                    myjson = myjson.replace((x + " :"), '\"' + x + '\" :')

                if len(myjson)<=0:
                    break

                if (myjson.strip()[0] == "{") == False:
                    myjson = "{ "+myjson.strip()

                if (myjson.strip()[len(myjson.strip())-1] == "}") == False:
                    myjson = myjson.strip()+"\" }"

                myjson = myjson.replace('\": ','\": \"')
                myjson = myjson.replace(', \"', '\", \"')

                myjson = myjson.replace('{', '{\"')
                myjson = myjson.replace('}', '\"}')

                while "  " in myjson:
                    myjson = myjson.replace('  ', ' ')

                while '\"\"' in myjson:
                    myjson = myjson.replace('\"\"', '\"')

                while '\" \"' in myjson:
                    myjson = myjson.replace('\" \"', '\"')

                myjson = myjson.replace('\", \"}', '\"}')
                myjson = myjson.replace('\",\"}', '\"}')

            #

            try:

                if myjson:
                    jsonloaded = json.loads(myjson)
                else:
                    jsonloaded =""

                if jsonloaded:
                    dictkeyslist = list(jsonloaded.keys())

                    notExpectedAttributes = list(set(dictkeyslist) - set(attributes))
                    if notExpectedAttributes:
                        for notExp in notExpectedAttributes:
                            del jsonloaded[notExp]
                        myjson = json.dumps(jsonloaded)

            except ValueError as e:

                if "{\"virus" in myjson:
                    res = re.findall(r'\{"virus.*?\}', myjson)
                    if res:
                        myjson = res[0]

                elif "{ \"virus" in myjson:
                    res = re.findall(r'\{ "virus.*?\}', myjson)
                    if res:
                        myjson = res[0]

                try:

                    if myjson:
                        jsonloaded = json.loads(myjson)
                    else:
                        jsonloaded = ""

                    if jsonloaded:
                        dictkeyslist = list(jsonloaded.keys())

                        notExpectedAttributes = list(set(dictkeyslist) - set(attributes))
                        if notExpectedAttributes:
                            for notExp in notExpectedAttributes:
                                del jsonloaded[notExp]
                            myjson = json.dumps(jsonloaded)

                except ValueError as e2:

                    try:

                        mjs = ""
                        for iatt, attrt in enumerate(attributes):

                            findseparatorAttr = ""
                            if "\""+attrt+"\":" in myjson:
                                findseparatorAttr = "\""+attrt+"\":"
                            elif "\""+attrt+"\"" in myjson:
                                findseparatorAttr = "\""+attrt+"\""
                            elif attrt in myjson:
                                findseparatorAttr = attrt

                            ccc = ""
                            if len(findseparatorAttr)>0:
                                aaa=myjson.split(findseparatorAttr)

                                if "\"," in aaa[1]:
                                    bbb = aaa[1].split("\",")
                                elif "\" \"" in aaa[1]:
                                    bbb = aaa[1].split("\" \"")
                                elif "}" in aaa[1]:  #last one
                                    bbb = aaa[1].split("}")
                                else:
                                    if iatt < (len(attributes)-1):
                                        nextiatt = iatt +1
                                    else:
                                        nextiatt = iatt
                                    bbb = aaa[1].split(attributes[nextiatt])

                                ccc = bbb[0]
                                ccc = ccc.replace("\"","").strip()
                                #if "\'" in ccc and not("\\'" in ccc):
                                #    ccc = ccc.replace("\'", "\\'").strip()
                                if "\\" in ccc:
                                    ccc = ccc.replace("\\", "").strip()

                            if iatt == 0:
                                mjs = mjs + "{ " + "\""+attrt+"\":" + " \""+ccc+"\""
                            else:
                                mjs = mjs + ", " + "\"" + attrt + "\":" + " \"" + ccc + "\""

                            if iatt == (len(attributes)-1):
                                mjs = mjs + "}"

                        if mjs:
                            myjson = mjs
                            jsonloaded = json.loads(myjson)
                        else:
                            jsonloaded = ""

                        if jsonloaded:
                            dictkeyslist = list(jsonloaded.keys())

                            notExpectedAttributes = list(set(dictkeyslist) - set(attributes))
                            if notExpectedAttributes:
                                for notExp in notExpectedAttributes:
                                    del jsonloaded[notExp]
                                myjson = json.dumps(jsonloaded)

                    except ValueError as e3:

                        print("debug - here is not contemplated json reconstruction errors")

        return myjson.strip()

    return myjson.strip()

def check_json_attribute(p, attr):
    doc = json.loads(json.dumps(p))
    try:
        doc.get(attr) # we don't care if the value exists. Only that 'get()' is accessible
        return True
    except AttributeError:
        return False


def run_epidemicExtractor(input_specify,model_name,df,DATE_IMPUTATION,MAX_TOKENS_PROMPT,TOKENS_TOLERANCE,service_provider,USE_CACHE,load_map_query_input_output,cache_name,JSON_RECONSTRUCT,temperature_value,InContextExamples):


    print("\nNumber of articles = "+str(df.shape[0]))

    start = time.time()
    output_logs_filename = str(input_specify) + "/OutputAnnotatedTexts-" + model_name + ".log"
    file_LOG = open(output_logs_filename, "w")
    file_LOG.write("LOGGING MODEL " + str(model_name) + "\n\n")
    file_LOG.write("Number of articles = "+str(df.shape[0]))

    for idxrowx, row in tqdm(df.iterrows()):
        
        if DATE_IMPUTATION == True:

            fileid = df['fileid'][idxrowx].strip().lower()
            if "promed" in fileid:
                imputedDate_str = fileid.split("/")[1].split("_")[0]
                imputedDate_str_date = datetime.datetime(int(imputedDate_str.split("-")[0]),
                                                         int(imputedDate_str.split("-")[1]),
                                                         int(imputedDate_str.split("-")[2]))
                if imputedDate_str_date:                    
                    df.at[idxrowx, 'date_cases_IMPUTED'] = imputedDate_str
          
        ttext = df['texts'][idxrowx]

        ttext = ttext.strip()

        max_size_tokens_summary = int((MAX_TOKENS_PROMPT - TOKENS_TOLERANCE))
        min_size_tokens_summary = int(0.5 * (MAX_TOKENS_PROMPT))

        myPromt = f"""
                Summarize the text below, delimited by triple \
                backticks, and focusing on any aspects \
                that are relevant to the virus outbreak, the place where the virus outbreak happened, \
                when the virus outbreak happened, and the number of cases occurred. \
                Do not invent. \
                The summary should be smaller than {max_size_tokens_summary} tokens.\
                The summary should also be bigger than {min_size_tokens_summary} tokens. \
                Write no explanations or notes . \
                Text: 
                """

        myDelimiter = "```"

        nt_Text = token_counter(ttext, model_name)
        nt_PromtonlyNoText = token_counter((myPromt + myDelimiter + myDelimiter), model_name)

        encod = encoding_getter(model_name)
        
        if ttext and ((nt_Text + nt_PromtonlyNoText) > (MAX_TOKENS_PROMPT - TOKENS_TOLERANCE)):

            separators = [".\\n"]
            
            text_splitter = TokenTextSplitter(
                # separators=separators,
                encoding_name=encod.name,
                chunk_size=((MAX_TOKENS_PROMPT - TOKENS_TOLERANCE) - nt_PromtonlyNoText),
                chunk_overlap=50,
                length_function=len,
                add_start_index=True,
            )
            
            texts = text_splitter.create_documents([ttext.strip()])
       
            totTxT = ""

            for idx, x in enumerate(texts):
                # print(idx, x)

                xTxt = x.page_content

                while '\\n' in xTxt:
                    xTxt = xTxt.replace('\\n', ' ')

                while '  ' in xTxt:
                    xTxt = xTxt.replace('  ', ' ')

                xTxt = xTxt.strip()

                if len(xTxt) <= 1:  # message too short, continue
                    continue

                
                max_size_tokens_summary = int((MAX_TOKENS_PROMPT - TOKENS_TOLERANCE) / len(texts))
                min_size_tokens_summary = int(0.5 * (MAX_TOKENS_PROMPT) / len(texts))

                ORI_max_size_tokens_summary = max_size_tokens_summary
                ORI_min_size_tokens_summary = min_size_tokens_summary

                orixTxT = xTxt

                myPromt = f"""
                                Summarize the text below, delimited by triple \
                                backticks, and focusing on any aspects \
                                that are relevant to the virus outbreak, the place where the virus outbreak happened, \
                                when the virus outbreak happened, and the number of cases occurred. \
                                Do not invent. \
                                The summary should be smaller than {max_size_tokens_summary} tokens.\
                                The summary should also be bigger than {min_size_tokens_summary} tokens. \
                                Write no explanations or notes . \
                                Text: 
                                """

                myDelimiter = "```"

                nt_Text = token_counter(xTxt, model_name)
                nt_PromtonlyNoText = token_counter((myPromt + myDelimiter + myDelimiter), model_name)

                epss = 5
                while ((nt_Text + nt_PromtonlyNoText) > (ORI_max_size_tokens_summary + epss)):

                    # this is a bit reduntant, but leave it
                    myPromt = f"""
                                    Summarize the text below, delimited by triple \
                                    backticks, and focusing on any aspects \
                                    that are relevant to the virus outbreak, the place where the virus outbreak happened, \
                                    when the virus outbreak happened, and the number of cases occurred. \
                                    Do not invent. \
                                    The summary should be smaller than {max_size_tokens_summary} tokens.\
                                    The summary should also be bigger than {min_size_tokens_summary} tokens. \
                                    Write no explanations or notes . \
                                    Text: 
                                    """
                    
                    myDelimiter = "```"

                    if service_provider == "openai":

                        if USE_CACHE:
                            xTxt = call_model_with_caching(input_text=xTxt, prompt=myPromt, model=model_name,
                                                           temperature=temperature_value, delimiter=myDelimiter,
                                                           InContextExamples=[],
                                                           handler=api_call_openai,
                                                           map_query_input_output=load_map_query_input_output,
                                                           cache_fp=cache_name,
                                                           verbose=True)
                        else:
                            xTxt = call_model(input_text=xTxt, prompt=myPromt, model=model_name,
                                              temperature=temperature_value, delimiter=myDelimiter,
                                              InContextExamples=[],
                                              handler=api_call_openai,
                                              verbose=True)
                    elif service_provider == "gptjrc":
                        if USE_CACHE:
                            xTxt = call_model_with_caching(input_text=xTxt, prompt=myPromt, model=model_name,
                                                           temperature=temperature_value, delimiter=myDelimiter,
                                                           InContextExamples=[],
                                                           handler=api_call_gptjrc,
                                                           map_query_input_output=load_map_query_input_output,
                                                           cache_fp=cache_name,
                                                           verbose=True)
                        else:
                            xTxt = call_model(input_text=xTxt, prompt=myPromt, model=model_name,
                                              temperature=temperature_value, delimiter=myDelimiter,
                                              InContextExamples=[],
                                              handler=api_call_gptjrc,
                                              verbose=True)
                    
                    if xTxt:
                        xTxt = xTxt.strip()

                        myPromt = f"""
                                    Summarize the text below, delimited by triple \
                                    backticks, and focusing on any aspects \
                                    that are relevant to the virus outbreak, the place where the virus outbreak happened, \
                                    when the virus outbreak happened, and the number of cases occurred. \
                                    Do not invent. \
                                    The summary should be smaller than {max_size_tokens_summary} tokens.\
                                    The summary should also be bigger than {min_size_tokens_summary} tokens. \
                                    Write no explanations or notes . \
                                    Text: 
                                    """

                        nt_Text = token_counter(xTxt, model_name)
                        nt_PromtonlyNoText = token_counter((myPromt + myDelimiter + myDelimiter), model_name)

                        if ((nt_Text + nt_PromtonlyNoText) > (ORI_max_size_tokens_summary + epss)):
                            # if (len(xTxt) > ORI_max_size_chars_summary):
                            max_size_tokens_summary = int(max_size_tokens_summary / 2)
                            min_size_tokens_summary = int(min_size_tokens_summary / 2)
                            print(
                                "\nSomething strange, the answer is still too big. The tokens length of the message is " + str(
                                    nt_Text + nt_PromtonlyNoText) + " which is bigger then " + str(
                                    ORI_max_size_tokens_summary) + ". Retry by halving the requested tokens length of the message to  " + str(
                                    max_size_tokens_summary))

                            

                        if min_size_tokens_summary < 100:  # it was not able to summarize the message...just drop some part then!
                            
                            text_splitter_INTERMEDIATE = TokenTextSplitter(
                                # separators=separators,
                                encoding_name=encod.name,
                                chunk_size=((ORI_max_size_tokens_summary) - nt_PromtonlyNoText),
                                chunk_overlap=min(50, int(((ORI_max_size_tokens_summary) - nt_PromtonlyNoText) / 2)),
                                length_function=len,
                                add_start_index=True,
                            )
                            

                            texts_INTERMEDIATE = text_splitter_INTERMEDIATE.create_documents([orixTxT.strip()])
                            
                            xTxt = texts_INTERMEDIATE[0].page_content  # take only the first one

                            while '\\n' in xTxt:
                                xTxt = xTxt.replace('\\n', ' ')

                            while '  ' in xTxt:
                                xTxt = xTxt.replace('  ', ' ')

                            xTxt = xTxt.strip()

                            nt_Text = token_counter(xTxt, model_name)

                    else:
                        xTxt = ""
                        nt_Text = 0  # 20231218 added consose otherwise don't exit from while

                if xTxt and xTxt.strip() != '':
                    totTxT = totTxT + " " + xTxt
                    totTxT = totTxT.strip()

            
            df.at[idxrowx, 'texts'] = totTxT.strip()

        else:

            while '\\n' in ttext:
                ttext = ttext.replace('\\n', ' ')

            while '  ' in ttext:
                ttext = ttext.replace('  ', ' ')

            df.at[idxrowx, 'texts'] = ttext.strip()

    ###

    
    df['texts_length'] = df['texts'].str.strip().str.len()

    
    myPromt = f"""From the text below, delimited by triple quotes, extract the following items:\
            1 - The name of the virus that has caused the outbreak. \
            2 - The name of the country where this virus outbreak occurred, if present. Do not invent. \        
            3 - The date when this virus outbreak occurred, if present. Do not invent. \
                Show the date in the format YYYY-mm-dd. \        
            4 - The number of cases derived exclusively from the virus outbreak mentioned in the text, if present. Do not invent. \
            5 - The number of deaths caused exclusively by the virus outbreak mentioned in the text, if present. Do not invent. \
            Format your response as a JSON object with the following keys: virus, country, date, cases, deaths. \ 
        	If the information is not present, do not invent and use "None" as the value. \
        	Write no explanations or notes . \
            Text: """

    myDelimiter = "```"

    startPROMPT = time.time()

    if service_provider == "openai":
        if USE_CACHE:
            lambda_model = partial(call_model_with_caching, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter,
                                   InContextExamples=InContextExamples, handler=api_call_openai,
                                   map_query_input_output=load_map_query_input_output, cache_fp=cache_name,
                                   verbose=True)
        else:
            lambda_model = partial(call_model, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter,
                                   InContextExamples=InContextExamples, handler=api_call_openai,
                                   verbose=True)
    elif service_provider == "gptjrc":
        if USE_CACHE:
            lambda_model = partial(call_model_with_caching, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter,
                                   InContextExamples=InContextExamples, handler=api_call_gptjrc,
                                   map_query_input_output=load_map_query_input_output, cache_fp=cache_name,
                                   verbose=True)
        else:
            lambda_model = partial(call_model, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter,
                                   InContextExamples=InContextExamples, handler=api_call_gptjrc,
                                   verbose=True)

    if lambda_model:
        df["json_original"] = df["texts"].progress_apply(lambda_model)

    endPROMPT = time.time()
    hours, rem = divmod(endPROMPT - startPROMPT, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        "Computational Time for EPIDEMIC PROMPT EXTRACTION... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes),
                                                                                               seconds))
    file_LOG.write("\nComputational Time for EPIDEMIC PROMPT EXTRACTION... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),
                                                                                                            int(minutes),
                                                                                                            seconds) + "\n")

    ####

    if JSON_RECONSTRUCT:

        print("\njson reconstruction")

        df['json_extracted'] = df['json_original'].copy()

        while df['json_extracted'].str.contains(r'\n').any():
            df['json_extracted'] = df['json_extracted'].replace(r'\n', ' ', regex=True)
        while df['json_extracted'].str.contains('  ').any():
            df['json_extracted'] = df['json_extracted'].str.replace("  ", ' ')

        attributes = ['virus', 'country', 'date', 'cases', 'deaths']
        df["json_extracted"] = df["json_extracted"].apply(json_reconstruction, attributes=attributes)

        # if False in df["json_extracted"].apply(is_json).str.contains(False).any()
        # df["json_extracted"].apply(is_json).isin([False])
        areJson = df["json_extracted"].apply(is_json)
        if False in areJson.unique():
            for idxr, rr in df.iterrows():
                # for idxr in range(df['json_extracted'].shape[0]):
                if areJson[idxr] == False:
                    print("PROBLEM WITH JSON AT INDEX " + str(idxr) + ":\n" + df["json_extracted"][idxr])
                    replacement_empty_myjson = '{\"virus\": \"\", \"country\": \"\", "date": \"\", \"cases\": \"\", \"deaths\": \"\"}'
                    df.at[idxr, "json_extracted"] = replacement_empty_myjson
                    print(" ...... Then replacing it with empty JSON --> " + df["json_extracted"][idxr])

        try:
            # df['virus'] = df.apply(lambda x: json.loads(x['json_extracted'])['virus'], axis=1)

            df_extract = df.apply(lambda x: pd.Series(
                json.loads(x['json_extracted']).values(),
                index=json.loads(x['json_extracted']).keys()), axis=1)

            df_extract.rename(columns={'virus': 'virus_extracted'}, inplace=True)
            df_extract.rename(columns={'country': 'country_extracted'}, inplace=True)
            df_extract.rename(columns={'date': 'date_extracted'}, inplace=True)
            df_extract.rename(columns={'cases': 'cases_extracted'}, inplace=True)
            df_extract.rename(columns={'deaths': 'deaths_extracted'}, inplace=True)

            df = pd.merge(df, df_extract, left_index=True, right_index=True)

        except Exception as err:

            logging.error(
                f'FAILED to extract json results\n\tError: {err}\nLeaving it as a single column then and not decompressing! Have a check...')

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    df.fillna(value=np.nan, inplace=True)
    df.replace(to_replace=[None], value=np.nan, inplace=True)
    df.replace(to_replace=['None.'], value=np.nan, inplace=True)
    df.replace(to_replace=['None'], value=np.nan, inplace=True)
    df.replace(to_replace=['null'], value=np.nan, inplace=True)
    df.replace(to_replace=[''], value=np.nan, inplace=True)
    df.replace(to_replace=[' '], value=np.nan, inplace=True)

    if 'cases_extracted' in df.columns.values:
        try:
            df['cases_extracted'] = df['cases_extracted'].astype(float)
        except Exception as err:
            logging.warning(
                f'FAILED to convert cases_extracted column to float\n\tError: {err}\nLeaving it as a text column then')
            file_LOG.write(
                "\nFAILED to convert cases_extracted column to float\n\tError: {err}\nLeaving it as a text column then" + "\n")

    if 'deaths_extracted' in df.columns.values:
        try:
            df['deaths_extracted'] = df['deaths_extracted'].astype(float)
        except Exception as err:
            logging.warning(
                f'FAILED to convert deaths_extracted column to float\n\tError: {err}\nLeaving it as a text column then')
            file_LOG.write(
                "\nFAILED to convert deaths_extracted column to float\n\tError: {err}\nLeaving it as a text column then" + "\n")

    ###

    print("\n")
    # print(df)

    #
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Overall Computational Time... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    file_LOG.write(
        "\nOverall Computational Time... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds) + "\n")
    #

    return df





if __name__ == "__main__":

    # MAX_TOKENS_PROMPT depends on the LLM selected
    # MAX_TOKENS_PROMPT = 1024  # for Pythia
    # MAX_TOKENS_PROMPT = 2048 #for GPT3
    #MAX_TOKENS_PROMPT = 2048  #mpt-30b-chat --> it should be 8192 but does not work, in reality Stefano said me that it is working at 2048
                            # same for llama-2-13b-chat, should be 4096 but works at 2048
    MAX_TOKENS_PROMPT = 4096 # for GPT3.5 - llama-3-70b-instruct - mistral-7b-openorca - zephyr-7b-beta
    #MAX_TOKENS_PROMPT = 16384 #gpt-35-turbo-16k
    #MAX_TOKENS_PROMPT = 32768  #gpt-4-32k
    #
    # https://platform.openai.com/docs/models/overview
    # Pythia12B max 1024 tokens -- around 4096 chars
    # GPT3 max 2049 tokens -- around 8192 chars
    # GPT3.5 max 4096 tokens -- around 16384 chars

    TOKENS_TOLERANCE = 800   #800 was good for llama-3-70b-instruct  #5000 tokens are for the InContext examples

    USE_CACHE = True # True #False

    DATE_IMPUTATION = True #False

    JSON_RECONSTRUCT = True #True False

    #service_provider = "openai"
    #model_name = "gpt-3.5-turbo-16k"
    # dglc available models: 'OA_SFT_Pythia_12B', 'JRC_RHLF_13B', 'OA_GPT3.5', 'OA_GPT3'
    # model_name = "gpt-3.5-turbo"  #OpenAI name
    # model_name = 'JRC_RHLF_13B'
    #model_name = "OA_SFT_Pythia_12B"   #EleutherAI-pythia-12b
    # model_name = "OA_GPT3"
    # model_name = "GPT@JRC_4"
    #
    #
    service_provider = "gptjrc"
    #model_name = "gpt-35-turbo-16k"  #June 2023. Context length of 16384 tokens.- around 48000
    #model_name = "gpt-4-32k"  #GPT-4 with a context length of 32,768 tokens -  around 100000
    #model_name = "llama-2-13b-chat" #"context_length": 2048 - around 6000
    #model_name = "llama-3-70b-instruct" #Context length of 4096 tokens. - around 14500
    #model_name = "mpt-30b-chat" #"context_length": 2048 - around 6000
    #model_name="mistral-7b-openorca" #Context length of 4096 tokens
    model_name = "zephyr-7b-beta"  #Context length of 4096 tokens

    # temperature: temperature_value (0: precise, 1: creative)
    temperature_value = 0.01  # 0.1

    ##################################################################################################

    InContextExamples = []
    
    if InContextExamples:
        for row in InContextExamples:
            for col in row:
                nt = token_counter(col, model_name)
                # print("\nNumber of Tokens in the example = " + str(nt))
                ntotExamplesTokens = ntotExamplesTokens + nt
        #
        print("\nNumber of Tokens of the all examples in the json extraction = " + str(ntotExamplesTokens))
        TOKENS_TOLERANCE=TOKENS_TOLERANCE+ntotExamplesTokens
        print("\nUpdated TOKENS_TOLERANCE to " + str(TOKENS_TOLERANCE))

    ##################################################################################################




    # OpenAI ChatGPT API
    if service_provider == "openai":
        MyOpenAPIKey = ""
        fkeyname="OpenAI-DigLifeAccount-APItoken.key"
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

    tqdm.pandas()

    ###########################################################################

    cache_name = ""
    load_map_query_input_output = {}
    if USE_CACHE:
        # cache_prefix_fp: prefix of the file to which write content of cache after each call
        cache_prefix_fp = "LLMQUERYDEPLOY"
        cache_name = cache_prefix_fp + "___" + "__".join([service_provider, model_name, str(temperature_value)]).replace(
            " ", "_") + ".json"

        if os.path.exists(cache_name):

            with open(cache_name) as f:
                load_map_query_input_output = json.load(f)

        else:
            load_map_query_input_output = {}



    ###########################################################################
    ### INPUT DATAFRAME : ###

    input_specify = Path(__file__).parent.resolve() / Path("/eos/jeodpp/data/projects/ETOHA/DATA/etohaSurveillanceScraper/corpus_processed/SUMMARIZED/")
    
    corpus = PickledCorpusReader(root=input_specify)

    fileids = corpus.get_fileids()

    texts = corpus.docs(fileids)

    df = pd.DataFrame()
    df['fileid'] = fileids
    df['texts'] = list(zip(texts))

    # #TOTAL_ProMED
    # #who_dons
    # filter = df['fileid'].str.contains('who_dons')
    # df = df[filter]

    if DATE_IMPUTATION == True:
        df['date_cases_IMPUTED'] = pd.Series(dtype='str')

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

    
    df = run_epidemicExtractor(input_specify, model_name, df, DATE_IMPUTATION, MAX_TOKENS_PROMPT, TOKENS_TOLERANCE,service_provider, USE_CACHE, load_map_query_input_output, cache_name, JSON_RECONSTRUCT, temperature_value,InContextExamples)

    
    # save input texts
    output_texts_filename = str(input_specify) + "/OutputAnnotatedTexts-" + model_name + ".csv"
    df.to_csv(output_texts_filename, sep=',', header=True, index=True, encoding='utf-8')

    print("\nEnd Computations")



