import os
import sys

import openai
import json
import time
from tqdm import tqdm

import logging

from functools import partial
import pandas as pd

import tiktoken




""" 
query LLM API end point on list of text, seamlessly

features:
- build in retry in case of error
- cache the results in case of crash
- call LLM with a lambda or as regular function call

supported API:
- OpenAI
- GPT@JRC
- F7 (DigLife)

"""




def encoding_getter(encoding_type: str):
    """
    Returns the appropriate encoding based on the given encoding type (either an encoding string or a model name).

    tiktoken supports three encodings used by OpenAI models:

    Encoding name	OpenAI models
    cl100k_base	gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    p50k_base	Codex models, text-davinci-002, text-davinci-003
    r50k_base (or gpt2)	GPT-3 models like davinci

    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    """
    if "k_base" in encoding_type:
        return tiktoken.get_encoding(encoding_type)
    else:
        try:
            my_enc = tiktoken.encoding_for_model(encoding_type)
            return my_enc
        except Exception as err:
            my_enc = tiktoken.get_encoding("cl100k_base")   #default for gpt-4, gpt-3.5-turbo
            return my_enc

def tokenizer(string: str, encoding_type: str) -> list:
    """
    Returns the tokens in a text string using the specified encoding.
    """
    encoding = encoding_getter(encoding_type)
    tokens = encoding.encode(string)
    return tokens

def token_counter(string: str, encoding_type: str) -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.
    """
    num_tokens = len(tokenizer(string, encoding_type))
    return num_tokens


# ### OPENAI API

def setup_openai(org=None, key=None):
    if org is not None:
        openai.organization = org

    if key is not None:
        openai.api_key = key
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    #
    print(model_list_openai())

def api_call_openai(prompt: str, input_text: str, model: str, temperature: int, timeout_retry: int=5, delimiter: str = "```", InContextExamples: list[[str]] = [], debug=False):
    """ call openai API, with a retry in case of RateLimitError """

    if not(prompt) or prompt.strip=="" or not(input_text) or input_text.strip=="":
        logging.warning("No text or promt supplied! Skypping it!")
        return None

    if delimiter and len(delimiter)>0:
        input_text = delimiter + input_text + delimiter

    response = None

    myMessages = []
    if InContextExamples:
        for row in InContextExamples:
            myMessages.append({"role": "system", "content": prompt})
            for indCol, colVal in enumerate(row):
                if indCol == 0:
                    if delimiter and len(delimiter) > 0:
                        myMessages.append({"role": "user", "content": (delimiter + colVal + delimiter)})
                    else:
                        myMessages.append({"role": "user", "content": colVal})
                elif indCol == 1:
                    myMessages.append({"role": "assistant", "content": colVal})

    myMessages.append({"role": "system", "content": prompt})
    myMessages.append({'role': 'user', 'content': input_text})

    max_retries = 50
    iteration = 1
    while response is None and max_retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                
                messages=myMessages,
                temperature=temperature,
                
            )
        except openai.error.RateLimitError as e:
            response = None
            max_retries = max_retries - 1
            print(e)
            nt = token_counter((prompt + input_text), model)
            print("Model "+str(model)+" - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration * timeout_retry), "s")
            print(time.sleep(int(iteration * timeout_retry)))
            iteration = iteration + 1
        except Exception as err:
            response = None
            max_retries = max_retries - 1
            print(err)
            nt = token_counter((prompt + input_text), model)
            print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration*timeout_retry), "s")
            print(time.sleep(int(iteration*timeout_retry)))
            iteration = iteration + 1

    if (response == None) and (max_retries <= 0):
        print("\n")
        print(prompt + input_text)
        print("\n")
        print("\nTried many times and did not succeed, there is something strange. Check the problem...exiting now\n")
        sys.exit()

    return response

def model_list_openai():
    return openai.Model.list()


### GPT@JRC API

def setup_gptjrc(token=None):
    if token is None:
        token=os.getenv("GPTJRC_TOKEN")
    openai.organization = ""
    openai.api_key = token
    openai.api_base = "https://api-gpt.jrc.ec.europa.eu/v1"
    #
    print(model_list_gptjrc())


def api_call_gptjrc(prompt: str, input_text: str, model: str, temperature: int, timeout_retry: int=5, delimiter: str = "```", InContextExamples: list[[str]] = [], debug=False):


    if not (prompt) or prompt.strip=="" or not(input_text) or input_text.strip=="":
        logging.warning("No text or promt supplied! Skypping it!")
        return None

    if delimiter and len(delimiter)>0:
        input_text = delimiter + input_text + delimiter

    response = None

    myMessages = []
    if InContextExamples:
        for row in InContextExamples:
            myMessages.append({"role": "system", "content": prompt})
            for indCol, colVal in enumerate(row):
                if indCol == 0:
                    if delimiter and len(delimiter) > 0:
                        myMessages.append({"role": "user", "content": (delimiter + colVal + delimiter)})
                    else:
                        myMessages.append({"role": "user", "content": colVal})
                elif indCol == 1:
                    myMessages.append({"role": "assistant", "content": colVal})

    myMessages.append({"role": "system", "content": prompt})
    myMessages.append({'role': 'user', 'content': input_text})

    max_retries = 50
    iteration = 1
    while response is None and max_retries>0:
        try:
            
            response = openai.ChatCompletion.create(
                headers={"Authorization": "Bearer "+openai.api_key},
                model=model,
                
                messages=myMessages,
                temperature=temperature,
                
            )

        except openai.error.RateLimitError as e:
            response = None
            max_retries = max_retries - 1
            print(e)
            nt = token_counter((prompt + input_text), model)
            print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration*timeout_retry), "s")
            print(time.sleep(int(iteration*timeout_retry)))
            iteration = iteration + 1
            print("\npromt:")
            print(prompt)
            print("\ninput_text:")
            print(input_text)
        except Exception as err:
            response = None
            max_retries = max_retries - 1
            print(err)
            nt = token_counter((prompt + input_text), model)
            print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration * timeout_retry), "s")
            print(time.sleep(int(iteration * timeout_retry)))
            iteration = iteration + 1
            print("\npromt:")
            print(prompt)
            print("\ninput_text:")
            print(input_text)
            if max_retries == 45 or max_retries == 40 or max_retries == 35 or max_retries == 30 or max_retries == 25 or max_retries == 20 or max_retries == 15 or max_retries == 10 or max_retries == 5:
                input_text = input_text[0:-1000]
                input_text = input_text + delimiter
                myMessages = []
                myMessages.append({"role": "system", "content": prompt})
                myMessages.append({'role': 'user', 'content': input_text})


    if (response == None) and (max_retries <= 0):
        print("\n")
        print(prompt + input_text)
        print("\n")
        print("\nTried many times and did not succeed, there is something strange. Check the problem...exiting now\n")
        sys.exit()

    return response

def model_list_gptjrc():
    return openai.Model.list()


### 

def clean_gpt_out(output_text :str):

    if "From the text below, delimited by triple quotes, extract the following items: 1 - The name of the virus that has caused the outbreak" in output_text:
        print("debug")

    if "<|assistant|>" in output_text:
        output_text = output_text.split("<|assistant|>")[0].strip()
    if "<|prompt|>" in output_text:
        output_text = output_text.split("<|prompt|>")[0].strip()
    if "<|prompter|>" in output_text:
        output_text = output_text.split("<|prompter|>")[0].strip()
    if "<|answer|>" in output_text:
        output_text = output_text.split("<|answer|>")[0].strip()
    if "<|im_end|>" in output_text:
        output_text = output_text.split("<|im_end|>")[0].strip()
    if "<|endofextract|>" in output_text:
        output_text = output_text.split("<|endofextract|>")[0].strip()
    if "<br>" in output_text:
        output_text = output_text.split("<br>")[0].strip()

    if "<|/assistant|>" in output_text:
        output_text = output_text.split("<|/assistant|>")[0].strip()
    if "<|/prompt|>" in output_text:
        output_text = output_text.split("<|/prompt|>")[0].strip()
    if "<|/prompter|>" in output_text:
        output_text = output_text.split("<|/prompter|>")[0].strip()
    if "<|/answer|>" in output_text:
        output_text = output_text.split("<|/answer|>")[0].strip()
    if "<|/im_end|>" in output_text:
        output_text = output_text.split("<|/im_end|>")[0].strip()
    if "<|/endofextract|>" in output_text:
        output_text = output_text.split("<|/endofextract|>")[0].strip()
    if "</br>" in output_text:
        output_text = output_text.split("</br>")[0].strip()

    if "</|assistant|>" in output_text:
        output_text = output_text.split("</|assistant|>")[0].strip()
    if "</|prompt|>" in output_text:
        output_text = output_text.split("</|prompt|>")[0].strip()
    if "</|prompter|>" in output_text:
        output_text = output_text.split("</|prompter|>")[0].strip()
    if "</|answer|>" in output_text:
        output_text = output_text.split("</|answer|>")[0].strip()
    if "</|im_end|>" in output_text:
        output_text = output_text.split("</|im_end|>")[0].strip()
    if "</|endofextract|>" in output_text:
        output_text = output_text.split("</|endofextract|>")[0].strip()

    while "```" in output_text:
        output_text = output_text.replace("```", " ")

    while "  " in output_text:
        output_text = output_text.replace("  ", " ")

    return output_text



### CALLING MODELS


def call_model_with_caching(input_text: str, prompt: str, model: str, temperature: int, handler,
                            map_query_input_output: dict, cache_fp: str, timeout_retry: int =5, delimiter: str = "```", InContextExamples: list[[str]] = [], verbose: bool = True):
    """ call openai's API but take care of caching of results
    input_text: input text
    prompt: prompt
    model: model name (as parameter of the query)
    temperature: temperature (0: precise, 1: creative)
    handler: delegate function that will make the call (not necessarily only OpenAI, could be any one)
    map_query_input_output: cache dict containing already processed data
    cache_fp: file to which write content of cache after each call
    """

    if not(input_text) or input_text.strip=="" or not(prompt) or prompt.strip=="":
        logging.warning("No text or promt supplied! Skypping it!")
        return None

    # try to read cache

    if map_query_input_output is not None:
        key = model + "__" + str(temperature) + "__" + prompt

        if key in map_query_input_output:
            if input_text in map_query_input_output[key]:
                output = map_query_input_output[key][input_text]
                

                if verbose:
                    print("RETRIEVED CACHED RESULT FOR:\n", prompt, "\n", delimiter, input_text, delimiter, "\n=>\n", output, "\n")

                return output

    #  call

    response = None

    try:
        response = handler(prompt, input_text, model, temperature, timeout_retry, delimiter, InContextExamples)
    except Exception as err:
        logging.error(f'FAILED WITH PROMPT: \'{prompt}\' \nLEN_TEXT: {len(input_text)}, \nTEXT: {(input_text)}, \nMODEL: {model}; \n\tError: {err}')
    

    if response:
        if isinstance(response, str):
            output_text = response
        else:
            output_text = response['choices'][0]['message']['content']

        # write to cache

        if map_query_input_output is not None:
            if not key in map_query_input_output:
                map_query_input_output[key] = {}

            if output_text:
                if output_text != "":
                    map_query_input_output[key][input_text] = output_text

                    with open(cache_fp, "w") as f:
                        json.dump(map_query_input_output, f)

        if verbose:
            print("API CALL REPLY FOR:\n", prompt, "\n", delimiter, input_text, delimiter, "\n=>\n", output_text, "\n")

        return output_text

    else:
        return None


def call_model(input_text: str, prompt: str, model: str, temperature: int, handler, timeout_retry: int =5, delimiter: str = "```", InContextExamples: list[[str]] = [],
               verbose: bool = True):
    """ call openai's API but take care of caching of resuts
    input_text: input text
    prompt: prompt
    model: model name (as parameter of the query)
    temperature: temperature (0: precise, 1: creative)
    handler: delegate function that will make the call (not necessarily only OpenAI, could be any one)
    """


    if not(input_text) or input_text.strip=="" or not(prompt) or prompt.strip=="":
        logging.warning("No text or promt supplied! Skypping it!")
        return None

    return call_model_with_caching(input_text, prompt, model, temperature, handler, None, None, timeout_retry, delimiter, InContextExamples, verbose)



def process_list(list_input_text: list[str], prompt: str, service_provider: str, model: str, temperature: int,
                 cache_prefix_fp: str = None, delimiter: str = "```", InContextExamples: list[[str]] = []):
    """ process a list of text with a prompt and a model
    list_input_text: list input text
    prompt: prompt
    service provide: either "openai" for the moment
    model: model name (as parameter of the query)
    temperature: temperature (0: precise, 1: creative)
    cache_prefix_fp: prefix of the file to which write content of cache after each call
    """

    if cache_prefix_fp is not None:
        cache_fp = cache_prefix_fp + "___" + "__".join([service_provider, model, str(temperature)]).replace(" ", "_") + ".json"

        if os.path.exists(cache_fp):
            with open(cache_fp) as f:
                map_query_input_output = json.load(f)
        else:
            map_query_input_output = {}
    else:
        map_query_input_output = None
        cache_fp = None

    handler = None
    if service_provider.lower() == "openai": handler = api_call_openai
    if service_provider.lower() == "gptjrc": handler = api_call_gptjrc

    list_output_text = []
    for input_text in tqdm(list_input_text):
        output_text = call_model_with_caching(input_text, prompt, model, temperature, handler, map_query_input_output,
                                              cache_fp, delimiter=delimiter, InContextExamples=InContextExamples)
        list_output_text.append(output_text)

    return list_output_text




if __name__ == "__main__":

    USE_CACHE = True #True #False

    #service_provider = "openai"
    #model_name = "gpt-3.5-turbo-16k"
    #
    # dglc available models: 'OA_SFT_Pythia_12B', 'JRC_RHLF_13B', 'OA_GPT3.5', 'OA_GPT3'
    # model_name = "gpt-3.5-turbo"  #OpenAI name
    # model_name = 'JRC_RHLF_13B'
    #model_name = "OA_SFT_Pythia_12B"   #EleutherAI-pythia-12b
    # model_name = "OA_GPT3"
    # model_name = "GPT@JRC_4"
    #
    #
    service_provider = "gptjrc"
    #model_name = "gpt-35-turbo-0613"
    #model_name = "gpt-35-turbo-16k"
    #model_name = "gpt-4-32k"  #GPT-4 with a context length of 32,768 tokens -  around 116000
    #model_name = "llama-3-70b-instruct"
    model_name = "llama-3-70b-instruct"
    #model_name = "llama-2-13b-chat"
    #model_name = "mpt-30b-chat"
    #model_name="mistral-7b-openorca"
    #model_name="zephyr-7b-beta" #zephyr-7b-beta

    # temperature: temperature_value (0: precise, 1: creative)
    temperature_value = 0.01  # 0.1

    ##################################################################################################

    #OpenAI ChatGPT API
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




    ###########################################################################


    if USE_CACHE:
        # cache_prefix_fp: prefix of the file to which write content of cache after each call
        cache_prefix_fp = "LLMQUERYTEST"
        cache_name = cache_prefix_fp + "___" + "__".join([service_provider, model_name, str(temperature_value)]).replace(" ", "_") + ".json"

        if os.path.exists(cache_name):
            with open(cache_name) as f:
                load_map_query_input_output = json.load(f)
        else:
            load_map_query_input_output = {}

    myPromt = f"""
        translate in Spanish the text below, delimited by triple  \
        Text: 
        """

    myDelimiter = "```"

    ###

    
    encod = encoding_getter(model_name)
    print("\nencodName = " + str(encod.name))

    InContextExamples = []
    
    if InContextExamples:
        ntotExamplesTokens = 0
        for row in InContextExamples:
            for col in row:
                nt = token_counter(col, model_name)
                #print("\nNumber of Tokens in the example = " + str(nt))
                ntotExamplesTokens = ntotExamplesTokens + nt
        #
        print("\nNumber of Tokens of the all examples = " + str(ntotExamplesTokens))

    ###

    if service_provider == "openai":
        if USE_CACHE:
            lambda_model = partial(call_model_with_caching, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_openai,
                                   map_query_input_output=load_map_query_input_output, cache_fp=cache_name, verbose=True)
        else:
            lambda_model = partial(call_model, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_openai,
                                   verbose=True)
    elif service_provider == "gptjrc":
        if USE_CACHE:
            lambda_model = partial(call_model_with_caching, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_gptjrc,
                                   map_query_input_output=load_map_query_input_output, cache_fp=cache_name, verbose=True)
        else:
            lambda_model = partial(call_model, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_gptjrc,
                                   verbose=True)
    

    if lambda_model:
        df = pd.DataFrame([["one, two, three, a step fortward Mary"], ["one, two, three, a step back"]], columns=["text"])
        df["text_es"] = df["text"].apply(lambda_model)

        print("\n")
        print(df)

    print("\nEnd Computations")



