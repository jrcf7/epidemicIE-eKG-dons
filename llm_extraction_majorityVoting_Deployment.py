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
import os
import logging
from tqdm import tqdm
import time


from sklearn.metrics import confusion_matrix, precision_score, average_precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, auc, classification_report


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



def run_EnsembleComputation(LIST_FILES,input_dir, week_abbreviations,month_abbreviations, DictList_syn_Virus,DictList_syn_Country ):

    MODELS = []
    VIRUSES = []
    COUNTRIES = []
    DATES = []
    CASES = []
    DEATHS = []

    for input_filename in LIST_FILES:

        modelName = input_filename.replace(input_dir, "").replace("OutputAnnotatedTexts-", "").replace(
            "Extractions.csv", "").replace(".csv", "")

        MODELS.append(modelName)

        print("\nCOMPUTING = " + modelName)

        df = pd.read_csv(input_filename, sep=',', header=0, encoding='utf-8')

        df = df.drop(columns=['Unnamed: 0', 'json_original', "json_extracted"])  # don't need here

        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        if "cases_count EpiTator" in df.columns:
            y_pred = df['cases_count EpiTator'].str.replace(r"(.*)\=", '', regex=True).str.strip()
        else:
            y_pred = df['cases_extracted']

        y_pred.fillna(value=np.nan, inplace=True)
        y_pred.replace(to_replace=[None], value=np.nan, inplace=True)
        y_pred.replace(to_replace=['None.'], value=np.nan, inplace=True)
        y_pred.replace(to_replace=['None'], value=np.nan, inplace=True)
        y_pred.replace(to_replace=['null'], value=np.nan, inplace=True)
        y_pred.replace(to_replace=[''], value=np.nan, inplace=True)
        y_pred.replace(to_replace=[' '], value=np.nan, inplace=True)

        print("\n\n...COMPUTING NUMBER OF CASES...")

        if y_pred.dtypes == "object":
            for idx, x in tqdm(enumerate(y_pred)):
                if x and (not (x is np.nan)):
                    if "to" in y_pred[idx].lower() or "more" in y_pred[idx].lower() or "over" in y_pred[
                        idx].lower() or "least" in y_pred[idx].lower() or "none" in y_pred[idx].lower() or "record" in \
                            y_pred[idx].lower():
                        y_pred.iat[idx] = np.nan
                    elif "-" in y_pred[idx]:
                        y_pred.iat[idx] = np.nan
                    elif "," in y_pred[idx]:
                        y_pred.iat[idx] = y_pred[idx].replace(",", "")

        y_pred = pd.to_numeric(y_pred, errors='coerce').fillna(np.nan)
        
        y_pred = y_pred.values

        y_pred = np.nan_to_num(y_pred, copy=True, nan=-1, posinf=None, neginf=None)

        CASES.append(y_pred)

        ##############################################

        ydeaths = None
        if "deaths_extracted" in df.columns:
            ydeaths = df["deaths_extracted"]

            ydeaths.fillna(value=np.nan, inplace=True)
            ydeaths.replace(to_replace=[None], value=np.nan, inplace=True)
            ydeaths.replace(to_replace=['None.'], value=np.nan, inplace=True)
            ydeaths.replace(to_replace=['None'], value=np.nan, inplace=True)
            ydeaths.replace(to_replace=['null'], value=np.nan, inplace=True)
            ydeaths.replace(to_replace=[''], value=np.nan, inplace=True)
            ydeaths.replace(to_replace=[' '], value=np.nan, inplace=True)

            print("\n\n...COMPUTING NUMBER OF DEATHS...")

            if df['deaths_extracted'].dtypes == "object":
                for idx, x in tqdm(enumerate(ydeaths)):
                    if x and (not (x is np.nan)):
                        if "to" in ydeaths[idx].lower() or "more" in ydeaths[idx].lower() or "over" in ydeaths[
                            idx].lower() or "least" in ydeaths[idx].lower() or "none" in ydeaths[
                            idx].lower() or "record" in ydeaths[idx].lower():
                            ydeaths.iat[idx] = np.nan
                        elif "-" in ydeaths[idx]:
                            ydeaths.iat[idx] = np.nan
                        elif "," in ydeaths[idx]:
                            ydeaths.iat[idx] = ydeaths[idx].replace(",", "")

            ydeaths = pd.to_numeric(ydeaths, errors='coerce').fillna(np.nan)
            ydeaths = ydeaths.values
            ydeaths = np.nan_to_num(ydeaths, copy=True, nan=-1, posinf=None, neginf=None)
        ###

        DEATHS.append(ydeaths)

        ##############################################

        if "date EpiTator" in df.columns:
            y_pred = df['date EpiTator'].values
        else:
            y_pred = df['date_extracted'].values

        print("\n\n...COMPUTING DATES...")

        for idx in tqdm(range(len(y_pred))):

            delimiters = ["-", "/"]
            string = str(y_pred[idx]).strip().lower()
            for delimiter in delimiters:
                string = " ".join(string.split(delimiter))
            datePred = string.split()

            if len(datePred) > 1:
                for ix in range(len(datePred)):
                    if (datePred[ix][0] == "0"):
                        datePred[ix] = datePred[ix][1:len(datePred[ix])]

            if "date EpiTator" in df.columns:
                datePred_date = datetime.datetime(int(datePred[2]), int(datePred[1]), int(datePred[0]))
            else:

                if (len(datePred) == 4):
                    i_name = -1
                    for i, val in enumerate(datePred):
                        if i_name >= 0:
                            break
                        if datePred[i]:
                            for j, valWeek in enumerate(week_abbreviations):
                                # print(j)
                                if valWeek in val.strip().lower():
                                    week_name = val.strip().lower()
                                    i_name = i
                                    break
                    if i_name >= 0:
                        datePred.remove(week_name)

                for i, val in enumerate(datePred):
                    if datePred[i]:
                        datePred[i] = datePred[i].replace(",", "").replace("st", "").replace("rd", "").replace("th", "")

                for i, val in enumerate(datePred):
                    if datePred[i]:
                        for j, valMonth in enumerate(month_abbreviations):
                            if valMonth in val.strip().lower():
                                month_name = val.strip().lower()
                                try:
                                    datetime_object = datetime.datetime.strptime(month_name, "%b")
                                    month_number = datetime_object.month
                                    datePred[i] = str(month_number)
                                    if i != 1:
                                        # month does not appear to be in position 1, try other positions and in case swap them:
                                        print(
                                            "Month does not appear in the right position (it should be in the middle) ...  " + str(
                                                datePred) + " ...try swapping them: ")
                                        valSupp = datePred[1]
                                        datePred[1] = datePred[i]
                                        datePred[i] = valSupp
                                        print(str(datePred))
                                except:
                                    month_number = (j + 1)
                                    datePred[i] = str(month_number)
                                    if (i != 1) and (len(datePred) > 1):
                                        # month does not appear to be in position 1, try other positions and in case swap them:
                                        print(
                                            "Month does not appear in the right position (it should be in the middle) ...  " + str(
                                                datePred) + " ...try swapping them: ")
                                        valSupp = datePred[1]
                                        datePred[1] = datePred[i]
                                        datePred[i] = valSupp
                                        print(str(datePred))
                                    

                try:

                    # added consose 20240207:
                    if len(datePred) < 3:
                        if int(datePred[0]) > 1900 and int(datePred[0]) < 2050:
                            datePred.append("1")
                            if len(datePred) == 2:
                                datePred.append("1")
                    #

                    datePred_date = datetime.datetime(int(datePred[0]),
                                                      (int(datePred[1]) if int(datePred[1]) > 0 else 1),
                                                      (int(datePred[2]) if int(datePred[2]) > 0 else 1))

                    y_pred[idx] = datePred[0] + "/" + datePred[1] + "/" + datePred[2]

                except:
                    try:
                        datePred_date = datetime.datetime(int(datePred[2]), int(datePred[1]), int(datePred[0]))
                        y_pred[idx] = datePred[2] + "/" + datePred[1] + "/" + datePred[0]
                    except:
                        if (str(datePred) != "['nan']") and (str(datePred) != "['unknown']"):
                            print(" ...Something went wrong with date " + str(datePred) + " ...leaving it like it is")
                        

        DATES.append(y_pred)

        ##############################################

        if "disease EpiTator" in df.columns:
            y_pred = df['disease EpiTator'].values
        else:
            y_pred = df['virus_extracted'].values

        print("\n\n...COMPUTING NAMES OF VIRUSES...")

        for idx in tqdm(range(len(y_pred))):
            # print(idx)

            if (isinstance(y_pred[idx], str) == False) and np.isnan(y_pred[idx]):
                y_pred[idx] = "Not mentioned"
            else:
                all_y_pred_list = []
                all_y_pred_list.append(str(y_pred[idx]).strip())
                all_y_pred_list.append(str(y_pred[idx]).strip().lower())
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

                intersect = set()
                FoundAtIy = -1
                all_y_pred_list_SET = set([x.lower() for x in all_y_pred_list])
                for iy in range(len(DictList_syn_Virus)):
                    b = DictList_syn_Virus[iy]
                    b_set = set([x.lower() for x in b])
                    intersect = all_y_pred_list_SET.intersection(b_set)
                    if len(intersect) > 0:
                        FoundAtIy = iy
                        break

                if FoundAtIy > -1:
                    listFound = DictList_syn_Virus[FoundAtIy]
                    y_pred[idx] = listFound[0]  # take the first element as it is the REPRESENTATIVE LABEL
                else:  # NOT FOUND!!!
                    print("\nERROR at ROW " + str(
                        idx) + " - VIRUS LABEL NOT FOUND IN THE DICTIONARY, IT SHOULD NOT HAPPEN")
                    sys.exit()

        VIRUSES.append(y_pred)

        #################################################

        if "geoname EpiTator" in df.columns:
            y_pred = df['geoname EpiTator'].values
        else:
            y_pred = df['country_extracted'].values

        print("\n\n...COMPUTING NAMES OF PLACES...")

        for idx in tqdm(range(len(y_pred))):
            # print(idx)
            if (isinstance(y_pred[idx], str) == False) and np.isnan(y_pred[idx]):
                y_pred[idx] = "Not mentioned"
            else:
                all_y_pred_list = []
                all_y_pred_list.append(str(y_pred[idx]).strip())
                all_y_pred_list.append(str(y_pred[idx]).strip().lower())
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

                intersect = set()
                FoundAtIy = -1
                all_y_pred_list_SET = set([x.lower() for x in all_y_pred_list])
                for iy in range(len(DictList_syn_Country)):
                    b = DictList_syn_Country[iy]
                    b_set = set([x.lower() for x in b])
                    intersect = all_y_pred_list_SET.intersection(b_set)
                    if len(intersect) > 0:
                        FoundAtIy = iy
                        break

                if FoundAtIy > -1:
                    listFound = DictList_syn_Country[FoundAtIy]
                    y_pred[idx] = listFound[0]  # take the first element as it is the REPRESENTATIVE LABEL
                else:  # NOT FOUND!!!
                    print("\nERROR at ROW " + str(
                        idx) + " - COUNTRY LABEL NOT FOUND IN THE DICTIONARY, IT SHOULD NOT HAPPEN")
                    sys.exit()

        COUNTRIES.append(y_pred)

    ###########################################################################

    majorityVIRUS = []
    majorityCOUNTRY = []
    majorityDATE = []
    majorityCASES = []
    majorityDEATHS = []

    
    print("\n\n...COMPUTING MAJORITY VOTING ENSEMBLE...")

    for idj in tqdm(range(len(VIRUSES[0]))):

        lista_i_viruses = []
        lista_i_countries = []
        lista_i_dates = []
        lista_i_deaths = []
        lista_i_cases = []
        for i, modddl in enumerate(MODELS):
            lista_i_viruses.append(VIRUSES[i][idj])
            lista_i_countries.append(COUNTRIES[i][idj])
            lista_i_dates.append(DATES[i][idj])
            lista_i_deaths.append(DEATHS[i][idj])
            lista_i_cases.append(CASES[i][idj])

        majorityVIRUS.append(most_frequent(lista_i_viruses))
        majorityCOUNTRY.append(most_frequent(lista_i_countries))
        majorityDATE.append(most_frequent(lista_i_dates))
        majorityCASES.append(most_frequent(lista_i_cases))
        majorityDEATHS.append(most_frequent(lista_i_deaths))

    
    df['virus_extracted'] = majorityVIRUS
    df['country_extracted'] = majorityCOUNTRY
    df['date_extracted'] = majorityDATE
    df['cases_extracted'] = majorityCASES
    df['deaths_extracted'] = majorityDEATHS

    # replace some emptyiness values
    df['virus_extracted'].replace('Not mentioned', '', inplace=True)
    df['country_extracted'].replace('Not mentioned', '', inplace=True)
    df['cases_extracted'].replace(-1, '', inplace=True)
    df['deaths_extracted'].replace(-1, '', inplace=True)

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

    if 'deaths_extracted' in df.columns.values:
        try:
            df['deaths_extracted'] = df['deaths_extracted'].astype(float)
        except Exception as err:
            logging.warning(
                f'FAILED to convert deaths_extracted column to float\n\tError: {err}\nLeaving it as a text column then')

    return df


def main():

    DEBUG = False

    input_specify = Path(__file__).parent.resolve() / Path("/eos/jeodpp/data/projects/ETOHA/DATA/etohaSurveillanceScraper/corpus_processed/SUMMARIZED/")

    input_dir = str(input_specify).strip() + "/"

    LIST_FILES = [
    (input_dir + "OutputAnnotatedTexts-llama-3-70b-instruct.csv"),
    (input_dir + "OutputAnnotatedTexts-mistral-7b-openorca.csv"),
    (input_dir + "OutputAnnotatedTexts-zephyr-7b-beta.csv"),
        ]

    month_abbreviations = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    week_abbreviations = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    
    somesyn_filename_Virus = input_dir + "defsyn-Dictionary-NEW-utf8-Viruses_ALL.csv"
    df_syn_Virus = pd.read_csv(somesyn_filename_Virus, sep=',', header=0, dtype=str, encoding='utf-8')
    DictList_syn_Virus = df_syn_Virus.stack().groupby(level=0).apply(list).tolist()

    somesyn_filename_Country = input_dir + "defsyn-Dictionary-NEW-utf8-Countries_ALL.csv"
    df_syn_Country = pd.read_csv(somesyn_filename_Country, sep=',', header=0, dtype=str, encoding='utf-8')
    DictList_syn_Country = df_syn_Country.stack().groupby(level=0).apply(list).tolist()

    start = time.time()

    df= run_EnsembleComputation(LIST_FILES, input_dir, week_abbreviations, month_abbreviations, DictList_syn_Virus,DictList_syn_Country)



    # save input texts
    output_texts_filename = input_dir + "OutputAnnotatedTexts-LLMs-ENSEMBLE.csv"
    df.to_csv(output_texts_filename, sep=',', header=True, index=False, encoding='utf-8')


    #
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nOverall Computational Time... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #

    print("\nEnd Computations")





if __name__ == "__main__":
    main()
