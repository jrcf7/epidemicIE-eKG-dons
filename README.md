# epidemicIE-eKG-dons
This repository contains the scripts for Epidemic Information Extraction from the World Health Organization (WHO) Disease Outbreak News ([DONs](https://www.who.int/emergencies/disease-outbreak-news)) with an *Ensemble* of open-souce Large Language Models (LLMs), namely [*Mistral-7B-OpenOrca*](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca), [*Meta-Llama-3-70B-Instruct*](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), and [*Zephyr-7B-Beta*](https://huggingface.co/HuggingFaceH4/Zephyr-7B-Beta).

The method focuses on extracting and structuring critical epidemic information from DONs using the ensemble of LLMs. The extracted information includes disease names, involved countries, dates of events, case totals, and mortality figures. This information is made publicly available through a dataset updated daily, adhering to FAIR (Findable, Accessible, Interoperable, and Reusable) principles and following the paradigms of Linked Open Data (LOD).
The approach involves creating an Epidemic Knowledge Graph (eKG) from the WHO DONs to offer a nuanced representation of the public health domain, providing a structured, interconnected, and formalized framework integrating the extracted epidemic information. The eKG is derived using Semantic Web technologies, such as RDF (Resource Description Framework) and OWL (Web Ontology Language), and is updated daily through an automated process.

All the information about eKG can be found on the dedicated space within the [Joint Research Centre Data Catalogue](https://data.jrc.ec.europa.eu/) at the permanent location: 
http://data.jrc.ec.europa.eu/dataset/89056048-7f5d-4d7c-96ad-f99d1c0f6601, or at the [European Data portal](https://data.europa.eu/en) at : \url{https://data.europa.eu/data/datasets/9ee32efe-af81-48e4-8ad6-a0db06802e03}. 

The broader goals of creating this dataset include supporting the rapid identification of emerging health threats, enabling more effective international collaboration, and accelerating the development of public health interventions.

The paper accompanying this repository is still under review. The details will be disclosed after the paper acceptance.

## Pre-requisites

The code was written in Python 3.10.11 and we used, in particular [Pandas](https://pandas.pydata.org/) for handling the raw extraction files, [Numpy](https://numpy.org/) for data preprocessing and feature extraction, [Scikit-learn](https://scikit-learn.org/) for common machine learning routines, [WordNet](https://wordnet.princeton.edu/) as lexical database for the English language, available via [NLTK](https://www.nltk.org/), assisting in word sense disambiguation and synonymy checking, Sentence Transformers [(entence-BERT (SBERT)](https://sbert.net/) to derive semantically meaningful sentence embeddings for semantic similarity at the sentence level, the [PyTorch](https://pytorch.org/) framework for handling the LLMs workflow, and [LangChain](https://www.langchain.com/) for data processing and integration with the LLMs through modular components and pre-built libraries. 

All the pre-trained LLMs employed in this study have been used through the GPT@JRC initiative of the [Joint Research Centre (JRC)](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/joint-research-centre_en) of the [European Commission](https://commission.europa.eu/index_en), which enables JRC staff to explore the potential uses of AI pre-trained models. GPT@JRC is hosted at the JRC datacentre and supports both open-source AI models, deployed on-premises at [JRC Big Data Analytics Platform (BDAP)](https://jeodpp.jrc.ec.europa.eu/bdap/), and commercial [OpenAI's GPT](https://openai.com/) models running in the European Cloud under a Commission contract with an opt-out clause on prompt analysis by third parties.

The [Protègè](http://protege.stanford.edu) editor is needed to manage, adjust and maintain the knowledge graph structure.

## Installation

Create a Python environment from file **epidemicIE-env.yml** containing all needed libraries:
``` bash
conda env create -f epidemicIE-env.yml
```

## Scripts execution

### Extraction process 

Execute the script to process a corpus of pickled documents, extract epidemic-related information using a chosen language model, and save the results:

```python Extractor_Deployment.py``

where you need need to specify as input variables (in the main()):

1. *MAX_TOKENS_PROMPT*: The maximum number of tokens that the language model prompt can contain. This limit depends on the capabilities of the specific language model being used.

2. *TOKENS_TOLERANCE*: A buffer number of tokens reserved for "InContextExamples" or other components of the prompt, ensuring the total token count doesn't exceed the model's maximum allowed tokens.

3. *USE_CACHE*: A boolean value indicating whether to use cached results to avoid repeated calls to the language model API for the same input, which can save time and API usage.

4. *DATE_IMPUTATION*: A boolean value indicating whether to attempt to impute dates for the cases mentioned in the corpus documents.

5. *JSON_RECONSTRUCT*: A boolean value indicating whether to attempt to reconstruct the JSON output obtained from the language model if it's not properly formatted.

6. *service_provider*: A string specifying the service provider for the language model API, such as "openai" or "gptjrc".

7. *model_name*: A string specifying the name of the language model used for processing the documents, which determines the model's capabilities and token limits.

8. *temperature_value*: A floating-point number used to set the "temperature" for the language model's responses, affecting the randomness and creativity of the output.

9. *InContextExamples*: A list of examples provided as context to the language model to guide its output. The examples can influence the responses to be more in line with the desired format or content.

10. *input_specify*: A Path object specifying the directory where the input corpus files are located and which will also be used to determine the output log and CSV file locations.

11. *cache_name*: A string specifying the filename for the cache, which is used to store and retrieve previous responses from the language model to avoid redundant API calls.

12. *load_map_query_input_output*: A dictionary used to map input queries to their corresponding outputs, which is part of the caching mechanism.

These variables are important for configuring the script to process the corpus data correctly, interact with the language model API efficiently, and output the desired information in an organized manner.
	
### Diseases and Countries Dictionaries creation 

Execute the script to enrich and populate dictionaries for viruses and countries by computing embeddings for terms, comparing them using cosine similarity, and merging similar terms to create an expanded dictionary; it processes CSV files containing the extracted information to enhance some seed dictionaries and outputs new, enriched dictionaries:

```python llm_extraction_Dictionary_Deployment.py``

where you need need to specify as input variables (in the main()):

1. *COSINE_THRESHOLD_VIRUS*: The cosine similarity threshold for virus terms, above which terms are considered similar enough to be merged in the virus dictionary.

2. *COSINE_THRESHOLD_COUNTRY*: The cosine similarity threshold for country terms, above which terms are considered similar enough to be merged in the country dictionary.

3. *input_specify*: A Path object that specifies the directory path where the input CSV files containing annotations are located and which also informs the directory for outputting the new dictionaries.

4. *LIST_FILES*: A list of file paths to CSV files that contain annotations for viruses and countries extracted from text. These files will be used to enrich the seed dictionaries.

5. *model_All*: An instance of SentenceTransformer initialized with a specific pre-trained model, used to compute embeddings for all terms.

6. *model_biobert*: An instance of SentenceTransformer initialized with a BioBERT pre-trained model, used to compute embeddings for biological terms.

7. *tokenizer_transformers_bio*: A tokenizer from the Hugging Face transformers library, associated with the BioBERT model, used for tokenizing biological terms.

8. *model_transformers_bio*: A BioBERT model from the Hugging Face transformers library, used to compute embeddings for biological terms.

9. *somesyn_filename_Virus*: The file path to a seed CSV file containing a list of known virus terms. These terms will be used to create an initial virus dictionary.

10. *somesyn_filename_Country*: The file path to a seed CSV file containing a list of known country terms. These terms will be used to create an initial country dictionary.

11. *DictList_syn_Virus_SEED*: A list of lists containing the virus seed terms loaded from somesyn_filename_Virus, structured to facilitate the processing of embeddings.

12. *DictList_syn_Country_SEED*: A list of lists containing the country seed terms loaded from somesyn_filename_Country, structured to facilitate the processing of embeddings.

These input variablesdetermine the sources of initial data, the models used for embedding generation, the similarity thresholds for term merging, and the locations where enriched dictionaries will be saved.

### Ensemble computation 

Execute the script perform the ensemble computation using data from multiple CSV files. It aggregates and consolidates extracted information about virus outbreaks, such as names of viruses, affected countries, dates, case numbers, and death tolls, by applying majority voting across different model outputs. The consolidated information by majority voting are placed in a single output CSV file.

```python llm_extraction_majorityVoting_Deployment.py``

where you need need to specify as input variables (in the main()):

1. *LIST_FILES*: A list of file paths to CSV files containing annotations from different models, which the script will process.
2. *input_dir*: The directory path where the input CSV files are located.
3. *week_abbreviations*: A list of abbreviated weekday names, used for date parsing.
4. *month_abbreviations*: A list of abbreviated month names, used for date parsing.
5. *DictList_syn_Virus*: A list of virus synonyms loaded from a CSV file, used for normalizing virus names.
6. *DictList_syn_Country*: A list of country synonyms loaded from a CSV file, used for normalizing country names.


### Citations:

If you use this package, we encourage you to add the following references:

- Consoli, S. et al. An Epidemic Knowledge Graph extracted from the World Health Organization's Disease Outbreak News. Scientific Data, submitted.
  
- [dataset] European Commission, Joint Research Centre (JRC). Epidemic Information Extraction from WHO Disease Outbreak News, European Data Portal, Joint Research Centre Data Catalogue, https://doi.org/10.2905/89056048-7f5d-4d7c-96ad-f99d1c0f6601 (2024). PID: http://data.jrc.ec.europa.eu/dataset/89056048-7f5d-4d7c-96ad-f99d1c0f6601 


### References:

- Auer, S. et al. Towards a Knowledge Graph for Science. In Proceedings of the 8th International Conference on Web Intelligence, Mining and Semantics, WIMS ’18 (Association for Computing Machinery, New York, NY, USA, 2018)

- Biderman, S. et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. In Proceedings of Machine Learning Research, vol. 202, 2397 – 2430 (2023)
  
- Brown, T. B. et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, vol.431 2020-December (2020)
  
- Consoli, S. et al. Epidemic Information Extraction for Event-Based Surveillance Using Large Language Models. In Proceedings of Ninth International Congress on Information and Communication Technology (ICICT 2024), vol. 1011434 LNNS, 241 – 252 (Lecture Notes in Networks and Systems, 2024)

- [dataset] European Commission, Joint Research Centre (JRC). Epidemic Information Extraction from WHO Disease Outbreak News, European Data Portal, Joint Research Centre Data Catalogue, https://doi.org/10.2905/89056048-7f5d-4d7c-96ad-f99d1c0f6601 (2024). PID: http://data.jrc.ec.europa.eu/dataset/89056048-7f5d-4d7c-96ad-f99d1c0f6601 

- Miller, G. A. WordNet: A lexical database for English. Commun. ACM 38, 39–41 (1995)

- Reimers, N. & Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (Association for Computational Linguistics, 2019)

- Sagi, O. & Rokach, L. Ensemble learning: A survey. Wiley Interdiscip. Rev. Data Min. Knowl. Discov. 8, 10.1002/widm (2018)
  
- Soille, P. et al. A versatile data-intensive computing platform for information retrieval from big geospatial data. Futur. Gener. Comput. Syst. 81, 30 – 40, 10.1016/j.future.2017.11.007 (2018)

- Vaswani, A. et al. Attention is all you need. In Advances in Neural Information Processing Systems, 5999–6009 (2017)



