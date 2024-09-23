# epidemicIE-eKG-dons
The repository contains the scripts for Epidemic Information Extraction from the World Health Organization (WHO) Disease Outbreak News ([DONs](https://www.who.int/emergencies/disease-outbreak-news)) with an *Ensemble* of open-souce Large Language Models (LLMs), namely [*Mistral-7B-OpenOrca*](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca), [*Meta-Llama-3-70B-Instruct*](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), and [*Zephyr-7B-Beta*](https://huggingface.co/HuggingFaceH4/Zephyr-7B-Beta).

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


## Citations:

If you use this package, we encourage you to add the following references:

- Consoli, S. ...
  
- [dataset] European Commission, Joint Research Centre (JRC). Epidemic Information Extraction from WHO Disease Outbreak News, European Data Portal, Joint Research Centre Data Catalogue, https://doi.org/10.2905/89056048-7f5d-4d7c-96ad-f99d1c0f6601 (2024). PID: http://data.jrc.ec.europa.eu/dataset/89056048-7f5d-4d7c-96ad-f99d1c0f6601 


## References:

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



