# This script is the one that build the different types of axioms. In particular.
#
# This script does the following:
# * load the Numberbatch embeddings and the stored dataframes
# * loads the NeSy4VRD VRD-World OWL ontology into the KG
# * converts NeSy4VRD visual relationship annotations into RDF triples
#   and loads them into the KG
# * uses the OWL reasoner of Python package OWLRL to materialise the KG
#   (to infer new relationships between image objects that are entailed by
#    the visual relationship data triples in the presence of the
#    VRD-World ontology)
# * uses a SPARQL query to extract the (potentially augmented) set of
#   VR-related triples for each image from the KG
# * converts the extracted triples back into (a potentially augmented set of)
#   NeSy4VRD visual relationship annotations
# * saves the reconstituted (augmented) NeSy4VRD visual relationship annotations
#   to a disk file in JSON format



import json
import os
import sqlite3
import multiprocessing as mp
from functools import lru_cache
import joblib
import ast
import numpy as np
from nltk.corpus import wordnet as wn
import requests
import pandas as pd
import re
from SPARQLWrapper import SPARQLWrapper, JSON
import diskcache as dc
import hashlib
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import pickle
import time
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator
from collections import Counter


# objects_and_attributes_hierarchies[['concept', 'Hypernyms', 'Hyponyms']].to_csv('objects_and_attributes_hierarchies.csv', index=False)
def load_numberbatch(file_path):
    """
    Load Numberbatch embeddings
    :param file_path: the file_path containing
    :return: dictionary where key is a concept and the value is its vector embeddings
    """
    embeddings = joblib.load(file_path)
    english_embeddings = {key: value for key, value in embeddings.items() if key.startswith('/c/en/')}
    return english_embeddings




def get_unique_hypernyms(concepts, concept_to_hypernyms):
    """
    Retrieve and flatten unique Hypernyms for each concept in the 'concepts' list.
    :param concepts: the list of concepts.
    :param concept_to_hypernyms: the mapping from concept to hypernyms.
    :return: the list of unique hypernyms.
    """
    hyperonyms = []
    for concept in concepts:
        hyperonyms.extend(concept_to_hypernyms.get(concept, []))
    return list(set(hyperonyms))

def get_hypernyms(concepts, concept_to_hypernyms):
    """
    Retrieve and flatten Hypernyms for each concept in the concepts list.
    :param concepts: the list of concepts.
    :param concept_to_hypernyms: the mapping from concepts to hypernyms.
    :return: the list of hypernyms.
    """
    hyperonyms = []
    for concept in concepts:
        hyperonyms.extend(concept_to_hypernyms.get(concept, []))
    return hyperonyms



def get_top_common_hypernyms(concepts, concept_to_hypernyms, top_n=50):
    """
    Retrieve the most common hypernyms.
    :param concepts: the list of concepts.
    :param concept_to_hypernyms: the mapping from concepts to hypernyms.
    :param top_n: number of hypernyms.
    :return: the list of hypernyms.
    """
    if not concepts:
        return []

    hypernym_counter = Counter()

    for concept in concepts:
        hypernyms = concept_to_hypernyms.get(concept, [])
        hypernym_counter.update(hypernyms)


    common_hypernyms = hypernym_counter.most_common(top_n)

    return [hypernym for hypernym, count in common_hypernyms]



def add_hypernyms_to_domain_and_range(df_range_domain, concept_to_hypernyms):
    """
    Add columns containing range and domain with hypernyms.
    :param df_range_domain: dataframe containing the columns 'positive_domain' and 'positive_range'.
    :return: the input dataframe with the addition of the columns 'positive_domain_with_hypernyms' and 'positive_range_with_hypernyms'
    """

    # df_range_domain['positive_domain_with_hypernyms'] = df_range_domain['positive_domain'].apply(lambda x: get_unique_hypernyms(x, concept_to_hypernyms))
    # df_range_domain['positive_range_with_hypernyms'] = df_range_domain['positive_range'].apply(lambda x: get_unique_hypernyms(x, concept_to_hypernyms))

    df_range_domain['positive_domain_with_hypernyms'] = df_range_domain['positive_domain'].apply(lambda x: get_hypernyms(x, concept_to_hypernyms))
    df_range_domain['positive_range_with_hypernyms'] = df_range_domain['positive_range'].apply(lambda x: get_hypernyms(x, concept_to_hypernyms))

    return df_range_domain


def filter_hypernyms_list(hypernyms_list, general_hypernyms_list):
    """
    Filter hypernyms_list to keep only strings present in general_hypernyms_list.
    :param hypernyms_list: list of hypernyms.
    :param general_hypernyms_list: list of general hypernyms.
    :return: the intersection between the two lists.
    """
    filtered_strings = [s for s in hypernyms_list if s in general_hypernyms_list]
    print("Size of the list:", len(filtered_strings))
    return filtered_strings


def apply_thresholds(concept_list, percentile = 90):
    """
    Filter concept_list to keep only strings present in a frequency higher than the 90% percentile.
    :param concept_list: list of concepts.
    :return: the list of concepts with a frequency higher than the 90% percentile.
    """
    if not concept_list:
        return []
    row_counts = pd.Series(Counter(concept_list))

    sorted_counts = row_counts.sort_values(ascending=False)

    lower_threshold_count = np.percentile(sorted_counts, percentile)

    filtered_concepts = row_counts[(row_counts >= lower_threshold_count)]
    
    return filtered_concepts.index.tolist()





