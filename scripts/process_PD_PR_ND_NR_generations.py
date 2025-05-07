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
from src.PD_PR_ND_NR_generations import load_numberbatch, add_hypernyms_to_domain_and_range, apply_thresholds, filter_hypernyms_list


aligned_unique_objects = joblib.load('aligned_unique_objects3.joblib')
aligned_unique_predicates = joblib.load('aligned_unique_predicates3.joblib')
aligned_vg_relationships = joblib.load('aligned_relationships2.joblib')
aligned_unique_attributes = joblib.load('aligned_unique_attributes3.joblib')
aligned_unique_objects_and_attributes = joblib.load('aligned_unique_objects_and_attributes3.joblib')
general_objects_and_attributes = joblib.load('50_general_objects_and_attributes.joblib')
general_predicates = joblib.load('30_general_predicates.joblib')
objects_and_attributes_hierarchies = joblib.load('objects_and_attributes_hierarchies3.joblib')
hypernyms_count = joblib.load('hypernyms_count.joblib')
general_hypernyms_count = joblib.load('general_hypernyms_count.joblib')
semantic_relations_general_predicates = joblib.load('semantic_relations_general_predicates.joblib')
semantic_relations_general_objects_and_attributes = joblib.load(
    'semantic_relations_general_objects_and_attributes.joblib')
semantic_and_functional_relations_general_predicates = joblib.load(
    'semantic_and_functional_relations_general_predicates3.joblib')
semantic_and_functional_relations_general_predicates_with_hierarchies = joblib.load(
    'semantic_and_functional_relations_general_predicates_with_hierarchies.joblib')
range_and_domain_classes_for_general_predicates = joblib.load('range_and_domain_classes_for_general_predicates.joblib')

file_path = 'numberbatch.joblib'
embedding_dict = load_numberbatch(file_path)


## In the following code add synsets of objects and subjects to the Visual Genome relationships dataframe
objects_to_join = aligned_unique_objects[['synsets', 'concept']]
attributes_to_join = aligned_unique_objects[['synsets', 'concept']]
objects_to_join['synsets'] = objects_to_join['synsets'].apply(tuple)
attributes_to_join['synsets'] = attributes_to_join['synsets'].apply(tuple)
objects_and_attributes_to_join = pd.concat([objects_to_join, attributes_to_join], ignore_index=True)
objects_and_attributes_to_join = objects_and_attributes_to_join.drop_duplicates()
objects_and_attributes_to_join = objects_and_attributes_to_join.reset_index(drop=True)


aligned_vg_relationships['subject_synsets'] = aligned_vg_relationships['subject_synsets'].apply(tuple)
aligned_vg_relationships['object_synsets'] = aligned_vg_relationships['object_synsets'].apply(tuple)
objects_and_attributes_to_join.rename(columns={'concept': 'subject_concept'}, inplace=True)
aligned_vg_relationships = aligned_vg_relationships.merge(
    objects_and_attributes_to_join,
    left_on='subject_synsets',
    right_on='synsets',
    how='left'
)

objects_and_attributes_to_join.rename(columns={'subject_concept': 'object_concept'}, inplace=True)
aligned_vg_relationships = aligned_vg_relationships.merge(
    objects_and_attributes_to_join,
    left_on='object_synsets',
    right_on='synsets',
    how='left'
)


# Generation of positive domain, positive range, positive domain with hypernyms (where the hypernyms of each element in the positive domain are added), positive range with hypernyms (where the hypernyms of each element in the positive range are added), positive domain with filtered hypernyms (where the elements in the positive domain with hypernyms are filtered to contain only the general objects and attributes), positive range with filtered hypernyms (where the elements in the positive range with hypernyms are filtered to contain only the general objects and attributes), positive domain using 'capable of' (positive domain obtained from the 'CapableOf' Conceptnet relationship)
range_and_domain_classes_for_general_predicates = aligned_unique_predicates.copy()

range_and_domain_classes_for_general_predicates['positive_domain'] = [[] for _ in range(
    len(range_and_domain_classes_for_general_predicates))]

for index, row in range_and_domain_classes_for_general_predicates.iterrows():
    union_set = set([row['concept']])

    subject_concepts = aligned_vg_relationships.loc[
        aligned_vg_relationships['concept'].isin(union_set),
        'subject_concept'
    ].tolist()

    range_and_domain_classes_for_general_predicates.at[index, 'positive_domain'] = subject_concepts

range_and_domain_classes_for_general_predicates['positive_range'] = [[] for _ in range(
    len(range_and_domain_classes_for_general_predicates))]

for index, row in range_and_domain_classes_for_general_predicates.iterrows():
    union_set = set([row['concept']])

    object_concepts = aligned_vg_relationships.loc[
        aligned_vg_relationships['concept'].isin(union_set),
        'object_concept'
    ].tolist()

    range_and_domain_classes_for_general_predicates.at[index, 'positive_range'] = object_concepts

concept_to_hypernyms = dict(
    zip(objects_and_attributes_hierarchies['concept'], objects_and_attributes_hierarchies['Hypernyms']))

range_and_domain_classes_for_general_predicates = add_hypernyms_to_domain_and_range(
    range_and_domain_classes_for_general_predicates)



general_hypernyms_for_positive_domain = general_hypernyms_count['Hypernyms'].copy()
general_hypernyms_for_positive_range = general_hypernyms_count['Hypernyms'].copy()
general_hypernyms_for_negative_domain = general_hypernyms_count['Hypernyms'].copy()
general_hypernyms_for_negative_range = general_hypernyms_count['Hypernyms'].copy()


general_hypernyms_list_for_positive_domain = list(general_hypernyms_for_positive_domain)
general_hypernyms_list_for_positive_range = list(general_hypernyms_for_positive_range)
general_hypernyms_list_for_negative_domain = list(general_hypernyms_for_negative_domain)
general_hypernyms_list_for_negative_range = list(general_hypernyms_for_negative_range)


range_and_domain_classes_for_general_predicates['positive_domain_with_hypernyms'] = \
range_and_domain_classes_for_general_predicates['positive_domain_with_hypernyms'].apply(apply_thresholds)
range_and_domain_classes_for_general_predicates['positive_range_with_hypernyms'] = \
range_and_domain_classes_for_general_predicates['positive_range_with_hypernyms'].apply(apply_thresholds)


range_and_domain_classes_for_general_predicates['positive_domain_with_filtered_hypernyms'] = \
    range_and_domain_classes_for_general_predicates['positive_domain_with_hypernyms'].apply(filter_hypernyms_list, args=(general_hypernyms_list_for_positive_domain,))
range_and_domain_classes_for_general_predicates['positive_range_with_filtered_hypernyms'] = \
    range_and_domain_classes_for_general_predicates['positive_range_with_hypernyms'].apply(filter_hypernyms_list, args=(general_hypernyms_list_for_positive_range,))



range_and_domain_classes_for_general_predicates['positive_domain_using_capable_of'] = [[] for _ in range(
    len(range_and_domain_classes_for_general_predicates))]

for index, row in range_and_domain_classes_for_general_predicates.iterrows():

    union_set = set([row['capable_of']])

    # Find subject_concepts for matching concepts in aligned_vg_relationships
    subject_concepts = aligned_vg_relationships.loc[
        aligned_vg_relationships['concept'].isin(union_set),
        'subject_concept'
    ].tolist()

    range_and_domain_classes_for_general_predicates.at[index, 'positive_domain_using_capable_of'] = subject_concepts

positive_domain_and_range_for_predicates = range_and_domain_classes_for_general_predicates[
    ['concept', 'positive_domain_with_filtered_hypernyms', 'positive_range_with_filtered_hypernyms']]
# joblib.dump(positive_domain_and_range_for_predicates, "positive_domain_and_range_for_predicates.joblib")


## Generation of filtered related to hypernyms, filtered capable of hypernyms, filtered not capable of hypernyms columns, obtain by filtering the related_to_hypernyms, capable_of_hypernyms, not_capable_of_hypernyms columns, keeping only the general objects and attributes in Visual Genome
semantic_and_functional_relations_general_predicates_with_hierarchies['filtered_related_to_hypernyms'] = \
semantic_and_functional_relations_general_predicates_with_hierarchies['related_to_hypernyms'].apply(
    filter_hypernyms_list, args=(general_hypernyms_list_for_positive_domain,))
semantic_and_functional_relations_general_predicates_with_hierarchies['filtered_capable_of_hypernyms'] = \
semantic_and_functional_relations_general_predicates_with_hierarchies['capable_of_hypernyms'].apply(
    filter_hypernyms_list, args=(general_hypernyms_list_for_positive_domain,))
semantic_and_functional_relations_general_predicates_with_hierarchies['filtered_not_capable_of_hypernyms'] = \
semantic_and_functional_relations_general_predicates_with_hierarchies['not_capable_of_hypernyms'].apply(
    filter_hypernyms_list, args=(general_hypernyms_list_for_negative_domain,))
# joblib.dump(semantic_and_functional_relations_general_predicates_with_hierarchies, "semantic_and_functional_relations_general_predicates_with_hierarchies.joblib")



# joblib.dump(range_and_domain_classes_for_general_predicates, 'range_and_domain_classes_for_general_predicates2.joblib')

