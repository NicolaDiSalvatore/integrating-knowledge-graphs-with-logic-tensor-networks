"""
Generate statistics on the generated axioms and Visual Genome concepts aligned with ConceptNet by computing statistics.

Step-by-step overview of the script:

1. Imports utility functions and libraries.
2. Load Visual Genome data
3. Compute Alignment Metrics.
4. Merge Aligned Predicates.
5. Count Axioms per Predicate
6. Enrich with Axioms Context-
7. Generate Aggregated Statistics across all triples and images.
8. Create Visualizations.
9. Compute Concept Frequencies and identifies the most general predicates and objects/attributes.
"""




import src
from src.statistics_generation import count_strings_in_df_column, count_strings_with_frequency, get_total_counts, categorize_relationships, load_joblib_file,standardize_most_similar_concept, fetch_relationship_counts_optimized, calculate_alignment_metrics, calculate_mean_std_alignment_metrics, count_equivalence_axioms_for_df, count_negative_axioms_for_df, count_predicate_in_lhs, process_row, clean_nan_from_lists, format_list_for_latex, wrap_labels
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from collections import Counter
import time
import seaborn as sns
import pandas as pd
import numpy as np



aligned_unique_objects_and_attributes = joblib.load('aligned_unique_objects_and_attributes3.joblib')
aligned_unique_attributes = joblib.load("aligned_unique_attributes2")
aligned_unique_objects = joblib.load("aligned_unique_objects2")
aligned_unique_predicates = joblib.load("aligned_unique_predicates2")
predicates_hierarchies = joblib.load("predicates_hierarchies.joblib")
aligned_predicates = joblib.load("aligned_predicates.joblib")
positive_range_fol_axioms = joblib.load("positive_range_fol_axioms_limited.joblib")
positive_domain_fol_axioms = joblib.load("positive_domain_fol_axioms_limited.joblib")
positive_domain_using_capable_of_fol_axioms = joblib.load("positive_domain_using_capable_of_fol_axioms_limited.joblib")
negative_domain_using_not_capable_of_fol_axioms = joblib.load("negative_domain_using_not_capable_of_fol_axioms_limited.joblib")
hypernyms_fol_axioms = joblib.load("hypernyms_fol_axioms.joblib")
hypernyms_fol_axioms2 = joblib.load("hypernyms_fol_axioms2.joblib")
negative_axioms_with_predicates = joblib.load("negative_axioms_with_predicates.joblib")
negative_axioms_with_predicates_antonyms = joblib.load("negative_axioms_with_predicates_antonyms.joblib")
negative_axioms_with_general_objects_and_attributes = joblib.load("negative_axioms_with_general_objects_and_attributes.joblib")
equivalence_fol_axioms_for_objects_and_attributes = joblib.load("equivalence_fol_axioms_for_objects_and_attributes.joblib")
equivalence_fol_axioms_for_predicates = joblib.load("equivalence_fol_axioms_for_predicates.joblib")
objects_and_attributes_hypernyms_fol_axioms_extended = joblib.load("hypernyms_fol_axioms_extended2.joblib")
predicates_fol_axioms = joblib.load('predicates_fol_axioms2.joblib')
scene_graphs = joblib.load('scene_graphs.joblib')
relationships_full = joblib.load("relationships_full.joblib")


relationships_full = relationships_full.merge(aligned_unique_objects_and_attributes, left_on='subject_synsets', right_on='synsets')
rows_with_none = scene_graphs[scene_graphs.isna().any(axis=1)]


# Calculation of similarities metrics
aligned_unique_predicates = calculate_alignment_metrics(aligned_unique_predicates)
column_to_move = 'top_5_concepts'
aligned_unique_predicates.insert(3, column_to_move, aligned_unique_predicates.pop(column_to_move))


aligned_unique_objects = calculate_alignment_metrics(aligned_unique_objects)
aligned_unique_objects.insert(3, column_to_move, aligned_unique_objects.pop(column_to_move))


aligned_unique_attributes = calculate_alignment_metrics(aligned_unique_attributes)
aligned_unique_attributes.insert(3, column_to_move, aligned_unique_attributes.pop(column_to_move))


aligned_unique_objects_and_attributes = calculate_alignment_metrics(aligned_unique_objects_and_attributes)
aligned_unique_objects_and_attributes.insert(3, column_to_move, aligned_unique_objects_and_attributes.pop(column_to_move))

objects_alignment_performance = calculate_mean_std_alignment_metrics(aligned_unique_objects)
attributes_alignment_performance = calculate_mean_std_alignment_metrics(aligned_unique_attributes)
predicates_alignment_performance = calculate_mean_std_alignment_metrics(aligned_unique_predicates)
objects_and_attributes_alignment_performance = calculate_mean_std_alignment_metrics(aligned_unique_objects_and_attributes)





#Code to update the aligned_predicates (aligned predicates -> aligned_predicates2)
aligned_objects_and_attributes = joblib.load("aligned_objects_and_attributes.joblib")
aligned_objects_and_attributes['synsets'] = aligned_objects_and_attributes['synsets'].apply(
    lambda x: tuple(x) if isinstance(x, (list, tuple, str)) else (x,))

aligned_predicates = aligned_predicates.drop(columns = ['object_name', 'subject_name', 'top_5_concepts', 'top_5_similarities'])
columns = ['object_synsets', 'subject_synsets','synsets']
for column in columns:
    aligned_predicates[column] = aligned_predicates[column].apply(
        lambda x: tuple(x) if isinstance(x, (list, tuple, str)) else (x,)
    )
aligned_predicates = aligned_predicates.drop_duplicates()
aligned_predicates = aligned_predicates.merge(
    aligned_objects_and_attributes,
    left_on=['subject_id', 'subject_x', 'subject_h', 'subject_w', 'subject_y', 'subject_synsets'],
    right_on=['object_id', 'x', 'h', 'w', 'y',  'synsets'],
    how='inner'
)

aligned_predicates = aligned_predicates.drop(columns=['object_id_y', 'synsets_y', 'h', 'w', 'y', 'x'])
aligned_predicates = aligned_predicates.rename(columns={'concept_y': 'subject_concept', 'similarity_y': 'subject_similarity', 'object_id_x': 'object_id', 'synsets_x': 'predicate_synsets', 'concept_x': 'predicate_concept', 'similarity_x': 'predicate_similarity'})


aligned_predicates = aligned_predicates.merge(
    aligned_objects_and_attributes,
    left_on=['object_id', 'object_x', 'object_h', 'object_w', 'object_y', 'object_synsets'],
    right_on=['object_id', 'x', 'h', 'w', 'y',  'synsets'],
    how='inner'
)

aligned_predicates = aligned_predicates.drop(columns=['synsets', 'h', 'w', 'y', 'x'])
aligned_predicates = aligned_predicates.rename(columns={'concept': 'object_concept', 'similarity': 'object_similarity'})

different_image_ids = aligned_predicates[aligned_predicates['image_id_x'] != aligned_predicates['image_id_y']]
aligned_predicates = aligned_predicates.drop('image_id_x', axis=1)
aligned_predicates = aligned_predicates.rename(columns={'image_id_y': 'image_id'})

# joblib.dump(aligned_predicates, "aligned_predicates2.joblib")




# Function to count occurrences of predicate_concept in the left-hand side (LHS) of axioms and generation of aligned_predicates3 (including fol axioms)
### Statistics on first-order logic axioms

positive_domain_fol_axioms = pd.DataFrame(positive_domain_fol_axioms)
positive_domain_fol_axioms = positive_domain_fol_axioms.dropna().drop_duplicates()


positive_range_fol_axioms = pd.DataFrame(positive_range_fol_axioms)
positive_range_fol_axioms = positive_range_fol_axioms.dropna().drop_duplicates()


positive_domain_using_capable_of_fol_axioms = pd.DataFrame(positive_domain_using_capable_of_fol_axioms)
positive_domain_using_capable_of_fol_axioms = positive_domain_using_capable_of_fol_axioms.dropna().drop_duplicates()


negative_domain_using_not_capable_of_fol_axioms = pd.DataFrame(negative_domain_using_not_capable_of_fol_axioms)
negative_domain_using_not_capable_of_fol_axioms = negative_domain_using_not_capable_of_fol_axioms.dropna().drop_duplicates()


negative_axioms_with_predicates = pd.DataFrame(negative_axioms_with_predicates)
negative_axioms_with_predicates = negative_axioms_with_predicates.dropna().drop_duplicates()


equivalence_fol_axioms_for_predicates = pd.DataFrame(equivalence_fol_axioms_for_predicates)
equivalence_fol_axioms_for_predicates = equivalence_fol_axioms_for_predicates.dropna().drop_duplicates()




aligned_predicates['positive_domain_fol_axioms_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_predicate_in_lhs(x, positive_domain_fol_axioms) if isinstance(x, (str, float)) else 0
)

aligned_predicates['positive_range_fol_axioms_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_predicate_in_lhs(x, positive_range_fol_axioms) if isinstance(x, (str, float)) else 0
)

aligned_predicates['positive_domain_using_capable_of_fol_axioms_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_predicate_in_lhs(x, positive_domain_using_capable_of_fol_axioms) if isinstance(x, (str, float)) else 0
)

aligned_predicates['negative_domain_using_not_capable_of_fol_axioms_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_predicate_in_lhs(x, negative_domain_using_not_capable_of_fol_axioms) if isinstance(x, (str, float)) else 0
)

aligned_predicates['negative_axioms_with_predicates_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_negative_axioms_for_df(x, negative_axioms_with_predicates) if isinstance(x, (str, float)) else 0
)

aligned_predicates['equivalence_fol_axioms_for_predicates_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_equivalence_axioms_for_df(x, equivalence_fol_axioms_for_predicates) if isinstance(x, (str, float)) else 0
)

# joblib.dump(aligned_predicates, "aligned_predicates3.joblib")



aligned_predicates = joblib.load("aligned_predicates3.joblib")



aligned_predicates = aligned_predicates.drop(columns = ['Frequency', 'subject_similarity', 'object_similarity'])
aligned_predicates = aligned_predicates.drop_duplicates()
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'positive_domain_fol_axioms_with_hypernyms']], left_on = 'subject_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'positive_domain_fol_axioms_with_hypernyms']], left_on = 'object_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.reset_index(drop=True)
aligned_predicates = aligned_predicates.merge(predicates_fol_axioms[['concept', 'positive_domain_axiom']], left_on = 'predicate_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.drop(columns = ['concept_x', 'concept_y', 'concept'])
aligned_predicates = aligned_predicates.rename(columns={'positive_domain_fol_axioms_with_hypernyms_x': 'subject_positive_domain_axioms',  'positive_domain_fol_axioms_with_hypernyms_y': 'object_positive_domain_axioms', 'positive_domain_axiom': 'predicate_positive_domain_axiom',})
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'positive_range_fol_axioms_with_hypernyms']], left_on = 'subject_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'positive_range_fol_axioms_with_hypernyms']], left_on = 'object_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(predicates_fol_axioms[['concept', 'positive_range_axiom']], left_on = 'predicate_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.drop(columns=['concept_x', 'concept_y', 'concept'])
aligned_predicates = aligned_predicates.rename(columns={'positive_range_fol_axioms_with_hypernyms_x': 'subject_positive_range_axioms',  'positive_range_fol_axioms_with_hypernyms_y': 'object_positive_range_axioms', 'positive_range_axiom': 'predicate_positive_range_axiom',})




## This allows to add, for each entity and axioms category, the axioms list and the axioms count, generating the dataframe aligned_predicates_extended2

aligned_predicates['positive_domain_axioms_in_triple'] = [
    process_row(subject, object, predicate)
    for subject, object, predicate in zip(
        aligned_predicates['subject_positive_domain_axioms'],
        aligned_predicates['object_positive_domain_axioms'],
        aligned_predicates['predicate_positive_domain_axiom']
    )
]
aligned_predicates['positive_domain_axioms_in_triple_count'] = aligned_predicates['positive_domain_axioms_in_triple'].apply(lambda x: len(x) if x is not None else 0)

aligned_predicates['positive_range_axioms_in_triple'] = [
    process_row(subject, object, predicate)
    for subject, object, predicate in zip(
        aligned_predicates['subject_positive_range_axioms'],
        aligned_predicates['object_positive_range_axioms'],
        aligned_predicates['predicate_positive_range_axiom']
    )
]
aligned_predicates['positive_range_axioms_in_triple_count'] = aligned_predicates['positive_range_axioms_in_triple'].apply(lambda x: len(x) if x is not None else 0)



aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'positive_domain_using_capable_of_fol_axioms_list_with_hypernyms']], left_on = 'subject_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'positive_domain_using_capable_of_fol_axioms_list_with_hypernyms']], left_on = 'object_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(predicates_fol_axioms[['concept', 'positive_domain_using_capable_of_axiom']], left_on = 'predicate_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.drop(columns = ['concept_x', 'concept_y', 'concept'])
aligned_predicates = aligned_predicates.rename(columns={'positive_domain_using_capable_of_fol_axioms_list_with_hypernyms_x': 'subject_positive_domain_using_capable_of_axioms',  'positive_domain_using_capable_of_fol_axioms_list_with_hypernyms_y': 'object_positive_domain_using_capable_of_axioms', 'positive_domain_using_capable_of_axiom': 'predicate_positive_domain_using_capable_of_axiom',})


aligned_predicates['positive_domain_using_capable_of_fol_axioms'] = [
    process_row(subject, object, predicate)
    for subject, object, predicate in zip(
        aligned_predicates['subject_positive_domain_using_capable_of_axioms'],
        aligned_predicates['object_positive_domain_using_capable_of_axioms'],
        aligned_predicates['predicate_positive_domain_using_capable_of_axiom']
    )
]
aligned_predicates['positive_domain_using_capable_of_fol_axioms_count'] = aligned_predicates['positive_domain_using_capable_of_fol_axioms'].apply(lambda x: len(x) if x is not None else 0)



aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'negative_domain_using_not_capable_of_fol_axioms_list_with_hypernyms']], left_on = 'subject_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'negative_domain_using_not_capable_of_fol_axioms_list_with_hypernyms']], left_on = 'object_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(predicates_fol_axioms[['concept', 'negative_domain_using_not_capable_of_axiom']], left_on = 'predicate_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.drop(columns = ['concept_x', 'concept_y', 'concept'])
aligned_predicates = aligned_predicates.rename(columns={'negative_domain_using_not_capable_of_fol_axioms_list_with_hypernyms_x': 'subject_negative_domain_using_not_capable_of_axioms',  'negative_domain_using_not_capable_of_fol_axioms_list_with_hypernyms_y': 'object_negative_domain_using_not_capable_of_axioms', 'negative_domain_using_not_capable_of_axiom': 'predicate_negative_domain_using_not_capable_of_axiom',})


aligned_predicates['negative_domain_using_not_capable_of_fol_axioms'] = [
    process_row(subject, object, predicate)
    for subject, object, predicate in zip(
        aligned_predicates['subject_negative_domain_using_not_capable_of_axioms'],
        aligned_predicates['object_negative_domain_using_not_capable_of_axioms'],
        aligned_predicates['predicate_negative_domain_using_not_capable_of_axiom']
    )
]
aligned_predicates['negative_domain_using_not_capable_of_fol_axioms_count'] = aligned_predicates['negative_domain_using_not_capable_of_fol_axioms'].apply(lambda x: len(x) if x is not None else 0)



aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'FOL_ontological_axioms',]], left_on = 'subject_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'FOL_ontological_axioms',]], left_on = 'object_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.drop(columns = ['concept_x', 'concept_y'])
aligned_predicates = aligned_predicates.rename(columns={'FOL_ontological_axioms_x': 'subject_ontological_axioms',  'FOL_ontological_axioms_y': 'object_ontological_axioms'})


aligned_predicates['ontological_fol_axioms'] = [
    process_row(subject, object)
    for subject, object in zip(
        aligned_predicates['subject_ontological_axioms'],
        aligned_predicates['object_ontological_axioms']
    )
]
aligned_predicates['ontological_fol_axioms_count'] = aligned_predicates['ontological_fol_axioms'].apply(lambda x: len(x) if x is not None else 0)



aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'negative_axioms_with_general_objects_and_attributes_list_with_hypernyms',]], left_on = 'subject_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'negative_axioms_with_general_objects_and_attributes_list_with_hypernyms',]], left_on = 'object_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(predicates_fol_axioms[['concept', 'negative_axioms_with_predicate']], left_on = 'predicate_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.drop(columns = ['concept_x', 'concept_y', 'concept'])
aligned_predicates = aligned_predicates.rename(columns={'negative_axioms_with_general_objects_and_attributes_list_with_hypernyms_x': 'subject_negative_axioms', 'negative_axioms_with_general_objects_and_attributes_list_with_hypernyms_y': 'object_negative_axioms', 'negative_axioms_with_predicate': 'predicate_negative_axioms'})


aligned_predicates['negative_fol_axioms'] = [
    process_row(subject, object, predicate)
    for subject, object, predicate in zip(
        aligned_predicates['subject_negative_axioms'],
        aligned_predicates['object_negative_axioms'],
        aligned_predicates['predicate_negative_axioms']
    )
]
aligned_predicates['negative_fol_axioms_count'] = aligned_predicates['negative_fol_axioms'].apply(lambda x: len(x) if x is not None else 0)



aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms',]], left_on = 'subject_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(objects_and_attributes_hypernyms_fol_axioms_extended[['concept', 'equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms',]], left_on = 'object_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.merge(predicates_fol_axioms[['concept', 'equivalence_axioms_with_predicate']], left_on = 'predicate_concept', right_on = 'concept', how = 'left')
aligned_predicates = aligned_predicates.drop(columns = ['concept_x', 'concept_y', 'concept'])
aligned_predicates = aligned_predicates.rename(columns={'equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms_x': 'subject_equivalence_axioms', 'equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms_y': 'object_equivalence_axioms', 'equivalence_axioms_with_predicate': 'predicate_equivalence_axioms'})


aligned_predicates['equivalence_fol_axioms'] = [
    process_row(subject, object, predicate)
    for subject, object, predicate in zip(
        aligned_predicates['subject_equivalence_axioms'],
        aligned_predicates['object_equivalence_axioms'],
        aligned_predicates['predicate_equivalence_axioms']
    )
]
aligned_predicates['equivalence_fol_axioms_count'] = aligned_predicates['equivalence_fol_axioms'].apply(lambda x: len(x) if x is not None else 0)


aligned_predicates['total_axioms'] = aligned_predicates[
    [
        'positive_domain_axioms_in_triple',
        'positive_range_axioms_in_triple',
        'positive_domain_using_capable_of_fol_axioms',
        'negative_domain_using_not_capable_of_fol_axioms',
        'ontological_fol_axioms',
        'negative_fol_axioms',
        'equivalence_fol_axioms'
    ]
].apply(lambda row: list(set(sum(row, []))), axis=1)

aligned_predicates['total_axioms_count'] = aligned_predicates['total_axioms'].apply(len)

aligned_predicates = joblib.load("aligned_predicates_extended.joblib")






## Generaton of plots


aligned_predicates = clean_nan_from_lists(aligned_predicates, columns)

for col in columns:
    aligned_predicates[f'{col}_count'] = aligned_predicates[col].map(lambda x: len(x) if isinstance(x, list) else 0)


# joblib.dump(aligned_predicates, "aligned_predicates_extended2.joblib")


columns = [
    'positive_domain_axioms_in_triple_count', 'positive_range_axioms_in_triple_count',
    'positive_domain_using_capable_of_fol_axioms_count', 'negative_domain_using_not_capable_of_fol_axioms_count',
    'ontological_fol_axioms_count', 'negative_fol_axioms_count', 'equivalence_fol_axioms_count', 'total_axioms_count'
]


triples_axioms_statistics = aligned_predicates[columns].agg(['mean', 'std'])
triples_axioms_statistics.loc['25%'] = aligned_predicates[columns].quantile(0.25)
triples_axioms_statistics.loc['75%'] = aligned_predicates[columns].quantile(0.75)

print(triples_axioms_statistics)


plt.gca().yaxis.set_major_formatter(ScalarFormatter())



# Plot 1: Distribution of theaxioms in triples
filtered_dataframe = aligned_predicates
# filtered_dataframe = aligned_predicates[aligned_predicates['positive_domain_axioms_in_triple_count']> 0]
plt.figure(1)
sns.histplot(filtered_dataframe['positive_domain_axioms_in_triple_count'], bins=20, kde=False)
plt.xlabel('Positive Domain Axioms In Triple Count')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for objects')
plt.xlim(0, filtered_dataframe['positive_domain_axioms_in_triple_count'].max())
# max_frequency = aligned_predicates['positive_domain_axioms_in_triple_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)



# filtered_dataframe = aligned_predicates[aligned_predicates['positive_domain_using_capable_of_fol_axioms_count']>0]
plt.figure(2)
sns.histplot(filtered_dataframe['positive_domain_using_capable_of_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Positive Domain Using Capable Of Axioms Count')
plt.ylabel('Frequency')
# plt.title('Distribution of the best similarity for attributes')
plt.xlim(0, filtered_dataframe['positive_domain_using_capable_of_fol_axioms_count'].max())
# max_frequency = filtered_dataframe['positive_domain_using_capable_of_fol_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)



# filtered_dataframe = aligned_predicates[aligned_predicates['positive_range_axioms_in_triple_count']>0]
plt.figure(3)
sns.histplot(filtered_dataframe['positive_range_axioms_in_triple_count'], bins=20, kde=False)
plt.xlabel('Positive Range Axioms In Triple Count')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for objects and attributes')
plt.xlim(0, filtered_dataframe['positive_range_axioms_in_triple_count'].max())
# max_frequency = filtered_dataframe['positive_range_axioms_in_triple_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)


# filtered_dataframe = aligned_predicates[aligned_predicates['negative_domain_using_not_capable_of_fol_axioms_count']>0]
plt.figure(4)
sns.histplot(filtered_dataframe['negative_domain_using_not_capable_of_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Negative Domain Using Not Capable Of Axioms Count')
plt.ylabel('Frequency')
# plt.title('Distribution of the best similarity for predicates')
plt.xlim(0, filtered_dataframe['negative_domain_using_not_capable_of_fol_axioms_count'].max())
# max_frequency = filtered_dataframe['negative_domain_using_not_capable_of_fol_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)


# filtered_dataframe = aligned_predicates[aligned_predicates['ontological_fol_axioms_count']>0]
plt.figure(5)
sns.histplot(filtered_dataframe['ontological_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Ontological Axioms Count')
plt.ylabel('Frequency')
# plt.title('Distribution of the best similarity for objects')
plt.xlim(0, filtered_dataframe['ontological_fol_axioms_count'].max())
# max_frequency = filtered_dataframe['ontological_fol_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)


filtered_dataframe = aligned_predicates
# filtered_dataframe = aligned_predicates[aligned_predicates['negative_fol_axioms_count']>0]
plt.figure(6)
sns.histplot(filtered_dataframe['negative_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Negative Axioms Count')
plt.ylabel('Frequency')
plt.xlim(0, filtered_dataframe['negative_fol_axioms_count'].max())
# max_frequency = filtered_dataframe['negative_fol_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)


# filtered_dataframe = aligned_predicates[aligned_predicates['equivalence_fol_axioms_count']>0]
plt.figure(7)
sns.histplot(filtered_dataframe['equivalence_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Equivalence Axioms Count')
plt.ylabel('Frequency')
plt.xlim(0, filtered_dataframe['equivalence_fol_axioms_count'].max())
max_frequency = filtered_dataframe['equivalence_fol_axioms_count'].value_counts().max()
plt.ylim(0, max_frequency*1.2)


# filtered_dataframe = aligned_predicates[aligned_predicates['negative_fol_axioms_count']>0]
plt.figure(8)
sns.histplot(filtered_dataframe['total_axioms_count'], bins=20, kde=False)
plt.xlabel('Total Axioms Count')
plt.ylabel('Frequency')
plt.xlim(0, filtered_dataframe['total_axioms_count'].max())
plt.show()





# aligned_predicates = joblib.load("aligned_predicates_extended2.joblib")

# The following code create list of axioms for each axiom typology
positive_domain_fol_axioms_list = []
for value in positive_domain_fol_axioms['axiom']:
    if value is not None and len(value) > 0:
        positive_domain_fol_axioms_list.append(value)
positive_domain_fol_axioms_list = list(set(positive_domain_fol_axioms_list))


positive_range_fol_axioms_list =[]
for value in positive_range_fol_axioms['axiom']:
    if value is not None and len(value) > 0:
        positive_range_fol_axioms_list.append(value)
positive_range_fol_axioms_list = list(set(positive_range_fol_axioms_list))


positive_domain_using_capable_of_fol_axioms_list =[]
for value in positive_domain_using_capable_of_fol_axioms['axiom']:
    if value is not None and len(value) > 0:
        positive_domain_using_capable_of_fol_axioms_list.append(value)
positive_domain_using_capable_of_fol_axioms_list = list(set(positive_domain_using_capable_of_fol_axioms_list))


negative_domain_using_not_capable_of_fol_axioms_list =[]
for value in negative_domain_using_not_capable_of_fol_axioms['axiom']:
    if value is not None and len(value) > 0:
        negative_domain_using_not_capable_of_fol_axioms_list.append(value)
negative_domain_using_not_capable_of_fol_axioms_list = list(set(negative_domain_using_not_capable_of_fol_axioms_list))



hypernyms_fol_axioms_list =[]
for value in hypernyms_fol_axioms['FOL_axioms_list']:
    if value is not None and len(value) > 0:
        for item in value:
            hypernyms_fol_axioms_list.append(item)
hypernyms_fol_axioms_list = list(set(hypernyms_fol_axioms_list))


# hypernyms_fol_axioms2_list =[]
# for value in hypernyms_fol_axioms['FOL_axioms_list']:
#     # Check if the value is not None and is not an empty list
#     if value is not None and len(value) > 0:
#        for item in value:
#             hypernyms_fol_axioms2_list.append(item)
# hypernyms_fol_axioms2_list = list(set(hypernyms_fol_axioms2_list))


hypernyms_fol_axioms_total_list = hypernyms_fol_axioms_list


negative_axioms_with_predicates_list =[]
for value in negative_axioms_with_predicates['negative_fol_axioms']:
    if value is not None and len(value) > 0:
        negative_axioms_with_predicates_list.append(value)
negative_axioms_with_predicates_list = list(set(negative_axioms_with_predicates_list))


negative_axioms_with_predicates_antonyms_list =[]
for value in negative_axioms_with_predicates_antonyms['negative_fol_axioms']:
    if value is not None and len(value) > 0:
        negative_axioms_with_predicates_antonyms_list.append(value)
negative_axioms_with_predicates_antonyms_list = list(set(negative_axioms_with_predicates_antonyms_list))


negative_axioms_with_general_objects_and_attributes_list =[]
for value in negative_axioms_with_general_objects_and_attributes['unique_axioms']:
    if value is not None and len(value) > 0:
        negative_axioms_with_general_objects_and_attributes_list.append(value)
negative_axioms_with_general_objects_and_attributes_list = list(set(negative_axioms_with_general_objects_and_attributes_list))

negative_axioms_total_list = set(negative_axioms_with_predicates_list)
negative_axioms_total_list.update(negative_axioms_with_predicates_antonyms_list)
negative_axioms_total_list.update(negative_axioms_with_general_objects_and_attributes_list)
negative_axioms_total_list = list(negative_axioms_total_list)

equivalence_axioms_list = set(equivalence_fol_axioms_for_objects_and_attributes)
equivalence_axioms_list.update(equivalence_fol_axioms_for_predicates)
equivalence_axioms_list = list(equivalence_axioms_list)


## The following code is made to plot the log count for each axioms category

lists = {
    "positive_domain_fol_axioms_list": positive_domain_fol_axioms,
    "positive_domain_using_capable_of_fol_axioms_list": positive_domain_using_capable_of_fol_axioms,
    "positive_range_fol_axioms_list": positive_range_fol_axioms,
    "negative_domain_using_not_capable_of_fol_axioms_list": negative_domain_using_not_capable_of_fol_axioms,
    "hypernyms_fol_axioms_list": hypernyms_fol_axioms,
    "hypernyms_fol_axioms2_list": hypernyms_fol_axioms2,
    "negative_axioms_with_predicates_list": negative_axioms_with_predicates,
    "negative_axioms_with_predicates_antonyms_list": negative_axioms_with_predicates_antonyms,
    "negative_axioms_with_general_objects_and_attributes_list": negative_axioms_with_general_objects_and_attributes,
    "equivalence_fol_axioms_for_objects_and_attributes_list": equivalence_fol_axioms_for_objects_and_attributes,
    "equivalence_fol_axioms_for_predicates_list": equivalence_fol_axioms_for_predicates
}



positive_domain_count = len(positive_domain_fol_axioms[positive_domain_fol_axioms['has_axiom']==True])
positive_domain_using_capable_of_count = len(positive_domain_using_capable_of_fol_axioms[positive_domain_using_capable_of_fol_axioms['has_axiom']==True])
positive_range_count = len(positive_range_fol_axioms[positive_range_fol_axioms['has_axiom']==True])
negative_domain_using_not_capable_count = len(negative_domain_using_not_capable_of_fol_axioms[negative_domain_using_not_capable_of_fol_axioms['has_axiom']==True])
hypernyms_count = len(hypernyms_fol_axioms_total_list)
negative_count = len(negative_axioms_total_list)
equivalence_count = len(equivalence_axioms_list)


data = pd.DataFrame({
    "First-Order Logic Axioms": [
        "Positive Domain", "Positive Domain Using Capable Of", "Positive Range",
        "Negative Domain Using Not Capable Of", "Ontological",
        "Negative", "Equivalence"
    ],
    "Log(Count)": [
        np.log(positive_domain_count), np.log(positive_domain_using_capable_of_count),
        np.log(positive_range_count), np.log(negative_domain_using_not_capable_count),
        np.log(hypernyms_count), np.log(negative_count), np.log(equivalence_count)
    ]
})

data = pd.DataFrame({
    "First-Order Logic Axioms": ["Positive Domain"] * np.log(len(positive_domain_fol_axioms)) +
                ["Positive Domain Using Capable Of"] * np.log(len(positive_domain_using_capable_of_fol_axioms)) +
                ["Positive Range"] * np.log(len(positive_range_fol_axioms)) +
                ["Negative Domain Using Not Capable Of"] * np.log(len(negative_domain_using_not_capable_of_fol_axioms)) +
                ["Ontological"] * np.log(len(hypernyms_fol_axioms2)) +
                ["Negative Using Predicates"] * np.log(len(negative_axioms_with_predicates)) +
                ["Negative Using Predicates Antonyms"] * np.log(len(negative_axioms_with_predicates_antonyms)) +
                ["Negative Using General Objects And Attributes"] * np.log(len(negative_axioms_with_general_objects_and_attributes)) +
                ["Equivalence Using General Objects And Attributes"] * np.log(len(equivalence_fol_axioms_for_objects_and_attributes)) +
                ["Equivalence Using Predicates"] * np.log(len(equivalence_fol_axioms_for_predicates))
})


pal = [ "#167288", "#8cdaec", "#b45248", "#d48c84", "#a89a49", "#d6cfa2", "#3cb464", "#9bddb1", "#643c6a", "#836394" ]





fig = plt.figure(figsize=(14, 9))


data['Log(Count)'] = np.log(data['First-Order Logic Axioms'].value_counts()).reindex(data['First-Order Logic Axioms']).values

pl = sns.barplot(x="First-Order Logic Axioms", y="Log(Count)", data=data, palette=pal)



wrapped_labels = wrap_labels([label.get_text() for label in pl.get_xticklabels()], max_width=10)
pl.set_xticklabels(wrapped_labels)
plt.xticks(rotation=0, ha='center', fontsize=10)
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
plt.show()


# Define the columns for which you want to apply the unique values and list size operations

aligned_predicates = aligned_predicates.rename(columns={
    'positive_domain_axioms_in_triple': 'positive_domain_axioms',
    'positive_range_axioms_in_triple': 'positive_range_axioms',
})
columns = [
    'positive_domain_axioms',
    'positive_range_axioms',
    'positive_domain_using_capable_of_fol_axioms',
    'negative_domain_using_not_capable_of_fol_axioms',
    'ontological_fol_axioms',
    'negative_fol_axioms',
    'equivalence_fol_axioms',
    'total_axioms'
]


# In the following part is built the dataframe fol_axioms_in_images, which contains the axioms and their count for each image. Afterwards, the statistics are plotted
# fol_axioms_in_images = joblib.load("fol_axioms_in_images2.joblib")


fol_axioms_in_images = aligned_predicates.groupby('image_id')[columns].apply(
    lambda df: df.apply(lambda x: list(set(sum(x, []))))
).reset_index()


for col in columns:
    fol_axioms_in_images[col] = fol_axioms_in_images[col].apply(lambda x: [i for i in x if i is not None] if isinstance(x, list) else x)

fol_axioms_in_images = clean_nan_from_lists(fol_axioms_in_images, columns)


for col in columns:
    fol_axioms_in_images[f'{col}_count'] = fol_axioms_in_images[col].apply(len)


# Define columns with count names for future calculations
columns = [
    'positive_domain_axioms_count', 'positive_range_axioms_count',
    'positive_domain_using_capable_of_fol_axioms_count', 'negative_domain_using_not_capable_of_fol_axioms_count',
    'ontological_fol_axioms_count', 'negative_fol_axioms_count', 'equivalence_fol_axioms_count', 'total_axioms_count'
]


images_axioms_statistics = fol_axioms_in_images[columns].agg(['mean', 'std'])
images_axioms_statistics.loc['25%'] = fol_axioms_in_images[columns].quantile(0.25)
images_axioms_statistics.loc['75%'] = fol_axioms_in_images[columns].quantile(0.75)

# print(images_axioms_statistics)


plt.figure(1)
sns.histplot(fol_axioms_in_images['positive_domain_axioms_count'], bins=20, kde=False)
plt.xlabel('Positive Domain Axioms Count')
plt.ylabel('Frequency')
plt.title('Distribution of the number of positive domain axioms')
plt.xlim(0, fol_axioms_in_images['positive_domain_axioms_count'].max())
# max_frequency = fol_axioms_in_images['positive_domain_axioms_in_triple_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)

plt.figure(2)
sns.histplot(fol_axioms_in_images['positive_domain_using_capable_of_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Positive Domain Using Capable Of Axioms Count')
plt.ylabel('Frequency')
plt.title('Distribution of the count of positive domain using capable of axioms in images')
plt.xlim(0, fol_axioms_in_images['positive_domain_using_capable_of_fol_axioms_count'].max())
# max_frequency = fol_axioms_in_images['positive_domain_using_capable_of_fol_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)


plt.figure(3)
sns.histplot(fol_axioms_in_images['positive_range_axioms_count'], bins=20, kde=False)
plt.xlabel('Positive Range Axioms Count')
plt.ylabel('Frequency')
plt.title('Distribution of the count of positive range axioms in images')
plt.xlim(0, fol_axioms_in_images['positive_range_axioms_count'].max())
# max_frequency = fol_axioms_in_images['positive_range_axioms_in_triple_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)



plt.figure(4)
sns.histplot(fol_axioms_in_images['negative_domain_using_not_capable_of_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Negative Domain Using Not Capable Of Axioms Count')
plt.ylabel('Frequency')
plt.title('Distribution of the count of negative domain using \'not capable of\' axioms in images')
plt.xlim(0, fol_axioms_in_images['negative_domain_using_not_capable_of_fol_axioms_count'].max())
# max_frequency = fol_axioms_in_images['negative_domain_using_not_capable_of_fol_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)



plt.figure(5)
sns.histplot(fol_axioms_in_images['ontological_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Ontological Axioms Count')
plt.ylabel('Frequency')
plt.title('Distribution of the count of ontological axioms in triples')
plt.xlim(0, fol_axioms_in_images['ontological_fol_axioms_count'].max())
# max_frequency = fol_axioms_in_images['ontological_fol_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)



plt.figure(6)
sns.histplot(fol_axioms_in_images['negative_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Negative Axioms Count')
plt.ylabel('Frequency')
plt.title('Distribution of the count of negative axioms in triples')
plt.xlim(0, fol_axioms_in_images['negative_fol_axioms_count'].max())
# max_frequency = fol_axioms_in_images['negative_fol_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)


plt.figure(7)
sns.histplot(fol_axioms_in_images['equivalence_fol_axioms_count'], bins=20, kde=False)
plt.xlabel('Equivalence Axioms Count')
plt.ylabel('Frequency')
plt.title('Distribution of the count of equivalence axioms in triples')
plt.xlim(0, fol_axioms_in_images['equivalence_fol_axioms_count'].max())
max_frequency = fol_axioms_in_images['equivalence_fol_axioms_count'].value_counts().max()
plt.ylim(0, max_frequency*1.2)



plt.figure(8)
sns.histplot(fol_axioms_in_images['total_axioms_count'], bins=20, kde=False)
plt.xlabel('Total Axioms Count')
plt.ylabel('Frequency')
plt.title('Distribution of the count of total axioms in triples')
plt.xlim(0, fol_axioms_in_images['total_axioms_count'].max())
# max_frequency = fol_axioms_in_images['total_axioms_count'].value_counts().max()
# plt.ylim(0, max_frequency*1.2)
plt.show()


# joblib.dump(fol_axioms_in_images, "fol_axioms_in_images.joblib")
# joblib.dump(fol_axioms_in_images, "fol_axioms_in_images2.joblib")






#The foollowing code is to plot the frequency of objects, attributes, predicates and objects/attributes hierarchies
db_path = 'assertions.db'


aligned_objects = load_joblib_file("aligned_objects.joblib")
aligned_objects = standardize_most_similar_concept(aligned_objects)
triples_count_df = fetch_relationship_counts_optimized(db_path, aligned_objects)
joblib_file = 'objects_Conceptnet_relationships_count.joblib'
# joblib.dump(triples_count_df, joblib_file)


aligned_attributes = load_joblib_file("aligned_attributes.joblib")
aligned_attributes = standardize_most_similar_concept(aligned_attributes)
triples_count_df = fetch_relationship_counts_optimized(db_path, aligned_attributes)
joblib_file = 'attributes_Conceptnet_relationships_count.joblib'
# joblib.dump(triples_count_df, joblib_file)


aligned_predicates = load_joblib_file("aligned_relationships.joblib")
aligned_predicates = standardize_most_similar_concept(aligned_predicates)
triples_count_df = fetch_relationship_counts_optimized(db_path, aligned_predicates)
joblib_file = 'predicates_Conceptnet_relationships_count.joblib'
# joblib.dump(triples_count_df, joblib_file)



triples_count_df = joblib.load('objects_Conceptnet_relationships_count.joblib')
# triples_count_df = joblib.load('predicates_Conceptnet_relationships_count.joblib')
total_count_relationships_df = get_total_counts(triples_count_df)
total_count_relationships_df = total_count_relationships_df.astype(int)
total_count_macrocategories_df = categorize_relationships(total_count_relationships_df)
total_count_macrocategories_df = total_count_macrocategories_df.astype(int)



objects_and_attributes_hierarchies = joblib.load("objects_and_attributes_hierarchies3.joblib")
all_concepts = [concept for sublist in objects_and_attributes_hierarchies['Hypernyms'] for concept in sublist]
concept_counts = Counter(all_concepts)
concept_freq_df = pd.DataFrame(concept_counts.items(), columns=['Concept', 'Frequency'])
objects_and_attributes_hierarchies_freq_df = concept_freq_df.sort_values(by='Frequency', ascending=False)
objects_and_attributes_hierarchies_freq_df = objects_and_attributes_hierarchies_freq_df.reset_index()
objects_and_attributes_hierarchies_freq_df = objects_and_attributes_hierarchies_freq_df.drop('index', axis=1)



# Conversion of aligned_relationships -> aligned_relationships2 (addition of top_5_most_similar_concepts and the frequency of each VG relationship),
# aligned_unique predicates2 -> aligned_unique_predicates3
# Statistics on Visual Genome relationships

#Selection of general predicates

aligned_predicates = load_joblib_file("aligned_relationships.joblib")
all_predicates = [concept for concept in aligned_predicates['most_similar_concept']]

aligned_unique_predicates = joblib.load('aligned_unique_predicates2.joblib')
aligned_unique_predicates['synsets'] = aligned_unique_predicates['synsets'].apply(tuple)
aligned_predicates['synsets'] = aligned_predicates['synsets'].apply(tuple)
aligned_unique_predicates['concept'] = aligned_unique_predicates['top_5_concepts'].apply(lambda x: x[0] if x else None)
aligned_unique_predicates['similarity'] = aligned_unique_predicates['top_5_similarities'].apply(lambda x: x[0] if x else None)
aligned_predicates = aligned_predicates.drop(['most_similar_concept', 'similarity'], axis=1)
aligned_predicates = aligned_predicates.merge(aligned_unique_predicates, on='synsets', how='left')


concept_counts = Counter(aligned_predicates['concept'])
concept_freq_df = pd.DataFrame(concept_counts.items(), columns=['concept', 'Frequency'])
concept_freq_df= concept_freq_df.dropna(subset=['concept'])
predicates_freq_df = concept_freq_df.sort_values(by='Frequency', ascending=False)
predicates_freq_df.reset_index
aligned_predicates = aligned_predicates.merge(predicates_freq_df, on='concept', how='left')
aligned_unique_predicates = aligned_unique_predicates.merge(predicates_freq_df, on='concept', how='left')
aligned_unique_predicates['synsets'] = aligned_unique_predicates['synsets'].apply(list)
aligned_unique_predicates['Frequency'] = aligned_unique_predicates['Frequency'].astype(int)
aligned_predicates['synsets'] = aligned_predicates['synsets'].apply(list)
general_predicates = predicates_freq_df.head(30).apply(list)
# joblib.dump(aligned_unique_predicates, 'aligned_unique_predicates3.joblib')
# joblib.dump(aligned_predicates, 'aligned_relationships2.joblib')
# joblib.dump(general_predicates, 'top_30_more_general_predicates.joblib')




# Conversion of aligned_objects -> aligned_objects2 (addition of top_5_most_similar_concepts and the frequency of each VG relationship),
# aligned_unique_objects2 -> aligned_unique_objects3
# Statistics on Visual Genome objects

# Conversion of aligned_attributes -> aligned_attributes2 (addition of top_5_most_similar_concepts and the frequency of each VG relationship),
# aligned_unique_attributes2 -> aligned_unique_attributes3
# Statistics on Visual Genome attributes


aligned_objects = load_joblib_file("aligned_objects.joblib")
aligned_unique_objects = joblib.load('aligned_unique_objects2.joblib')
aligned_unique_objects['synsets'] = aligned_unique_objects['synsets'].apply(tuple)
aligned_objects['synsets'] = aligned_objects['synsets'].apply(tuple)
aligned_unique_objects['concept'] = aligned_unique_objects['top_5_concepts'].apply(lambda x: x[0] if x else None)
aligned_unique_objects['similarity'] = aligned_unique_objects['top_5_similarities'].apply(lambda x: x[0] if x else None)
aligned_objects = aligned_objects.drop(['most_similar_concept', 'similarity'], axis=1)
aligned_objects = aligned_objects.merge(aligned_unique_objects, on='synsets', how='left')


aligned_attributes = load_joblib_file("aligned_attributes.joblib")
aligned_unique_attributes = joblib.load('aligned_unique_attributes2.joblib')
aligned_unique_attributes['synsets'] = aligned_unique_attributes['synsets'].apply(tuple)
aligned_attributes['synsets'] = aligned_attributes['synsets'].apply(tuple)
aligned_unique_attributes['concept'] = aligned_unique_attributes['top_5_concepts'].apply(lambda x: x[0] if x else None)
aligned_unique_attributes['similarity'] = aligned_unique_attributes['top_5_similarities'].apply(lambda x: x[0] if x else None)
aligned_attributes = aligned_attributes.drop(['most_similar_concept', 'similarity'], axis=1)
aligned_attributes = aligned_attributes.merge(aligned_unique_attributes, on='synsets', how='left')

concept_counts = Counter(aligned_attributes['concept'])
concept_freq_df = pd.DataFrame(concept_counts.items(), columns=['concept', 'Frequency'])
concept_freq_df= concept_freq_df.dropna(subset=['concept'])
concept_freq_df = concept_freq_df.sort_values(by='Frequency', ascending=False)
concept_freq_df = concept_freq_df.reset_index()
aligned_attributes = aligned_attributes.merge(concept_freq_df, on='concept', how='left')
aligned_unique_attributes = aligned_unique_attributes.merge(concept_freq_df, on='concept', how='left')
aligned_unique_attributes['synsets'] = aligned_unique_attributes['synsets'].apply(list)
aligned_attributes['synsets'] = aligned_attributes['synsets'].apply(list)
general_attributes = concept_freq_df.head(30).apply(list)
joblib.dump(aligned_unique_attributes, 'aligned_unique_attributes3.joblib')
joblib.dump(aligned_attributes, 'aligned_attributes2.joblib')
joblib.dump(general_attributes, 'top_30_more_general_attributes.joblib')



#Union of objects and attributes
aligned_unique_objects = joblib.load('aligned_unique_objects3.joblib')
aligned_unique_predicates = joblib.load('aligned_unique_predicates3.joblib')
aligned_unique_attributes = joblib.load('aligned_unique_attributes3.joblib')
aligned_unique_objects = aligned_unique_objects.drop('synsets', axis=1)
aligned_unique_objects = aligned_unique_objects.drop_duplicates(subset=['concept'])
aligned_unique_attributes = aligned_unique_attributes.drop('synsets', axis=1)
aligned_unique_attributes = aligned_unique_attributes.drop_duplicates(subset=['concept'])


aligned_unique_objects_and_attributes = pd.concat([aligned_unique_objects, aligned_unique_attributes], ignore_index=True)
aligned_unique_objects_and_attributes['top_5_concepts'] = aligned_unique_objects_and_attributes['top_5_concepts'].apply(tuple)
aligned_unique_objects_and_attributes['top_5_similarities'] = aligned_unique_objects_and_attributes['top_5_similarities'].apply(tuple)
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes.groupby(['concept', 'similarity', 'top_5_concepts', 'top_5_similarities'], as_index=False)['Frequency'].sum()
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes.drop_duplicates(subset=['concept'])
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes.sort_values(by='Frequency', ascending=False)
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes.reset_index(drop=True)
aligned_unique_objects_and_attributes['top_5_concepts'] = aligned_unique_objects_and_attributes['top_5_concepts'].apply(list)
aligned_unique_objects_and_attributes['top_5_similarities'] = aligned_unique_objects_and_attributes['top_5_similarities'].apply(list)
# joblib.dump(aligned_unique_objects_and_attributes, 'aligned_unique_objects_and_attributes3.joblib')
# joblib.dump(aligned_unique_objects_and_attributes.head(50), '50_general_objects_and_attributes.joblib')


aligned_objects_and_attributes = pd.concat([aligned_objects, aligned_attributes], axis=0)
aligned_objects_and_attributes = aligned_objects_and_attributes.drop(columns= ['index', 'merged_object_ids'])
columns = ['synsets', 'names', 'top_5_similarities',  'top_5_concepts']
for column in columns:
    aligned_objects_and_attributes[column] = aligned_objects_and_attributes[column].apply(
        lambda x: tuple(x) if isinstance(x, (list, tuple, str)) else (x,)
    )



# duplicates = aligned_objects_and_attributes[
#     aligned_objects_and_attributes.duplicated(subset=['image_id', 'object_id', 'synsets', 'names', 'h', 'w', 'y', 'x'], keep=False)
# ]

aligned_objects_and_attributes = aligned_objects_and_attributes.dropna(subset=['concept'])
aligned_objects_and_attributes = aligned_objects_and_attributes.reset_index(drop=True)
aligned_objects_and_attributes = aligned_objects_and_attributes[['image_id','object_id', 'synsets', 'h', 'w', 'y', 'x', 'concept', 'similarity']].drop_duplicates()
aligned_objects_and_attributes = aligned_objects_and_attributes.drop_duplicates()

aligned_objects_and_attributes['synsets'] = aligned_objects_and_attributes['synsets'].apply(lambda x: list(x) if isinstance(x, (list, tuple, str)) else (x,))

# joblib.dump(aligned_objects_and_attributes, "aligned_objects_and_attributes.joblib")




# aligned_objects = joblib.load('aligned_objects2.joblib')
# aligned_attributes = joblib.load('aligned_attributes2.joblib')
# aligned_unique_objects = joblib.load('aligned_unique_objects3.joblib')
# aligned_unique_predicates = joblib.load('aligned_unique_predicates3.joblib')
# aligned_unique_attributes = joblib.load('aligned_unique_attributes3.joblib')
# aligned_unique_objects_and_attributes = joblib.load('aligned_unique_objects_and_attributes3.joblib')
# general_objects_and_attributes = joblib.load('50_general_objects_and_attributes.joblib')
# general_predicates = joblib.load('30_general_predicates.joblib')
# objects_and_attributes_hierarchies = joblib.load("objects_and_attributes_hierarchies3.joblib")
# hypernyms_count = joblib.load("hypernyms_count.joblib")
# aligned_predicates = joblib.load('aligned_relationships2.joblib')


# # Calculation of similarity statistics

lower_quantile = 0.1
upper_quantile = 0.99

min_count_threshold = hypernyms_count['count'].quantile(lower_quantile)
max_count_threshold = hypernyms_count['count'].quantile(upper_quantile)
min_visual_genome_threshold = hypernyms_count['count_in_Visual_Genome'].quantile(lower_quantile)
max_visual_genome_threshold = hypernyms_count['count_in_Visual_Genome'].quantile(upper_quantile)



sns.set_theme(style="darkgrid")


plt.figure(1)
sns.histplot(hypernyms_count['count'], bins=20, kde=True)
plt.xlabel('Count')
plt.ylabel('Frequency')
# plt.title('Distribution of the best similarity for objects')
plt.xlim(hypernyms_count['count'].min(), hypernyms_count['count'].max())


log_counts = np.log10(hypernyms_count['count'] + 1)
plt.figure(2)
sns.histplot(log_counts, bins=20, kde=True)
plt.xlabel('Log10(Count)')
plt.ylabel('Frequency')
# plt.title('Log Distribution of the best similarity for objects')


plt.figure(3)
sns.histplot(hypernyms_count['count_in_Visual_Genome'], bins=20, kde=True)
plt.xlabel('Count in Visual Genome')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for objects')
plt.xlim(hypernyms_count['count_in_Visual_Genome'].min(), hypernyms_count['count_in_Visual_Genome'].max())


log_counts = np.log10(hypernyms_count['count_in_Visual_Genome'] + 1)
plt.figure(4)
sns.histplot(log_counts, bins=20, kde=True)
plt.xlabel('Log10(Count in Visual Genome)')
plt.ylabel('Frequency')
plt.title('Log Distribution of the best similarity for objects')
plt.show()






## The following code show plots representing counts of concepts in triples

hypernyms_count_df = count_strings_in_df_column(objects_and_attributes_hierarchies, 'Hypernyms')
objects_count = count_strings_in_df_column(aligned_objects, 'concept')
attributes_count = count_strings_in_df_column(aligned_attributes, 'concept')
objects_and_attributes_count = pd.concat([objects_count, attributes_count])
objects_and_attributes_count = objects_and_attributes_count.groupby('concept', as_index=False).sum()
objects_and_attributes_count = objects_and_attributes_count.sort_values(by='count', ascending=False).reset_index(drop=True)
hypernyms_count_df2 = count_strings_with_frequency(objects_and_attributes_hierarchies, 'Hypernyms')
hypernyms_count_df = hypernyms_count_df.merge(hypernyms_count_df2, on='Hypernyms', how='left')
# joblib.dump(hypernyms_count_df, 'hypernyms_count.joblib')




sns.set_theme(style="darkgrid")

plt.figure(1)
sns.histplot(aligned_unique_objects['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique objects')
plt.xlim(aligned_unique_objects['similarity'].min(), aligned_unique_objects['similarity'].max())



plt.figure(2)
sns.histplot(aligned_unique_attributes['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique attributes')
plt.xlim(aligned_unique_attributes['similarity'].min(), aligned_unique_attributes['similarity'].max())



plt.figure(3)
sns.histplot(aligned_unique_objects_and_attributes['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique objects and attributes')
plt.xlim(aligned_unique_objects_and_attributes['similarity'].min(), aligned_unique_objects_and_attributes['similarity'].max())


plt.figure(4)
sns.histplot(aligned_unique_predicates['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique predicates')
plt.xlim(aligned_unique_predicates['similarity'].min(), aligned_unique_predicates['similarity'].max())
plt.show()



## Now the plot excluding the similarity value equal to 1 (since they can be biased towards entities with an unique concept aligned in Conceptnet)


aligned_unique_objects = aligned_unique_objects[aligned_unique_objects['similarity'] < 1]
aligned_unique_attributes = aligned_unique_attributes[aligned_unique_attributes['similarity'] < 1]
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes[aligned_unique_objects_and_attributes['similarity'] < 1]
aligned_unique_predicates = aligned_unique_predicates[aligned_unique_predicates['similarity'] < 1]


plt.figure(1)
sns.histplot(aligned_unique_objects['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique objects (excluding values equals to 1)')
plt.xlim(aligned_unique_objects['similarity'].min(), aligned_unique_objects['similarity'].max())


plt.figure(2)
sns.histplot(aligned_unique_attributes['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique attributes (excluding values equals to 1)')
plt.xlim(aligned_unique_attributes['similarity'].min(), aligned_unique_attributes['similarity'].max())


plt.figure(3)
sns.histplot(aligned_unique_objects_and_attributes['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique objects and attributes (excluding values equals to 1)')
plt.xlim(aligned_unique_objects_and_attributes['similarity'].min(), aligned_unique_objects_and_attributes['similarity'].max())


plt.figure(4)
sns.histplot(aligned_unique_predicates['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique predicates (excluding values equals to 1)')
plt.xlim(aligned_unique_predicates['similarity'].min(), aligned_unique_predicates['similarity'].max())
plt.show()







# Script contained in statistics_generation2


aligned_objects = joblib.load('aligned_objects2.joblib')
aligned_attributes = joblib.load('aligned_attributes2.joblib')
aligned_unique_objects = joblib.load('aligned_unique_objects3.joblib')
aligned_unique_predicates = joblib.load('aligned_unique_predicates3.joblib')
aligned_unique_attributes = joblib.load('aligned_unique_attributes3.joblib')
aligned_unique_objects_and_attributes = joblib.load('aligned_unique_objects_and_attributes3.joblib')
general_objects_and_attributes = joblib.load('50_general_objects_and_attributes.joblib')
general_predicates = joblib.load('30_general_predicates.joblib')
objects_and_attributes_hierarchies = joblib.load("objects_and_attributes_hierarchies3.joblib")
hypernyms_count = joblib.load("hypernyms_count.joblib")
aligned_relationships = joblib.load('aligned_relationships2.joblib')

#  Example
# hypernyms_count_df = count_strings_in_df_column(objects_and_attributes_hierarchies, 'Hypernyms')



# Already calculated the following counts

objects_count = count_strings_in_df_column(aligned_objects, 'concept')
attributes_count = count_strings_in_df_column(aligned_attributes, 'concept')
objects_and_attributes_count = pd.concat([objects_count, attributes_count])

objects_and_attributes_count = objects_and_attributes_count.groupby('concept', as_index=False).sum()
objects_and_attributes_count = objects_and_attributes_count.sort_values(by='count', ascending=False).reset_index(drop=True)

hypernyms_count_df2 = count_strings_with_frequency(objects_and_attributes_hierarchies, 'Hypernyms')
hypernyms_count_df = hypernyms_count_df.merge(hypernyms_count_df2, on='Hypernyms', how='left')
# joblib.dump(hypernyms_count_df, 'hypernyms_count.joblib')




# The folowing code make the plots of the distribution of the best similarity for Visual Genome unique entities
sns.set_theme(style="darkgrid")


plt.figure(1)
sns.histplot(aligned_unique_objects['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique objects')
plt.xlim(aligned_unique_objects['similarity'].min(), aligned_unique_objects['similarity'].max())


plt.figure(2)
sns.histplot(aligned_unique_attributes['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique attributes')
plt.xlim(aligned_unique_attributes['similarity'].min(), aligned_unique_attributes['similarity'].max())


plt.figure(3)
sns.histplot(aligned_unique_objects_and_attributes['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique objects and attributes')
plt.xlim(aligned_unique_objects_and_attributes['similarity'].min(), aligned_unique_objects_and_attributes['similarity'].max())



plt.figure(4)
sns.histplot(aligned_unique_predicates['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
# plt.title('Distribution of the best similarity for predicates')
plt.xlim(aligned_unique_predicates['similarity'].min(), aligned_unique_predicates['similarity'].max())
plt.show()


aligned_unique_objects = aligned_unique_objects[aligned_unique_objects['similarity'] < 1]
aligned_unique_attributes = aligned_unique_attributes[aligned_unique_attributes['similarity'] < 1]
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes[aligned_unique_objects_and_attributes['similarity'] < 1]
aligned_unique_predicates = aligned_unique_predicates[aligned_unique_predicates['similarity'] < 1]



plt.figure(1)
sns.histplot(aligned_unique_objects['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique objects (only those lower than 1)')
plt.xlim(aligned_unique_objects['similarity'].min(), aligned_unique_objects['similarity'].max())


plt.figure(2)
sns.histplot(aligned_unique_attributes['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique attributes (only those lower than 1)')
plt.xlim(aligned_unique_attributes['similarity'].min(), aligned_unique_attributes['similarity'].max())


plt.figure(3)
sns.histplot(aligned_unique_objects_and_attributes['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique objects and attributes (only those lower than 1)')
plt.xlim(aligned_unique_objects_and_attributes['similarity'].min(), aligned_unique_objects_and_attributes['similarity'].max())


plt.figure(4)
sns.histplot(aligned_unique_predicates['similarity'], bins=20, kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of the best similarity for unique predicates (only those lower than 1)')
plt.xlim(aligned_unique_predicates['similarity'].min(), aligned_unique_predicates['similarity'].max())
plt.show()


aligned_unique_predicates = calculate_alignment_metrics(aligned_unique_predicates)
column_to_move = 'top_5_concepts'
aligned_unique_predicates.insert(3, column_to_move, aligned_unique_predicates.pop(column_to_move))


aligned_unique_objects = calculate_alignment_metrics(aligned_unique_objects)
column_to_move = 'top_5_concepts'
aligned_unique_objects.insert(3, column_to_move, aligned_unique_objects.pop(column_to_move))


aligned_unique_attributes = calculate_alignment_metrics(aligned_unique_attributes)
column_to_move = 'top_5_concepts'
aligned_unique_attributes.insert(3, column_to_move, aligned_unique_attributes.pop(column_to_move))


aligned_unique_objects_and_attributes = calculate_alignment_metrics(aligned_unique_objects_and_attributes)
column_to_move = 'top_5_concepts'
aligned_unique_objects_and_attributes.insert(3, column_to_move, aligned_unique_objects_and_attributes.pop(column_to_move))


objects_alignment_performance = calculate_mean_std_alignment_metrics(aligned_unique_objects)
attributes_alignment_performance = calculate_mean_std_alignment_metrics(aligned_unique_attributes)
objects_and_attributes_alignment_performance = calculate_mean_std_alignment_metrics(aligned_unique_objects_and_attributes)


# aligned_objects_and_attributes = joblib.load("aligned_objects_and_attributes.joblib")


aligned_objects_and_attributes['synsets'] = aligned_objects_and_attributes['synsets'].apply(
    lambda x: tuple(x) if isinstance(x, (list, tuple, str)) else (x,))


aligned_relationships = aligned_relationships.drop(columns = ['object_name', 'subject_name', 'top_5_concepts', 'top_5_similarities'])
columns = ['object_synsets', 'subject_synsets','synsets']
for column in columns:
    aligned_relationships[column] = aligned_relationships[column].apply(
        lambda x: tuple(x) if isinstance(x, (list, tuple, str)) else (x,)
    )
aligned_relationships = aligned_relationships.drop_duplicates()
aligned_relationships = aligned_relationships.merge(
    aligned_objects_and_attributes,
    left_on=['subject_id', 'subject_x', 'subject_h', 'subject_w', 'subject_y', 'subject_synsets'],
    right_on=['object_id', 'x', 'h', 'w', 'y',  'synsets'],
    how='inner'
)

aligned_relationships = aligned_relationships.drop(columns=['object_id_y', 'synsets_y', 'h', 'w', 'y', 'x'])
aligned_relationships = aligned_relationships.rename(columns={'concept_y': 'subject_concept', 'similarity_y': 'subject_similarity', 'object_id_x': 'object_id', 'synsets_x': 'predicate_synsets', 'concept_x': 'predicate_concept', 'similarity_x': 'predicate_similarity'})


aligned_relationships = aligned_relationships.merge(
    aligned_objects_and_attributes,
    left_on=['object_id', 'object_x', 'object_h', 'object_w', 'object_y', 'object_synsets'],
    right_on=['object_id', 'x', 'h', 'w', 'y',  'synsets'],
    how='inner'
)

aligned_relationships = aligned_relationships.drop(columns=['synsets', 'h', 'w', 'y', 'x'])
aligned_relationships = aligned_relationships.rename(columns={'concept': 'object_concept', 'similarity': 'object_similarity'})

different_image_ids = aligned_relationships[aligned_relationships['image_id_x'] != aligned_relationships['image_id_y']]
aligned_relationships = aligned_relationships.drop('image_id_x', axis=1)
aligned_relationships = aligned_relationships.rename(columns={'image_id_y': 'image_id'})

# joblib.dump(aligned_relationships, "aligned_predicates2.joblib")

predicates_hierarchies = joblib.load("predicates_Conceptnet_relationships_count.joblib")
aligned_predicates = joblib.load("aligned_predicates2.joblib")
positive_range_fol_axioms = joblib.load("positive_range_fol_axioms.joblib")
positive_domain_fol_axioms = joblib.load("positive_domain_fol_axioms.joblib")
positive_domain_using_capable_of_fol_axioms = joblib.load("positive_domain_using_capable_of_fol_axioms.joblib")
negative_domain_using_not_capable_of_fol_axioms = joblib.load("negative_domain_using_not_capable_of_fol_axioms.joblib")
hypernyms_fol_axioms = joblib.load("hypernyms_fol_axioms.joblib")
hypernyms_fol_axioms2 = joblib.load("hypernyms_fol_axioms2.joblib")
negative_axioms_with_predicates = joblib.load("negative_axioms_with_predicates.joblib")
negative_axioms_with_predicates_antonyms = joblib.load("negative_axioms_with_predicates_antonyms.joblib")
negative_axioms_with_general_objects_and_attributes = joblib.load("negative_axioms_with_general_objects_and_attributes.joblib")
equivalence_fol_axioms_for_objects_and_attributes = joblib.load("equivalence_fol_axioms_for_objects_and_attributes.joblib")
equivalence_fol_axioms_for_predicates = joblib.load("equivalence_fol_axioms_for_predicates.joblib")


# relationships_full = joblib.load("relationships_full.joblib")
# relationships_full = relationships_full.head(100)
# relationships_full = relationships_full.merge(aligned_unique_objects_and_attributes, left_on='subject_synsets', right_on='synsets')
# rows_with_none = scene_graphs[scene_graphs.isna().any(axis=1)]

# Function to count occurrences of predicate_concept in the left-hand side (LHS) of axioms



# The following code counts, for each predicates, the number of axioms containing it (for each typology of axioms)

positive_domain_fol_axioms = pd.DataFrame(positive_domain_fol_axioms)
positive_domain_fol_axioms = positive_domain_fol_axioms.dropna().drop_duplicates()

positive_range_fol_axioms = pd.DataFrame(positive_range_fol_axioms)
positive_range_fol_axioms = positive_range_fol_axioms.dropna().drop_duplicates()

positive_domain_using_capable_of_fol_axioms = pd.DataFrame(positive_domain_using_capable_of_fol_axioms)
positive_domain_using_capable_of_fol_axioms = positive_domain_using_capable_of_fol_axioms.dropna().drop_duplicates()

negative_domain_using_not_capable_of_fol_axioms = pd.DataFrame(negative_domain_using_not_capable_of_fol_axioms)
negative_domain_using_not_capable_of_fol_axioms = negative_domain_using_not_capable_of_fol_axioms.dropna().drop_duplicates()

negative_axioms_with_predicates = pd.DataFrame(negative_axioms_with_predicates)
negative_axioms_with_predicates = negative_axioms_with_predicates.dropna().drop_duplicates()

equivalence_fol_axioms_for_predicates = pd.DataFrame(equivalence_fol_axioms_for_predicates)
equivalence_fol_axioms_for_predicates = equivalence_fol_axioms_for_predicates.dropna().drop_duplicates()


aligned_predicates['positive_domain_fol_axioms_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_predicate_in_lhs(x, positive_domain_fol_axioms) if isinstance(x, (str, float)) else 0
)


aligned_predicates['positive_range_fol_axioms_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_predicate_in_lhs(x, positive_range_fol_axioms) if isinstance(x, (str, float)) else 0
)


aligned_predicates['positive_domain_using_capable_of_fol_axioms_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_predicate_in_lhs(x, positive_domain_using_capable_of_fol_axioms) if isinstance(x, (str, float)) else 0
)


aligned_predicates['negative_domain_using_not_capable_of_fol_axioms_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_predicate_in_lhs(x, negative_domain_using_not_capable_of_fol_axioms) if isinstance(x, (str, float)) else 0
)


aligned_predicates['negative_axioms_with_predicates_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_negative_axioms_for_df(x, negative_axioms_with_predicates) if isinstance(x, (str, float)) else 0
)


aligned_predicates['equivalence_fol_axioms_for_predicates_count'] = aligned_predicates['predicate_concept'].apply(
    lambda x: count_equivalence_axioms_for_df(x, equivalence_fol_axioms_for_predicates) if isinstance(x, (str, float)) else 0
)

# joblib.dump(aligned_predicates, "aligned_predicates3.joblib")
