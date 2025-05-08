"""
This script generates first-order logic (FOL) axioms using 
semantic embeddings (ConceptNet Numberbatch) and data generated in other scripts. 
More information about the axioms is contained in the master's thesis pdf file and the presentation.

Step-by-step overview of the script:

1. Import required libraries and FOL generation functions.
2. Load various precomputed semantic structures and hierarchies (joblib files).
3. Load Numberbatch embeddings to support concept similarity computations.
4. Generate equivalence axioms.
5. Generate negative axioms.
6. Generated ontological axioms from hierarchical structures.
7. Generate domain and range axioms.
8. Extend hypernym axioms by layering in domain, range, negative, and equivalence 
    information for each concept.
"""




import pandas as pd
from itertools import combinations
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cosine
from src.fol_axioms import load_numberbatch, generate_equivalence_fol_axioms_for_predicates, generate_equivalence_fol_axioms_for_objects_and_attributes, generate_negative_axioms, generate_negative_axioms_with_objects_and_attributes, generate_negative_axioms_with_predicates_antonyms, generate_negative_axioms_with_predicates, generate_hypernyms_fol_axioms_extended, generate_hypernyms_fol_axioms, check_equivalence_fol_axioms_for_predicates, generate_positive_domain_fol_axioms, generate_negative_domain_fol_axioms, generate_positive_range_fol_axioms, check_negative_fol_axioms_for_predicates


general_predicates = joblib.load('30_general_predicates.joblib')
objects_and_attributes_hierarchies = joblib.load('objects_and_attributes_hierarchies3.joblib')
range_and_domain_classes_for_general_predicates = joblib.load('range_and_domain_classes_for_general_predicates2.joblib')
# general_hypernyms = joblib.load('general_hypernyms.joblib')
positive_domain_and_range_for_predicates = joblib.load("positive_domain_and_range_for_predicates.joblib")
general_hypernyms_count = joblib.load('general_hypernyms_count.joblib')
general_hypernyms_hierarchies = joblib.load("general_hypernyms_hierarchies.joblib")
semantic_and_functional_relations_general_predicates = joblib.load("semantic_and_functional_relations_general_predicates2.joblib")
semantic_and_functional_relations_general_predicates_with_hierarchies = joblib.load("semantic_and_functional_relations_general_predicates_with_hierarchies.joblib")
concepts_semantically_related_to_objects_and_attributes = joblib.load("concepts_semantically_related_to_objects_and_attributes.joblib")
semantic_relations_predicates = joblib.load("semantic_relations_predicates2.joblib")



file_path = 'numberbatch.joblib'
embeddings_dict = load_numberbatch(file_path)
concepts = list(embeddings_dict.keys())
vectors = np.array(list(embeddings_dict.values()), dtype=np.float16)
admissible_hypernyms = list(general_hypernyms_count['Hypernyms'])

# Generation of equivalence first-order logic axioms for predicates
equivalence_fol_axioms_for_predicates = generate_equivalence_fol_axioms_for_predicates(semantic_and_functional_relations_general_predicates, embeddings_dict)
# joblib.dump(equivalence_fol_axioms_for_predicates, "equivalence_fol_axioms_for_predicates.joblib")

# Generation of equivalence first-order logic axioms for objects and attributes
concepts_semantically_related_to_objects_and_attributes = pd.merge(concepts_semantically_related_to_objects_and_attributes, objects_and_attributes_hierarchies, on="concept", how="inner")
equivalence_fol_axioms_for_objects_and_attributes = generate_equivalence_fol_axioms_for_objects_and_attributes(concepts_semantically_related_to_objects_and_attributes, embeddings_dict)
# joblib.dump(equivalence_fol_axioms_for_objects_and_attributes, "equivalence_fol_axioms_for_objects_and_attributes.joblib")


# Generation of negative first-order logic axioms for general objects and attributes
negative_axioms_with_general_objects_and_attributes = generate_negative_axioms(general_hypernyms_hierarchies, embeddings_dict)
# joblib.dump(negative_axioms_with_general_objects_and_attributes, "negative_axioms_with_general_objects_and_attributes.joblib")



negative_axioms_with_general_objects_and_attributes = generate_negative_axioms_with_objects_and_attributes(general_hypernyms_hierarchies, embeddings_dict)
# joblib.dump(negative_axioms_with_general_objects_and_attributes, "negative_axioms_with_general_objects_and_attributes.joblib")


# Generation of negative first-order logic axioms for predicates
negative_axioms_with_predicates_antonyms = generate_negative_axioms_with_predicates_antonyms(range_and_domain_classes_for_general_predicates, embeddings_dict)
# joblib.dump(negative_axioms_with_predicates_antonyms, "negative_axioms_with_predicates_antonyms.joblib")


negative_axioms_with_predicates = generate_negative_axioms_with_predicates(range_and_domain_classes_for_general_predicates, embeddings_dict)
# joblib.dump(negative_axioms_with_predicates, "negative_axioms_with_predicates.joblib")


hypernyms_fol_axioms = generate_hypernyms_fol_axioms(objects_and_attributes_hierarchies, admissible_hypernyms = list(general_hypernyms_count['Hypernyms']))
hypernyms_fol_axioms2 = generate_hypernyms_fol_axioms(general_hypernyms_hierarchies, admissible_hypernyms = list(general_hypernyms_count['Hypernyms']))

# joblib.dump(hypernyms_fol_axioms, "hypernyms_fol_axioms.joblib")
# joblib.dump(hypernyms_fol_axioms2, "hypernyms_fol_axioms2.joblib")



positive_domain_fol_axioms = generate_positive_domain_fol_axioms(positive_domain_and_range_for_predicates, 'positive_domain_with_filtered_hypernyms')
positive_range_fol_axioms = generate_positive_range_fol_axioms(positive_domain_and_range_for_predicates, 'positive_range_with_filtered_hypernyms')

# negative_domain_fol_axioms = generate_negative_domain_fol_axioms(range_and_domain_classes_for_general_predicates, 'negative_domain_with_filtered_hypernyms')
# negative_range_fol_axioms = generate_negative_range_fol_axioms(range_and_domain_classes_for_general_predicates, 'negative_domain_with_filtered_hypernyms')


# joblib.dump(positive_domain_fol_axioms, "positive_domain_fol_axioms_limited.joblib")
# joblib.dump(positive_range_fol_axioms, "positive_range_fol_axioms_limited.joblib")


positive_domain_using_capable_of_fol_axioms = generate_positive_domain_fol_axioms(semantic_and_functional_relations_general_predicates_with_hierarchies, 'filtered_capable_of_hypernyms')
negative_domain_using_not_capable_of_fol_axioms = generate_negative_domain_fol_axioms(semantic_and_functional_relations_general_predicates_with_hierarchies, 'filtered_not_capable_of_hypernyms')

# joblib.dump(positive_domain_using_capable_of_fol_axioms, "positive_domain_using_capable_of_fol_axioms_limited.joblib")
# joblib.dump(negative_domain_using_not_capable_of_fol_axioms, "negative_domain_using_not_capable_of_fol_axioms_limited.joblib")


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




positive_domain_fol_axioms_list = positive_domain_fol_axioms[['axiom']].dropna()
positive_domain_fol_axioms_list = positive_domain_fol_axioms_list['axiom'].tolist()


positive_range_fol_axioms_list = positive_range_fol_axioms[['axiom']].dropna()
positive_range_fol_axioms_list = positive_range_fol_axioms_list['axiom'].tolist()


positive_domain_using_capable_of_fol_axioms_list = positive_domain_using_capable_of_fol_axioms[['axiom']].dropna()
positive_domain_using_capable_of_fol_axioms_list = positive_domain_using_capable_of_fol_axioms_list['axiom'].tolist()


negative_domain_using_not_capable_of_fol_axioms_list = negative_domain_using_not_capable_of_fol_axioms[['axiom']].dropna()
negative_domain_using_not_capable_of_fol_axioms_list = negative_domain_using_not_capable_of_fol_axioms_list['axiom'].tolist()


negative_axioms_with_general_objects_and_attributes_list =  negative_axioms_with_general_objects_and_attributes[['unique_axioms']].dropna()
negative_axioms_with_general_objects_and_attributes_list =  negative_axioms_with_general_objects_and_attributes_list['unique_axioms'].tolist()


negative_axioms_with_predicates_list = negative_axioms_with_predicates[['negative_fol_axioms']]
negative_axioms_with_predicates_list = negative_axioms_with_predicates_list['negative_fol_axioms'].tolist()

negative_axioms_with_predicates_antonyms_list = negative_axioms_with_predicates_antonyms[['negative_fol_axioms']]
negative_axioms_with_predicates_antonyms_list = negative_axioms_with_predicates_antonyms_list['negative_fol_axioms'].tolist()

negative_axioms_with_predicates_list = list(set(negative_axioms_with_predicates_list).union(negative_axioms_with_predicates_antonyms_list))


equivalence_fol_axioms_for_objects_and_attributes_list = list(equivalence_fol_axioms_for_objects_and_attributes)
equivalence_fol_axioms_for_predicates_list = list(equivalence_fol_axioms_for_predicates)

hypernyms_fol_axioms_extended = generate_hypernyms_fol_axioms_extended(objects_and_attributes_hierarchies[['concept', 'Hypernyms']], positive_range_fol_axioms_list, axioms_with_hypernyms_column_name='positive_range_fol_axioms_with_hypernyms', admissible_hypernyms = list(general_hypernyms_count['Hypernyms']))

hypernyms_fol_axioms_extended = generate_hypernyms_fol_axioms_extended(hypernyms_fol_axioms_extended.drop('FOL_ontological_axioms', axis=1), positive_domain_fol_axioms_list, axioms_with_hypernyms_column_name='positive_domain_fol_axioms_with_hypernyms', admissible_hypernyms = list(general_hypernyms_count['Hypernyms']))
hypernyms_fol_axioms_extended = generate_hypernyms_fol_axioms_extended(hypernyms_fol_axioms_extended.drop('FOL_ontological_axioms', axis=1), positive_domain_using_capable_of_fol_axioms_list, axioms_with_hypernyms_column_name='positive_domain_using_capable_of_fol_axioms_list_with_hypernyms', admissible_hypernyms = list(general_hypernyms_count['Hypernyms']))
hypernyms_fol_axioms_extended = generate_hypernyms_fol_axioms_extended(hypernyms_fol_axioms_extended.drop('FOL_ontological_axioms', axis=1), negative_domain_using_not_capable_of_fol_axioms_list, axioms_with_hypernyms_column_name='negative_domain_using_not_capable_of_fol_axioms_list_with_hypernyms', admissible_hypernyms = list(general_hypernyms_count['Hypernyms']))


hypernyms_fol_axioms_extended = generate_hypernyms_fol_axioms_extended(hypernyms_fol_axioms_extended.drop('FOL_ontological_axioms', axis=1), negative_axioms_with_general_objects_and_attributes_list, axioms_with_hypernyms_column_name='negative_axioms_with_general_objects_and_attributes_list_with_hypernyms', admissible_hypernyms = list(general_hypernyms_count['Hypernyms']))
hypernyms_fol_axioms_extended = generate_hypernyms_fol_axioms_extended(hypernyms_fol_axioms_extended.drop('FOL_ontological_axioms', axis=1), equivalence_fol_axioms_for_objects_and_attributes_list, axioms_with_hypernyms_column_name='equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms', admissible_hypernyms = list(general_hypernyms_count['Hypernyms']))
hypernyms_fol_axioms_extended['equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms'] = hypernyms_fol_axioms_extended.apply(
lambda row: [axiom for axiom in row['equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms'] if row['concept'] in axiom],
 axis=1)
hypernyms_fol_axioms_extended = hypernyms_fol_axioms_extended.drop(columns= {'mathcing_equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms', 'matching_equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms_count'})
hypernyms_fol_axioms_extended['matching_equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms_count'] = hypernyms_fol_axioms_extended['equivalence_fol_axioms_for_objects_and_attributes_list_with_hypernyms'].apply(
lambda x: len(x) if x else 0)

# joblib.dump(hypernyms_fol_axioms_extended, "hypernyms_fol_axioms_extended2.joblib")


hypernyms_fol_axioms_extended = joblib.load("hypernyms_fol_axioms_extended2.joblib")