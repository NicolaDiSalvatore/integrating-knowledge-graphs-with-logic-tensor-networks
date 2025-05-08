"""
This script generates hierarchies (hypernym/hyponym structures) for aligned Visual Genome data.

Step-by-step overview of the script:

1. Loads required modules and libraries.
2. Load Visual Genome Data.
3. Extract Concept Columns.
4. Generate Hierarchies for Visual Genome data.
5. Generate Hierarchies for General Hypernyms.
6. Generate Hierarchies for Functional Relations (`related_to`, `capable_of`, `not_capable_of`)
"""

import src
import joblib
import time
import logging
from src.hierarchies_generation import find_hierarchical_concepts_batch

logger = logging.getLogger("scripts.process_hierarchies_generation")

aligned_unique_objects = joblib.load('aligned_unique_objects3.joblib')
aligned_unique_predicates = joblib.load('aligned_unique_predicates3.joblib')
aligned_unique_attributes = joblib.load('aligned_unique_attributes3.joblib')
aligned_unique_objects_and_attributes = joblib.load('aligned_unique_objects_and_attributes3.joblib')
general_objects_and_attributes = joblib.load('50_general_objects_and_attributes.joblib')
general_predicates = joblib.load('30_general_predicates.joblib')
general_hypernyms_count = joblib.load('general_hypernyms_count.joblib')
semantic_and_functional_relations_general_predicates = joblib.load('semantic_and_functional_relations_general_predicates3.joblib')


objects_concepts = aligned_unique_objects[['concept']]
predicates_concepts = aligned_unique_predicates[['concept']]
attributes_concepts = aligned_unique_attributes[['concept']]


# Generate hierarchies for objects
start = time.time()
objects_hierarchies = find_hierarchical_concepts_batch(aligned_unique_objects)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken for objects hierarchies generation: {total_time_minutes:.4f} minutes")
# joblib.dump(objects_hierarchies, 'objects_hierarchies3.joblib')
# objects_and_attributes_hierarchies.to_json('objects_hierarchies3.json', orient='records', lines=True)


start = time.time()
attributes_hierarchies = find_hierarchical_concepts_batch(attributes_concepts)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken for attributes hierarchies generation: {total_time_minutes:.4f} minutes")
# joblib.dump(attributes_hierarchies, 'attributes_hierarchies.joblib')
# attributes_hierarchies.to_json('attributes_hierarchies.json', orient='records', lines=True)


# Generate hierarchies for objects and attributes
start = time.time()
objects_and_attributes_hierarchies = find_hierarchical_concepts_batch(aligned_unique_objects_and_attributes)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken for objects and attributes hierarchies generation: {total_time_minutes:.4f} minutes")
# joblib.dump(objects_and_attributes_hierarchies, 'objects_and_attributes_hierarchies3.joblib')
# objects_and_attributes_hierarchies.to_json('objects_and_attributes_hierarchies3.json', orient='records', lines=True)


# Generate hierarchies for predicates
start = time.time()
predicates_hierarchies = find_hierarchical_concepts_batch(predicates_concepts)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken for predicates hierarchies generation: {total_time_minutes:.4f} minutes")
# joblib.dump(predicates_hierarchies, 'predicates_hierarchies.joblib')
# predicates_hierarchies.to_json('predicates_hierarchies.json', orient='records', lines=True)
# predicates_hierarchies.to_csv("predicates_hierarchies.csv", sep=';')


# Generate hierarchies for general hypernyms
general_hypernyms_count.rename(columns={'Hypernyms': 'concept'}, inplace=True)
general_hypernyms_hierarchies = find_hierarchical_concepts_batch(general_hypernyms_count)
# joblib.dump(general_hypernyms_hierarchies, "general_hypernyms_hierarchies.joblib")

# spotted = semantic_and_functional_relations_general_predicates[semantic_and_functional_relations_general_predicates['related_to'] != semantic_and_functional_relations_general_predicates['related_to_with_synonyms']]


# Generate hierarchies for concepts in related_to, capable_of and not_capable_of columns
semantic_and_functional_relations_general_predicates_with_hierarchies = find_hierarchical_concepts_batch(semantic_and_functional_relations_general_predicates, concept_column= 'related_to', hypernyms_column='related_to_hypernyms', hyponyms_column = 'related_to_hyponyms')
semantic_and_functional_relations_general_predicates_with_hierarchies = find_hierarchical_concepts_batch(semantic_and_functional_relations_general_predicates_with_hierarchies, concept_column= 'capable_of', hypernyms_column='capable_of_hypernyms', hyponyms_column = 'capable_of_hyponyms')
semantic_and_functional_relations_general_predicates_with_hierarchies = find_hierarchical_concepts_batch(semantic_and_functional_relations_general_predicates_with_hierarchies, concept_column= 'not_capable_of', hypernyms_column='not_capable_of_hypernyms', hyponyms_column = 'not_capable_of_hyponyms')
# joblib.dump(semantic_and_functional_relations_general_predicates_with_hierarchies, "semantic_and_functional_relations_general_predicates_with_hierarchies.joblib")
