import time
import joblib
import logging
import src
from src.similarities_generation import find_extended_synonyms_in_df, find_extended_antonyms_in_df


logger = logging.getLogger("scripts.process_similarities_generation")

# Load the concepts
aligned_unique_objects = joblib.load('aligned_unique_objects3.joblib')
aligned_unique_predicates = joblib.load('aligned_unique_predicates3.joblib')
aligned_unique_attributes = joblib.load('aligned_unique_attributes3.joblib')
aligned_unique_objects_and_attributes = joblib.load('aligned_unique_objects_and_attributes3.joblib')
general_objects_and_attributes = joblib.load('50_general_objects_and_attributes.joblib')
general_predicates = joblib.load('30_general_predicates.joblib')


# Find synonyms and antonyms for Visual Genome general predicates
start = time.time()
semantic_relations_general_predicates = find_extended_synonyms_in_df(general_predicates)
semantic_relations_general_predicates = find_extended_antonyms_in_df(semantic_relations_general_predicates)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken to find synonyms and antonyms for Visual Genome predicates: {total_time_minutes:.4f} minutes")
# joblib.dump(semantic_relations_general_predicates, 'semantic_relations_general_predicates.joblib')


# Find synonyms and antonyms for Visual Genome predicates
start = time.time()
semantic_relations_predicates = find_extended_synonyms_in_df(aligned_unique_predicates)
semantic_relations_predicates = find_extended_antonyms_in_df(semantic_relations_predicates)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken to find synonyms and antonyms for Visual Genome predicates: {total_time_minutes:.4f} minutes")
# joblib.dump(semantic_relations_predicates, 'semantic_relations_predicates2.joblib')


# Find synonyms for Visual Genome general objects and attributes
start = time.time()
semantic_relations_general_objects_and_attributes = find_extended_synonyms_in_df(general_objects_and_attributes)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken to find synonyms for Visual Genome general objects and attributes: {total_time_minutes:.4f} minutes")
# joblib.dump(semantic_relations_general_objects_and_attributes, 'semantic_relations_general_objects_and_attributes.joblib')


# Find synonyms for Visual Genome objects and attributes
start = time.time()
semantic_relations_objects_and_attributes = find_extended_synonyms_in_df(aligned_unique_objects_and_attributes)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken to find synonyms for Visual Genome objects and attributes: {total_time_minutes:.4f} minutes")
# joblib.dump(semantic_relations_objects_and_attributes, 'semantic_relations_objects_and_attributes.joblib')


# Find synonyms for Visual Genome objects
start = time.time()
concepts_semantically_related_to_objects = find_extended_synonyms_in_df(aligned_unique_objects)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"Time taken to find synonyms for Visual Genome objects: {total_time_minutes:.4f} minutes")
concepts_semantically_related_to_objects['extended_synonyms'] = concepts_semantically_related_to_objects['extended_synonyms'].apply(lambda synonyms: [s for s in synonyms if s != '/c/en'])
# joblib.dump(concepts_semantically_related_to_objects, 'concepts_semantically_related_to_objects.joblib')
# concepts_semantically_related_to_objects.to_json('concepts_semantically_related_to_objects.json', orient='records', lines=True)



# Find synonyms for Visual Genome attributes
start = time.time()
concepts_semantically_related_to_attributes = find_extended_synonyms_in_df(aligned_unique_attributes)
end = time.time()
total_time = end - start
total_time_minutes = total_time / 60
logger.info(f"to find synonyms for Visual Genome attributes: {total_time_minutes:.4f} minutes")
concepts_semantically_related_to_attributes['extended_synonyms'] = concepts_semantically_related_to_attributes['extended_synonyms'].apply(lambda synonyms: [s for s in synonyms if s != '/c/en'])
# joblib.dump(concepts_semantically_related_to_attributes, 'concepts_semantically_related_to_attributes.joblib')
# concepts_semantically_related_to_attributes.to_json('concepts_semantically_related_to_attributes.json', orient='records', lines=True)