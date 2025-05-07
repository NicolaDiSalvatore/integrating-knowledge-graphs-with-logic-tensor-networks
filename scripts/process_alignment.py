import pandas as pd
import logging
from pathlib import Path
from src.alignment import *
import os
import nltk
import time
import src


nltk.download('words')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")



# Set up logging
logger = logging.getLogger("scripts.process_alignment")

# Directory containing data
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
logger.info(f"Data directory: {data_dir}")


# List of JSON files to load
joblib_files = [
    # 'attributes.joblib',
    # 'attribute_synsets.joblib',
    # 'objects.joblib',
    # 'objects_attributes.joblib',
    # 'object_synsets.joblib',
    # 'qa_region_mapping_full.joblib',
    # 'qa_to_region_mapping.joblib',
    # 'question_answers.joblib',
    # 'region_descriptions.joblib',
    # 'region_descriptions_full.joblib',
    # 'region_graphs.joblib',
    # 'region_graphs_full.joblib',
    'relationships.joblib',
    # 'relationships_full.joblib',
    # 'relationship_synsets.joblib',
    # 'scene_graphs.joblib',
    # 'scene_graphs_full.joblib',
    # 'synsets.joblib'
]


loaded_data_dict = {}

# Load each .joblib file
for joblib_file in joblib_files:
    joblib_filepath = os.path.join(data_dir, joblib_file)

    data_df = joblib.load(joblib_filepath)

    # Store the loaded DataFrame in the dictionary
    var_name = joblib_file.replace('.joblib', '_data')
    loaded_data_dict[var_name] = data_df
    locals()[var_name] = data_df
    logger.info(f"Successfully loaded {joblib_file} into {var_name}")


attributes_data = loaded_data_dict['attributes_data']
attributes_data.head()

attribute_synsets_data = loaded_data_dict['attribute_synsets_data']
attribute_synsets_data.head()

objects_data = loaded_data_dict['objects_data']
objects_data.head()

object_synsets_data = loaded_data_dict['object_synsets_data']
object_synsets_data.head()

qa_to_region_mapping_data = loaded_data_dict['qa_to_region_mapping_data']
qa_to_region_mapping_data.head()

question_answers_data = loaded_data_dict['question_answers_data']
question_answers_data.head()

region_descriptions_data = loaded_data_dict['region_descriptions_data']
region_descriptions_data.head()

region_graphs_data = loaded_data_dict['region_graphs_data']
region_graphs_data.head()

relationships_data = loaded_data_dict['relationships_data']
relationships_data.head()

relationship_synsets_data = loaded_data_dict['relationship_synsets_data']
relationship_synsets_data.head()

scene_graphs_data = loaded_data_dict['scene_graphs_data']
scene_graphs_data.head()

synsets_data = loaded_data_dict['synsets_data']
synsets_data.head()




file_path = 'numberbatch.joblib'
embeddings_dict = load_numberbatch(file_path)
concepts = list(embeddings_dict.keys())
vectors = np.array(list(embeddings_dict.values()), dtype=np.float16)


# # Find the most similar concept for a WordNet synset (this is made to test the function find_similar_concept_for_synset_batch)
# start_time = time.time()
# synset_name = 'dog.n.01'
# most_similar_concept, similarity = find_similar_concept_for_synset_batch(synset_name, concepts, vectors, embeddings_dict)
# end_time = time.time()
# time_taken = end_time - start_time
# minutes, seconds = divmod(time_taken, 60)
# logger.info(f"The most similar concept in ConceptNet to '{synset_name}' is '{most_similar_concept}' with similarity {similarity:.4f}")
# logger.info(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")


# Aligning Visual Genome objects with ConceptNet
start_time = time.time()
aligned_objects = find_similar_concepts_in_dataframe(objects_data, embeddings_dict, vectors, concepts)
end_time = time.time()
time_taken = end_time - start_time
minutes, seconds = divmod(time_taken, 60)
logger.info(f"Time taken for the objects alignment: {minutes} minutes and {seconds:.2f} seconds")


# Aligning Visual Genome relationships with ConceptNet
start_time = time.time()
aligned_relationships = find_top_5_similar_concepts_in_dataframe(relationships_data, embeddings_dict, vectors, concepts)
end_time = time.time()
time_taken = end_time - start_time
minutes, seconds = divmod(time_taken, 60)
logger.info(f"Time taken for the relationships alignment: {minutes} minutes and {seconds:.2f} seconds")

# Aligning Visual Genome attributes with ConceptNet
start_time = time.time()
aligned_attributes = find_similar_concepts_in_dataframe(attributes_data, embeddings_dict, vectors, concepts)
end_time = time.time()
time_taken = end_time - start_time
minutes, seconds = divmod(time_taken, 60)
logger.info(f"Time taken for the attributes alignment: {minutes} minutes and {seconds:.2f} seconds")


# Save the DataFrame to a .joblib file
# joblib.dump(aligned_objects, 'aligned_objects.joblib')
# joblib.dump(aligned_relationships, 'aligned_relationships.joblib')
# joblib.dump(aligned_attributes, 'aligned_attributes.joblib')


# The following code save the objects alignment dropping all feature apart from concepts and the similarity
aligned_unique_objects = aligned_objects[['synsets', 'most_similar_concept', 'similarity']]
aligned_unique_objects['most_similar_concept'] = aligned_unique_objects['most_similar_concept'].apply(format_concept)
aligned_unique_objects = aligned_unique_objects.dropna(subset=['most_similar_concept'])
aligned_unique_objects['synsets'] = aligned_unique_objects['synsets'].apply(tuple)
aligned_unique_objects = aligned_unique_objects.drop_duplicates()
aligned_unique_objects['synsets'] = aligned_unique_objects['synsets'].apply(list)
aligned_unique_objects.columns = ['synsets', 'concept', 'similarity']
aligned_unique_objects = aligned_unique_objects.reset_index()
aligned_unique_objects = aligned_unique_objects.drop('index', axis=1)
# joblib.dump(aligned_unique_objects, 'unique_objects_alignment.joblib')


# The following code save the predicates alignment dropping all feature apart from concepts and the similarity
aligned_unique_predicates = aligned_relationships[['synsets', 'most_similar_concept', 'similarity']]
aligned_unique_predicates['most_similar_concept'] = aligned_unique_predicates['most_similar_concept'].apply(format_concept)
aligned_unique_predicates = aligned_unique_predicates.dropna(subset=['most_similar_concept'])
aligned_unique_predicates['synsets'] = aligned_unique_predicates['synsets'].apply(tuple)
aligned_unique_predicates = aligned_unique_predicates.drop_duplicates()
aligned_unique_predicates['synsets'] = aligned_unique_predicates['synsets'].apply(list)
aligned_unique_predicates.columns = ['synsets', 'concept', 'similarity']
aligned_unique_predicates = aligned_unique_predicates.reset_index()
aligned_unique_predicates = aligned_unique_predicates.drop('index', axis=1)
# joblib.dump(aligned_unique_predicates, 'unique_predicates_alignment.joblib')


# The following code save the attributes alignment dropping all feature apart from concepts and the similarity
aligned_unique_attributes = aligned_attributes[['synsets', 'most_similar_concept', 'similarity']]
aligned_unique_attributes['most_similar_concept'] = aligned_unique_attributes['most_similar_concept'].apply(format_concept)
aligned_unique_attributes = aligned_unique_attributes.dropna(subset=['most_similar_concept'])
aligned_unique_attributes['synsets'] = aligned_unique_attributes['synsets'].apply(tuple)
aligned_unique_attributes = aligned_unique_attributes.drop_duplicates()
aligned_unique_attributes['synsets'] = aligned_unique_attributes['synsets'].apply(list)
aligned_unique_attributes.columns = ['synsets', 'concept', 'similarity']
aligned_unique_attributes = aligned_unique_attributes.reset_index()
aligned_unique_attributes = aligned_unique_attributes.drop('index', axis=1)
# joblib.dump(aligned_unique_attributes, 'unique_attributes_alignment.joblib')


# The following code save the attributes and objects alignment dropping all feature apart from concepts and the similarity
aligned_unique_objects_and_attributes = pd.concat([aligned_unique_objects, aligned_unique_attributes], ignore_index=True)
aligned_unique_objects_and_attributes['synsets'] = aligned_unique_objects_and_attributes['synsets'].apply(tuple)
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes.drop_duplicates()
aligned_unique_objects_and_attributes['synsets'] = aligned_unique_objects_and_attributes['synsets'].apply(list)
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes.reset_index()
aligned_unique_objects_and_attributes = aligned_unique_objects_and_attributes.drop('index', axis=1)
# joblib.dump(aligned_unique_objects_and_attributes, 'unique_objects_and_attributes_alignment.joblib')




# Alignment of unique objects, attributes and relationships

start_time = time.time()
aligned_unique_objects = find_top_5_similar_concepts_in_dataframe(aligned_unique_objects, embeddings_dict, vectors, concepts)
# joblib.dump(aligned_unique_objects, 'aligned_unique_objects2.joblib')
aligned_unique_attributes = find_top_5_similar_concepts_in_dataframe(aligned_unique_attributes, embeddings_dict, vectors, concepts)
# joblib.dump(aligned_unique_attributes, 'aligned_unique_attributes2.joblib')
aligned_unique_predicates = find_top_5_similar_concepts_in_dataframe(aligned_unique_predicates, embeddings_dict, vectors, concepts)
# joblib.dump(aligned_unique_predicates, 'aligned_unique_predicates2.joblib')
end_time = time.time()
time_taken = end_time - start_time
minutes, seconds = divmod(time_taken, 60)
logger.info(f"Time taken for the alignment of unique objects, attributes and relationships: {minutes} minutes and {seconds:.2f} seconds")



