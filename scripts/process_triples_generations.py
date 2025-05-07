import joblib
from src.triples_generation import add_related_and_capable_to_concept_columns, add_related_and_capable_columns, add_related_and_capable_to_synonyms_columns, add_related_and_capable_to_antonyms_columns, format_predicate, find_relationships_with_synonyms, find_relationships_with_synonyms_complete
import logging
import pandas as pd
import src

logger = logging.getLogger("scripts.process_triples_generation")


# relationships_within_visual_genome = joblib.load('relationships_within_visual_genome.joblib')
# relationships_starting_from_visual_genome_objects = joblib.load('relationships_starting_from_visual_genome_objects.joblib')
# relationships_ending_in_visual_genome_objects = joblib.load('relationships_ending_in_visual_genome_objects.joblib')
aligned_unique_objects = joblib.load('aligned_unique_objects3.joblib')
aligned_unique_predicates = joblib.load('aligned_unique_predicates3.joblib')
aligned_vg_relationships = joblib.load('aligned_relationships2.joblib')
aligned_unique_attributes = joblib.load('aligned_unique_attributes3.joblib')
aligned_unique_objects_and_attributes = joblib.load('aligned_unique_objects_and_attributes3.joblib')
general_objects_and_attributes = joblib.load('50_general_objects_and_attributes.joblib')
general_predicates = joblib.load('30_general_predicates.joblib')
objects_and_attributes_hierarchies = joblib.load('objects_and_attributes_hierarchies3.joblib')
hypernyms_count = joblib.load('hypernyms_count.joblib')
semantic_relations_general_predicates = joblib.load('semantic_relations_general_predicates.joblib')
semantic_relations_predicates = joblib.load('concepts_semantically_related_to_predicates3.joblib')
semantic_relations_general_objects_and_attributes = joblib.load('semantic_relations_general_objects_and_attributes.joblib')



aligned_objects = joblib.load('aligned_objects.joblib')
objects_concepts = aligned_objects['most_similar_concept']
objects_concepts = objects_concepts.apply(format_predicate)
objects_concepts = objects_concepts.to_frame()
objects_concepts = objects_concepts.dropna()
objects_count = objects_concepts.value_counts()
objects_low_frequency_threshold = objects_count.quantile(0.25)
objects_count = objects_count.to_frame(name='objects_count').reset_index()
objects_count.columns = ['concept', 'count']
# objects_count['isLowFrequency'] = (objects_count['count'] <= objects_low_frequency_threshold)
objects_concepts = objects_concepts.drop_duplicates()
objects_concepts.columns = ['concept']



aligned_relationships = joblib.load('aligned_relationships.joblib')
predicates_concepts = aligned_relationships['most_similar_concept']
predicates_concepts = predicates_concepts.apply(format_predicate)
predicates_concepts = predicates_concepts.to_frame()
predicates_concepts = predicates_concepts.dropna()
predicates_count = predicates_concepts.value_counts()
predicates_low_frequency_threshold = predicates_count.quantile(0.25)
predicates_count = predicates_count.to_frame(name='predicates_count').reset_index()
predicates_count.columns = ['concept', 'count']
predicates_count['isLowFrequency'] = (predicates_count['count'] <= predicates_low_frequency_threshold)
predicates_concepts = predicates_concepts.drop_duplicates()
predicates_concepts.columns = ['concept']



aligned_attributes = joblib.load('aligned_attributes.joblib')
attributes_concepts = aligned_attributes['most_similar_concept']
attributes_concepts = attributes_concepts.apply(format_predicate)
attributes_concepts = attributes_concepts.to_frame()
attributes_concepts = attributes_concepts.dropna()
attributes_count = attributes_concepts.value_counts()
attributes_low_frequency_threshold = attributes_count.quantile(0.25)
attributes_count = attributes_count.to_frame(name='attributes_count').reset_index()
attributes_count.columns = ['concept', 'count']
attributes_count['isLowFrequency'] = (attributes_count['count'] <= attributes_low_frequency_threshold)
attributes_concepts = attributes_concepts.drop_duplicates()
attributes_concepts.columns = ['concept']


concepts_count = pd.concat([objects_count, attributes_count, predicates_count])
concepts_count = concepts_count.groupby('concept', as_index=False).sum()
low_frequency_threshold = concepts_count['count'].quantile(0.25)
concepts_count['isLowFrequency'] = (concepts_count['count'] <= low_frequency_threshold)


# Retrieve the Conceptnet relationships between couples of Visual Genome objects/attributes/relationships (including synonyms)
relationships_within_visual_genome = find_relationships_with_synonyms(concepts_count)
relationships_within_visual_genome = relationships_within_visual_genome[relationships_within_visual_genome['start'] != relationships_within_visual_genome['end']]
relationships_within_visual_genome['startIsPredicate'] = relationships_within_visual_genome['start'].isin(predicates_concepts['concept'])
relationships_within_visual_genome['endIsPredicate'] = relationships_within_visual_genome['end'].isin(predicates_concepts['concept'])
relationships_within_visual_genome = pd.merge(relationships_within_visual_genome, concepts_count, left_on='start', right_on='concept', how='left')
relationships_within_visual_genome = pd.merge(relationships_within_visual_genome, concepts_count, left_on='end', right_on='concept', how='left', suffixes=('', '_end_concept'))
relationships_within_visual_genome.rename(columns={'count': 'count_start_concept', 'isLowFrequency': 'isLowFrequency_start_concept'}, inplace=True)
relationships_within_visual_genome.drop(columns=['concept', 'concept_end_concept'], inplace=True)
# joblib.dump(relationships_within_visual_genome, 'relationships_within_visual_genome.joblib')
# relationships_within_visual_genome.to_json('relationships_within_visual_genome.json', orient='records', lines=True)


relationships_starting_from_visual_genome_objects, relationships_ending_in_visual_genome_objects = find_relationships_with_synonyms_complete(objects_concepts)
# joblib.dump(relationships_starting_from_visual_genome_objects, "relationships_starting_from_visual_genome_objects.joblib")
# joblib.dump(relationships_ending_in_visual_genome_objects, "relationships_ending_in_visual_genome_objects.joblib")
# relationships_starting_from_visual_genome_objects.to_json('relationships_starting_from_visual_genome_objects.json', orient='records', lines=True)
# relationships_ending_in_visual_genome_objects.to_json('relationships_ending_in_visual_genome_objects.json', orient='records', lines=True)



# semantic_and_functional_relations_general_predicates = add_related_and_capable_columns(semantic_relations_general_predicates)
semantic_and_functional_relations_general_predicates = add_related_and_capable_columns(aligned_unique_predicates)


semantic_and_functional_relations_general_predicates = add_related_and_capable_to_synonyms_columns(semantic_relations_general_predicates)
semantic_and_functional_relations_general_predicates = add_related_and_capable_to_concept_columns(semantic_relations_predicates)
semantic_and_functional_relations_general_predicates = add_related_and_capable_to_synonyms_columns(semantic_and_functional_relations_general_predicates)
spotted = semantic_and_functional_relations_general_predicates[semantic_and_functional_relations_general_predicates['related_to'] != semantic_and_functional_relations_general_predicates['related_to_with_synonyms']  ]
semantic_and_functional_relations_general_predicates = add_related_and_capable_to_antonyms_columns(semantic_relations_general_predicates)


# joblib.dump(semantic_and_functional_relations_general_predicates, 'semantic_and_functional_relations_general_predicates3.joblib')
semantic_and_functional_relations_general_predicates = joblib.load('semantic_and_functional_relations_general_predicates3.joblib')
