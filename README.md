# Master's Thesis: Integrating Knowledge Graphs With Logic Tensor Networks
Knowledge graphs are representations of knowledge structured as a graph. They are made of nodes (entities) that represent objects, edges (relationships) which represent connections or associations between nodes; attributes (properties) that store additional information about nodes or edges; and weights which indicate the strength or confidence of a relationship. This thesis investigates the integration of knowledge graphs such as ConceptNet and WordNet with Logic Tensor Networks (LTNs), a neuro-symbolic framework that combines first-order fuzzy logic with neural networks, to enhance scene graph generation. 
The methodology includes aligning WordNet synsets with ConceptNet concepts and employing ConceptNet relationships like "IsA," "CapableOf," "NotCapableOf" and"Synonym", and ConceptNet embeddings (Numberbatch) to automatically generate first-order logic statements that work as input for Logic Tensor Networks, using the Visual Genome dataset as a foundation. 
The study utilizes Logic Tensor Networks to inject prior knowledge into neural networks and guide scene graph generation, using the Visual Genome dataset as a foundation. Results highlight a dense number of first-order logic statement generation (1507 for each image), with a focus on range and domain constraints for predicates. The thesis also outlines potential algorithms for expanding automatic axiom generation, though these were not implemented due to time constraints. 
This research demonstrates the value of combining symbolic reasoning with prior knowledge to improve the efficiency of AI systems in semantic image interpretation tasks. 

## Data and Code
The src directory contains utility modules, while the scripts directory holds the executable scripts. To ensure proper execution, the scripts should be run in the following order: first process_flattening.py, followed by process_alignment.py, and then the others.

The ConceptNet data is available at:
https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

The ConceptNet Numberbatch embeddings can be downloaded from:
https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz

The Visual Genome json files are available via the Visual Genome website:
https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
