# Script to create the OWL ontology file from the env_requirements JSON
# Run this once to generate data/fishontology.owl
# Requires: pip install owlready2

import json
from owlready2 import *

# Load the JSON data (assuming it's saved as data/env_requirements.json)
with open('../data/env_requirements.json', 'r') as f:
    env_data = json.load(f)

onto = get_ontology("http://example.org/fishontology.owl")

with onto:
    class Species(Thing):
        pass

    # Data properties (functional for simplicity, as each has one value)
    class phMin(FunctionalProperty, Species >> float):
        pass
    class phMax(FunctionalProperty, Species >> float):
        pass
    class phOptimalMin(FunctionalProperty, Species >> float):
        pass
    class phOptimalMax(FunctionalProperty, Species >> float):
        pass
    class tempMin(FunctionalProperty, Species >> float):
        pass
    class tempMax(FunctionalProperty, Species >> float):
        pass
    class tempOptimalMin(FunctionalProperty, Species >> float):
        pass
    class tempOptimalMax(FunctionalProperty, Species >> float):
        pass
    class doMin(FunctionalProperty, Species >> float):
        pass
    class doOptimal(FunctionalProperty, Species >> float):
        pass

# Create individuals and set properties
for species_name, data in env_data.items():
    s = Species(species_name)
    s.phMin = data['pH']['min']
    s.phMax = data['pH']['max']
    s.phOptimalMin = data['pH']['optimal'][0]
    s.phOptimalMax = data['pH']['optimal'][1]
    s.tempMin = data['temperature_C']['min']
    s.tempMax = data['temperature_C']['max']
    s.tempOptimalMin = data['temperature_C']['optimal'][0]
    s.tempOptimalMax = data['temperature_C']['optimal'][1]
    s.doMin = data['dissolved_oxygen_mgL']['min']
    s.doOptimal = data['dissolved_oxygen_mgL']['optimal']

# Save the ontology
onto.save(file="./fishontology.owl", format="rdfxml")
print("OWL file created: data/fishontology.owl")