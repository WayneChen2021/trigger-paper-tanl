import json

entity_types = set()
with open('train.json', 'r') as f:
    examples = json.loads(f.read())

for ex in examples.values():
    for template in ex['templates']:
        for k in template.keys():
            if not k in ['incident_type', 'Triggers']:
                entity_types.add(k)

print(entity_types)