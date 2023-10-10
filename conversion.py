import argparse
import json
import ast
from nltk.tokenize import TreebankWordTokenizer as tbwt
from itertools import product, reduce
from copy import deepcopy

def build_entity(name, spans, head, tail):
    entity_head = -1
    entity_tail = -1
    for i, tup in enumerate(spans):
        if head >= tup[0] and head <= tup[1]:
            entity_head = i
            break
    try:
        assert entity_head != -1
    except Exception as e:
        print(head, tail)
        raise e

    for i, tup in enumerate(spans[entity_head:]):
        if tail >= tup[0] and tail <= tup[1]:
            entity_tail = entity_head + i
            break
    
    return {
        "type": name,
        "start": entity_head,
        "end": entity_tail + 1
    }

def enumerate_examples(message_id, container, triggers_per_temp):
    trigger_sets = list(product(*[range(len(sublist)) for sublist in container['triggers']]))
    num_examples = 0
    if len(container['triggers']):
        num_examples = reduce(lambda x , y : x * y, [min(len(sublist), triggers_per_temp) for sublist in container['triggers']])
    relation_name_map = {
        "PerpInd": "perpetrating individual",
        "PerpOrg": "perpetrating organization",
        "Target": "target",
        "Weapon": "weapon",
        "Victim": "victim"
    }
    
    base_example = {
        "entities": [build_entity("template entity", container['token_spans'], tup[0], tup[1]) for tup in container['entities']],
        "triggers": [],
        "relations": [],
        "tokens": [container['text'][tup[0] : tup[1]] for tup in container['token_spans']],
        "id": message_id
    }
    trig_examples, arg_examples = [], []
    for trigger_set in trigger_sets:
        add_trig_example, add_arg_example = len(trig_examples) < num_examples, len(arg_examples) < num_examples
        if add_trig_example or add_arg_example:
            for template_ind, ind in enumerate(trigger_set):
                new_example = deepcopy(base_example)
                
                trigger_tup = container['triggers'][template_ind][ind]
                new_example['triggers'].append(build_entity(
                    f"trigger for {container['incident_types'][template_ind]}",
                    container['token_spans'],
                    trigger_tup[1],
                    trigger_sets[1] + len(trigger_tup[0])
                ))
            
            if len(new_example['triggers']) == len(set([str(e) for e in new_example['triggers']])):
                if add_trig_example:
                    trig_examples.append(deepcopy(new_example))
                if add_arg_example:
                    new_example['relations'] = [{
                        "head": entity_index,
                        "tail": trig_index,
                        "type": f"{relation_name_map[rel_type]} for {container['incident_types'][trig_index]} event"
                    } for (entity_index, trig_index, rel_type) in container['relations']]
                    
                    arg_examples.append(new_example)   

    return trig_examples, arg_examples, base_example

def main(in_file, train_trig, train_arg, train_event, test_trig, test_arg, test_event, num_trigs, span_selection):
    with open(in_file, "r") as f:
        info = json.loads(f.read())
    
    containers = {}
    for example in info.values():      
        if all(len(template['Triggers']) for template in example['templates']):
            text = example['text'].lower().replace('[', '(').replace(']', ')')
            container = {
                'text': text,
                'token_spans': list(tbwt().span_tokenize(text)),
                'entities': [],
                'triggers': [template['triggers'] for template in example['templates']],
                'incident_types': [template['incident_type'] for template in example['templates']],
                'relations': []
            }

            for i, template in enumerate(example['templates']):
                for role, entity_lst in template.items():
                    if isinstance(entity_lst, list):
                        for coref_list in entity_lst:
                            if span_selection == "longest":
                                span_tup = sorted(coref_list, key = lambda tup : len(tup[0]))
                            else:
                                span_tup = coref_list[0]

                            span_tup = (span_tup[1], span_tup[1] + len(span_tup[0]))
                            try:
                                entity_index = container['entities'].index(span_tup)
                            except ValueError:
                                entity_index = len(container['entities'])
                                container['entities'].append(span_tup)

                            container['relations'].append(
                                (entity_index, i, role)
                            )
    
    out_train_trigs, out_train_args, out_train_event = [], [], []
    out_test_trigs, out_test_args, out_test_event = [], [], []
    for message_id, container in containers.items():
        trig_examples, arg_examples, event_example = enumerate_examples(message_id, container, num_trigs)
        if 'TST' in message_id:
            out_test_trigs += trig_examples
            out_test_args += arg_examples
            out_test_event += event_example
        else:
            out_train_trigs += trig_examples
            out_train_args += arg_examples
            out_train_event += event_example
    
    if train_trig:
        with open(train_trig, "w") as f:
            f.write(json.dumps(out_train_trigs))
    
    if train_arg:
        with open(train_arg, "w") as f:
            f.write(json.dumps(out_train_args))

    if train_event:
        with open(train_event, "w ") as f:
            f.write(json.dumps(out_train_event))
    
    if test_trig:
        with open(test_trig, "w") as f:
            f.write(json.dumps(out_test_trigs))
    
    if test_arg:
        with open(test_arg, "w") as f:
            f.write(json.dumps(out_test_args))

    if test_event:
        with open(test_event, "w ") as f:
            f.write(json.dumps(out_test_event))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_train_trig", type=str, required=False)
    parser.add_argument("--out_train_args", type=str, required=False)
    parser.add_argument("--out_train_event", type=str, required=False)
    parser.add_argument("--out_test_trig", type=str, required=False)
    parser.add_argument("--out_test_args", type=str, required=False)
    parser.add_argument("--out_test_event", type=str, required=False)
    parser.add_argument("--num_trigs", type=int, required=False, default=1)
    parser.add_argument("--span_selection", type=str, required=False, default="earliest") # "earliest" or "longest"
    args = parser.parse_args()

    main(args.in_file, args.out_train_trig, args.out_train_args, args.out_train_event, args.out_test_trig, args.out_test_args, args.out_test_event, args.num_trigs, args.span_selection)