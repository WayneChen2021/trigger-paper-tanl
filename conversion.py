import argparse
import json
import ast
from nltk.tokenize import TreebankWordTokenizer as tbwt
from itertools import product
from functools import reduce
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
            new_example = deepcopy(base_example)
            for template_ind, ind in enumerate(trigger_set):
                trigger_tup = container['triggers'][template_ind][ind]
                new_example['triggers'].append(build_entity(
                    f"trigger for {container['incident_types'][template_ind]} event",
                    container['token_spans'],
                    trigger_tup[1],
                    trigger_tup[1] + len(trigger_tup[0])
                ))

            if len(new_example['triggers']) == len(set([str(e) for e in new_example['triggers']])):
                if add_trig_example:
                    trig_examples.append(deepcopy(new_example))
                if add_arg_example and len(arg_examples) < num_examples:
                    for ref_trig_index in range(len(new_example['triggers'])):
                        new_example_copy = deepcopy(new_example)
                        new_example_copy['relations'] = [{
                            "head": entity_index,
                            "tail": 0,
                            "type": f"{relation_name_map[rel_type]} for {container['incident_types'][trig_index]} event"
                        } for (entity_index, trig_index, rel_type) in filter(lambda triple : triple[1] == ref_trig_index, container['relations'])]
                    
                        arg_examples.append(new_example_copy)
    
    trigger_set = trigger_sets[0]
    for template_ind, ind in enumerate(trigger_set):
        trigger_tup = container['triggers'][template_ind][ind]
        base_example['triggers'].append(build_entity(
            f"trigger for {container['incident_types'][template_ind]}",
            container['token_spans'],
            trigger_tup[1],
            trigger_tup[1] + len(trigger_tup[0])
        ))
    
    base_example['relations'] = [{
        "head": entity_index,
        "tail": trig_index,
        "type": f"{relation_name_map[rel_type]} for {container['incident_types'][trig_index]} event"
    } for (entity_index, trig_index, rel_type) in container['relations']]

    return trig_examples, arg_examples, base_example

def main(in_file, train_trig, train_arg, train_event, test_trig, test_arg, test_event, num_trigs, span_selection, trigger_selection, event_header):
    with open(in_file, "r") as f:
        info = json.loads(f.read())
    
    event_header_len = len(event_header)
    if event_header_len:
        num_trigs = 1
    containers = {}
    for example in info.values():      
        if all(len(template['Triggers']) for template in example['templates']):
            text = event_header + example['text'].lower().replace('[', '(').replace(']', ')')
            if span_selection == "longest":
                for template in example['templates']:
                    for role, entity_lst in template.items():
                        if role not in ['Triggers', 'incident_type']:
                            new_entity_lst = [sorted(coref_list, key = lambda tup : -1 * len(tup[0])) for coref_list in entity_lst]
                            template[role] = new_entity_lst

            if event_header_len:
                earliest_entity = []
                for template in example['templates']:
                    earliest_start = len(text)
                    for role, entities_list in template.items():
                        if not role in ["incident_type", "Triggers"]:
                            for coref_spans in entities_list:
                                earliest_start = min(coref_spans[0][1], earliest_start)
                    earliest_entity.append(earliest_start)
                example['templates'] = sorted(list(enumerate(example['templates'])), key = lambda tup : earliest_entity[tup[0]])
                example['templates'] = list(map(lambda tup : tup[1], example['templates']))

                container = {
                    'text': text,
                    'token_spans': list(tbwt().span_tokenize(text)),
                    'entities': [],
                    'triggers': [
                        [[f"event {i}", text.index(f"event {i}")]]
                        for i in range(len(example['templates']))],
                    'incident_types': [template['incident_type'] for template in example['templates']],
                    'relations': []
                }
            else:
                for template in example['templates']:
                    if trigger_selection == "position":
                        template['Triggers'] = sorted(template['Triggers'], key = lambda tup : tup[1])
                
                container = {
                    'text': text,
                    'token_spans': list(tbwt().span_tokenize(text)),
                    'entities': [],
                    'triggers': [template['Triggers'] for template in example['templates']],
                    'incident_types': [template['incident_type'] for template in example['templates']],
                    'relations': []
                }

            for i, template in enumerate(example['templates']):
                for role, entity_lst in template.items():
                    if role not in ['Triggers', 'incident_type']:
                        for coref_list in entity_lst:
                            if span_selection == "longest":
                                span_tup = sorted(coref_list, key = lambda tup : len(tup[0]))
                            else:
                                span_tup = coref_list[0]
                            span_tup = (span_tup[1] + event_header_len, span_tup[1] + len(span_tup[0]) + event_header_len)
                            try:
                                entity_index = container['entities'].index(span_tup)
                            except ValueError:
                                entity_index = len(container['entities'])
                                container['entities'].append(span_tup)

                            container['relations'].append(
                                (entity_index, i, role)
                            )
            
            containers[example['id']] = container
    
    out_train_trigs, out_train_args, out_train_event = [], [], []
    out_test_trigs, out_test_args, out_test_event = [], [], []
    for message_id, container in containers.items():
        trig_examples, arg_examples, event_example = enumerate_examples(message_id, container, num_trigs)
        if 'TST' in message_id:
            out_test_trigs += trig_examples
            out_test_args += arg_examples
            out_test_event.append(event_example)
        else:
            out_train_trigs += trig_examples
            out_train_args += arg_examples
            out_train_event.append(event_example)
    
    if train_trig:
        with open(train_trig, "w") as f:
            f.write(json.dumps(out_train_trigs))
    
    if train_arg:
        with open(train_arg, "w") as f:
            f.write(json.dumps(out_train_args))

    if train_event:
        with open(train_event, "w") as f:
            f.write(json.dumps(out_train_event))
    
    if test_trig:
        with open(test_trig, "w") as f:
            f.write(json.dumps(out_test_trigs))
    
    if test_arg:
        with open(test_arg, "w") as f:
            f.write(json.dumps(out_test_args))

    if test_event:
        with open(test_event, "w") as f:
            f.write(json.dumps(out_test_event))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_train_trig", type=str, required=False)
    parser.add_argument("--out_train_arg", type=str, required=False)
    parser.add_argument("--out_train_event", type=str, required=False)
    parser.add_argument("--out_test_trig", type=str, required=False)
    parser.add_argument("--out_test_arg", type=str, required=False)
    parser.add_argument("--out_test_event", type=str, required=False)
    parser.add_argument("--num_trigs", type=int, required=False, default=1)
    parser.add_argument("--span_selection", type=str, required=False, default="earliest") # "earliest" or "longest"
    parser.add_argument("--trigger_selection", type=str, required=False, default="position") # "position" or "popularity"
    parser.add_argument("--dummy_trigs", action='store_true')
    parser.add_argument("--num_dummy_events", type=int, required=False, default=10)
    args = parser.parse_args()

    event_header = ""
    if args.dummy_trigs:
        for i in range(args.num_dummy_events):
            event_header += f"event {i} "
        event_header += "(SEP) "

    main(args.in_file, args.out_train_trig, args.out_train_arg, args.out_train_event, args.out_test_trig, args.out_test_arg, args.out_test_event, args.num_trigs, args.span_selection, args.trigger_selection, event_header)