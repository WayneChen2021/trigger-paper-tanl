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

def enumerate_examples(message_id, container, triggers_per_temp, relation_map, trigger_map):
    trigger_sets = list(product(*[range(min(len(sublist), 2)) for sublist in container['triggers']]))
    num_examples = 0
    if len(container['triggers']):
        num_examples = reduce(lambda x , y : x * y, [min(len(sublist), triggers_per_temp) for sublist in container['triggers']])
    
    base_example = {
        "entities": [build_entity("template entity", container['token_spans'], tup[0], tup[1]) for tup in container['entities']],
        "triggers": [],
        "relations": [],
        "tokens": [container['text'][tup[0] : tup[1]] for tup in container['token_spans']],
        "id": str(message_id)
    }
    trig_examples, arg_examples = [], {}
    for trigger_set in trigger_sets:
        add_trig_example, add_arg_example = len(trig_examples) < num_examples, len(arg_examples) < num_examples
        if add_trig_example or add_arg_example:
            new_example = deepcopy(base_example)
            for template_ind, ind in enumerate(trigger_set):
                trigger_tup = container['triggers'][template_ind][ind]
                new_example['triggers'].append(build_entity(
                    f"trigger for {trigger_map[container['incident_types'][template_ind]]} event",
                    container['token_spans'],
                    trigger_tup[1],
                    trigger_tup[1] + len(trigger_tup[0])
                ))

            if len(new_example['triggers']) == len(set([str(e) for e in new_example['triggers']])):
                if add_trig_example:
                    trig_examples.append(deepcopy(new_example))
                if add_arg_example and len(arg_examples) < num_examples:
                    for ref_trig_index, trig in enumerate(new_example['triggers']):
                        new_example_copy = deepcopy(new_example)
                        new_example_copy['relations'] = [{
                            "head": entity_index,
                            "tail": 0,
                            "type": f"{relation_map[rel_type]} for {trigger_map[container['incident_types'][trig_index]]} event"
                        } for (entity_index, trig_index, rel_type) in filter(lambda triple : triple[1] == ref_trig_index, container['relations'])]
                        new_example_copy['triggers'] = [trig]
                    
                        arg_examples[str(trig)] = new_example_copy
    
    trigger_set = trigger_sets[0]
    for template_ind, ind in enumerate(trigger_set):
        trigger_tup = container['triggers'][template_ind][ind]
        base_example['triggers'].append(build_entity(
            f"trigger for {trigger_map[container['incident_types'][template_ind]]} event",
            container['token_spans'],
            trigger_tup[1],
            trigger_tup[1] + len(trigger_tup[0])
        ))
    
    base_example['relations'] = [{
        "head": entity_index,
        "tail": trig_index,
        "type": f"{relation_map[rel_type]} for {trigger_map[container['incident_types'][trig_index]]} event"
    } for (entity_index, trig_index, rel_type) in container['relations']]

    return trig_examples, list(arg_examples.values()), base_example

def determine_split_muc(id, in_file_name):
    if 'TST3' in id or 'TST4' in id:
        return 'test'
    elif 'TST1' in id or 'TST2' in id:
        return 'dev'
    else:
        return 'train'

def determine_split_wikievent(id, in_file_name):
    if 'dev' in in_file_name:
        return 'dev'
    elif 'test' in in_file_name:
        return 'test'
    else:
        return 'train'

def main(in_file, train_trig, train_arg, train_event, test_trig, test_arg, test_event, dev_trig, dev_arg, dev_event, gtt_train, gtt_test, gtt_dev, num_trigs, span_selection, trigger_selection, event_header, relation_map, trigger_map, splitter_func):
    with open(in_file, "r") as f:
        info = json.loads(f.read())
    
    event_header_len = len(event_header)
    if event_header_len:
        num_trigs = 1
    containers = {}

    for example in info.values():  
        if all('Triggers' in template for template in example['templates']):    
            if event_header_len or all(len(template['Triggers']) for template in example['templates']):
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
    out_dev_trigs, out_dev_args, out_dev_event = [], [], []
    out_test_trigs, out_test_args, out_test_event = [], [], []
    gtt_train_events, gtt_test_events, gtt_dev_events = [], [], []
    for message_id, container in containers.items():
        trig_examples, arg_examples, event_example = enumerate_examples(message_id, container, num_trigs, relation_map, trigger_map)
        if not len(trig_examples):
            trig_examples = [deepcopy(event_example)]
            trig_examples[0]['triggers'] = []
            trig_examples[0]['entities'] = []
            trig_examples[0]['relations'] = []
        
        gtt = info[str(message_id)]
        gtt['docid'] = gtt['id']
        gtt['doctext'] = gtt['text'].lower().replace("[", "(").replace("]", ")")
        del gtt['id']
        del gtt['text']
        del gtt['source']
        for template in gtt['templates']:
            del template['Triggers']
            for role, entities in template.items():
                if role in relation_map:
                    template[role] = [[[tup[0].lower().replace("[", "(").replace("]", ")")] for tup in coref_span_lst] for coref_span_lst in entities]

        split = splitter_func(message_id, in_file)
        if split == 'test':
            out_test_trigs += trig_examples
            out_test_args += arg_examples
            out_test_event.append(event_example)
            gtt_test_events.append(gtt)
        elif split == 'dev':
            out_dev_trigs += trig_examples
            out_dev_args += arg_examples
            out_dev_event.append(event_example)
            gtt_dev_events.append(gtt)
        else:
            out_train_trigs += trig_examples
            out_train_args += arg_examples
            out_train_event.append(event_example)
            gtt_train_events.append(gtt)
    
    sort_tanl = lambda doc : doc['id']
    sort_gtt = lambda doc : doc['docid']
    out_train_trigs, out_train_args, out_train_event = sorted(out_train_trigs, key = sort_tanl), sorted(out_train_args, key = sort_tanl), sorted(out_train_event, key = sort_tanl)
    out_test_trigs, out_test_args, out_test_event = sorted(out_test_trigs, key = sort_tanl), sorted(out_test_args, key = sort_tanl), sorted(out_test_event, key = sort_tanl)
    gtt_train_events, gtt_test_events = sorted(gtt_train_events, key = sort_gtt), sorted(gtt_test_events, key = sort_gtt)
    
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
    
    if dev_trig:
        with open(dev_trig, "w") as f:
            f.write(json.dumps(out_dev_trigs))
    
    if dev_arg:
        with open(dev_arg, "w") as f:
            f.write(json.dumps(out_dev_args))

    if dev_event:
        with open(dev_event, "w") as f:
            f.write(json.dumps(out_dev_event))
    
    if gtt_train:
        with open(gtt_train, "w") as f:
            f.write(json.dumps(gtt_train_events))
    
    if gtt_test:
        with open(gtt_test, "w") as f:
            f.write(json.dumps(gtt_test_events))

    if gtt_dev:
        with open(gtt_dev, "w") as f:
            f.write(json.dumps(gtt_dev_events))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_train_trig", type=str, required=False)
    parser.add_argument("--out_train_arg", type=str, required=False)
    parser.add_argument("--out_train_event", type=str, required=False)
    parser.add_argument("--out_test_trig", type=str, required=False)
    parser.add_argument("--out_test_arg", type=str, required=False)
    parser.add_argument("--out_test_event", type=str, required=False)
    parser.add_argument("--out_dev_trig", type=str, required=False)
    parser.add_argument("--out_dev_arg", type=str, required=False)
    parser.add_argument("--out_dev_event", type=str, required=False)
    parser.add_argument("--num_trigs", type=int, required=False, default=1)
    parser.add_argument("--out_train_gtt", type=str, required=False)
    parser.add_argument("--out_test_gtt", type=str, required=False)
    parser.add_argument("--out_dev_gtt", type=str, required=False)
    parser.add_argument("--span_selection", type=str, required=False, default="earliest") # "earliest" or "longest"
    parser.add_argument("--trigger_selection", type=str, required=False, default="position") # "position" or "popularity"
    parser.add_argument("--dummy_trigs", action='store_true')
    parser.add_argument("--num_dummy_events", type=int, required=False, default=100)
    parser.add_argument("--dataset", type=str, required=True) # one of "MUC", "WikiEvents"
    args = parser.parse_args()

    if args.dataset == "MUC":
        relation_map = {
            "PerpInd": "perpetrating individual",
            "PerpOrg": "perpetrating organization",
            "Target": "target",
            "Weapon": "weapon",
            "Victim": "victim"
        }
        trigger_map = {
            "kidnapping": "kidnapping",
            "attack": "attack",
            "bombing": "bombing",
            "robbery": "robbery",
            "arson": "arson",
            "forced work stoppage": "forced work stoppage"
        }
        splitter_func = determine_split_muc
    elif args.dataset == "WikiEvents":
        relation_map = {
            "Defeated": "defeated",
            "Investigator": "investigator",
            "Killer": "killer",
            "Jailer": "jailer",
            "Destroyer": "destroyer",
            "ManufacturerAssembler": "manufacturer or assembler",
            "Instrument": "instrument",
            "PlaceOfEmployment": "place of employment",
            "Learner": "learner",
            "Components": "components",
            "Vehicle": "vehicle",
            "Disabler": "disabler",
            "Injurer": "injurer",
            "Communicator": "communicator",
            "BodyPart": "body part",
            "Disease": "disease",
            "Detainee": "detainee",
            "Position": "position",
            "Patient": "patient",
            "PassengerArtifact": "passenger or artifact",
            "Defendant": "defendant",
            "Attacker": "attacker",
            "IdentifiedRole": "identified role",
            "Preventer": "preventer",
            "CrashObject": "crash object",
            "Victim": "victim",
            "Identifier": "identifier",
            "AcquiredEntity": "acquired entity",
            "Researcher": "researcher",
            "Regulator": "regulator",
            "Observer": "observer",
            "Artifact": "artifact",
            "Target": "target",
            "Participant": "participant",
            "Recipient": "recipient",
            "Topic": "topic",
            "Impeder": "impeder",
            "Treater": "treater",
            "Subject": "subject",
            "Destination": "destination",
            "Giver": "giver",
            "Perpetrator": "perpetrator",
            "ExplosiveDevice": "explosive device",
            "Transporter": "transporter",
            "Employee": "employee",
            "PaymentBarter": "payment or barter",
            "Place": "place",
            "Damager": "damager",
            "JudgeCourt": "judge or court",
            "ArtifactMoney": "artifact or money",
            "IdentifiedObject": "identified object",
            "ObservedEntity": "observed entity",
            "Demonstrator": "demonstrator",
            "Victor": "victor",
            "TeacherTrainer": "teacher or trainer",
            "Prosecutor": "prosecutor",
            "Dismantler": "dismantler",
            "DamagerDestroyer": "damager or destroyer",
            "Origin": "origin"
        }
        trigger_map = {
            "Life.Die": "person death",
            "Contact.Contact": "communication",
            "Life.Injure": "person injured",
            "Conflict.Attack": "attack",
            "Movement.Transportation": "transportation",
            "GenericCrime.GenericCrime": "criminal activity",
            "Contact.ThreatenCoerce": "contact to threaten or coerce",
            "Personnel.EndPosition": "person leaves organization",
            "Control.ImpedeInterfereWith": "impediment or interference",
            "Justice.ArrestJailDetain": "arrest or jail with detainment",
            "Conflict.Demonstrate": "protest demonstration",
            "Contact.RequestCommand": "contact to discuss topic",
            "Justice.ChargeIndict": "charged or indicted",
            "Justice.Sentence": "sentencing",
            "Justice.TrialHearing": "tried for crime",
            "Medical.Intervention": "medical intervention",
            "Justice.InvestigateCrime": "crime investigation",
            "Cognitive.IdentifyCategorize": "identification",
            "Cognitive.Inspection": "observation",
            "Justice.ReleaseParole": "released on paraole",
            "Disaster.DiseaseOutbreak": "disease outbreak",
            "ArtifactExistence.DamageDestroyDisableDismantle": "artifact destroyed",
            "ArtifactExistence.ManufactureAssemble": "artifact assembled",
            "Justice.Convict": "convicted trial prosecution",
            "Transaction.ExchangeBuySell": "transaction",
            "Cognitive.Research": "research activity",
            "Personnel.StartPosition": "person joins organization",
            "Disaster.Crash": "vehicular crash",
            "Justice.Acquit": "acquitted trial prosecution",
            "Conflict.Defeat": "defeat",
            "Life.Infect": "infected",
            "Cognitive.TeachingTrainingLearning": "teaching",
            "Transaction.Donation": "donation"
        }
        splitter_func = determine_split_wikievent

    event_header = ""
    if args.dummy_trigs:
        for i in range(args.num_dummy_events):
            event_header += f"event {i} "
        event_header += "(SEP) "

    main(args.in_file, args.out_train_trig, args.out_train_arg, args.out_train_event, args.out_test_trig, args.out_test_arg, args.out_test_event, args.out_dev_trig, args.out_dev_arg, args.out_dev_event, args.out_train_gtt, args.out_test_gtt, args.out_dev_gtt, args.num_trigs, args.span_selection, args.trigger_selection, event_header, relation_map, trigger_map, splitter_func)