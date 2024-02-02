import json
import argparse
import os
import re
from openai import OpenAI
from copy import deepcopy

def has_filled_role(template):
    for role, fillers in template.items():
        if not role in ['incident_type', 'Triggers'] and len(fillers):
            return True
    
    return False

def find_example(annotation_dir, curr_split_examples, curr_split, event_type, docid):
    return_example, return_num = None, None
    for ex in curr_split_examples:
        for i, template in enumerate(ex['templates']):
            if template['incident_type'] == event_type and ex['docid'] != docid and has_filled_role(template) and len(template['Triggers']):
                return_example = ex
                return_num = i
                trigger_locations = [str(m.start()) for m in re.finditer(template['Triggers'][0][0][0], ex['doctext'])]
                if len(trigger_locations) > 1:
                    return ex, i
    
    for split_name in os.listdir(annotation_dir):
        if split_name != curr_split:
            with open(os.path.join(annotation_dir, split_name), 'r') as f:
                diff_split_examples = json.loads(f.read()).values()
            
            for ex in diff_split_examples:
                for i, template in enumerate(ex['templates']):
                    if template['incident_type'] == event_type and has_filled_role(template) and len(template['Triggers']):
                        return_example = ex
                        return_num = i
                        trigger_locations = [str(m.start()) for m in re.finditer(template['Triggers'][0][0][0], ex['doctext'])]
                        if len(trigger_locations) > 1:
                            return ex, i
    
    return return_example, return_num

def get_all_types(templates):
    event_types, role_types = set(), set()
    for template in templates:
        event_types.add(template['incident_type'])
        for role in template.keys():
            if not role in ['incident_type', 'Triggers']:
                role_types.add(role)
    
    return event_types, role_types

def format_example(example, template_num, describer):    
    event_types, role_types = get_all_types(example['templates'])
    arg_descriptions = []
    for role in role_types:
        arg_description = role.lower()
        if 'entity_describers' in describer and role in describer['entity_describers']:
            arg_description = describer['entity_describers'][role]
        arg_descriptions.append(f'{role} indicates a(n) {arg_description}')
    
    event_descriptions = []
    for event_type in event_types:
        event_description = event_type.lower()
        if 'event_describers' in describer and event_type in describer['event_describers']:
            event_description = describer['event_describers'][event_type]
        event_descriptions.append(f"{event_type} indicates a(n) {event_description} event")
    
    all_events = ""
    for i, template in enumerate(example['templates']):
        del template['Triggers']
        if i != template_num:
            all_events += json.dumps(template) + "\n"
    
    other_events_str = ""
    if len(all_events):
        other_events_str = f'For reference, the other events in the document are\n{all_events}'

    return f"""<start text>{example['doctext']}<end text>
Find one trigger from the above text that best indicates that the existence of the following {event_type} event with arguments. A trigger is often a verb, sometimes a noun, and can contain more than 1 word. Also output the starting index of the trigger in the text.
{example['templates'][template_num]}
{other_events_str}
For the incident_type field, {', '.join(event_descriptions)}. For the arguments, {', '.join(arg_descriptions)}."""

def format_trigger_location(text, trigger_span):
    locations = [str(m.start()) for m in re.finditer(trigger_span, text)]
    return f'"{trigger_span}" is located at {", ".join(locations)}. Which location are you referring to?'

def format_trigger_not_in_text(trigger):
    return f'"{trigger}" is not in the text. Pick another span.'

def main(annotation_dir, event_describer, gpt_version, temperature):
    openai_client = OpenAI(api_key=os.environ['OPENAI_KEY'])
    with open(event_describer, 'r') as f:
        event_describer = json.loads(f.read())
    
    for split_name in os.listdir(annotation_dir):
        with open(os.path.join(annotation_dir, split_name), 'r') as f:
            examples = json.loads(f.read()).values()
        
        for ex in examples:
            print(f'Starting {ex["docid"]} in {split_name}')
            for i, template in enumerate(ex['templates']):
                same_event_type_ex, template_num = find_example(annotation_dir, examples, split_name, template['incident_type'], ex['docid'])
                if same_event_type_ex is None:
                    print(f"No example of event type {template['incident_type']} exists with some roles filled")
                    template['Triggers'] = []
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": format_example(deepcopy(same_event_type_ex), template_num, event_describer)
                        },
                        {
                            "role": "assistant",
                            "content": same_event_type_ex['templates'][template_num]['Triggers'][0][0][0]
                        },
                        {
                            "role": "user",
                            "content": format_trigger_location(same_event_type_ex['doctext'], same_event_type_ex['templates'][template_num]['Triggers'][0][0][0])
                        },
                        {
                            "role": "assistant",
                            "content": str(same_event_type_ex['templates'][template_num]['Triggers'][0][0][1])
                        },
                        {
                            "role": "user",
                            "content": format_example(deepcopy(ex), i, event_describer)
                        }
                    ]
                    print(json.dumps(messages, indent=4))
                    1/0
                    response = openai_client.chat.completions.create(
                        model=gpt_version,
                        temperature=temperature,
                        messages=messages
                    )
                    picked_trigger_span = response.choices[0].message.content
                    messages.append({
                        "role": "assistant",
                        "content": picked_trigger_span
                    })
                    
                    while True:
                        if picked_trigger_span in ex['doctext']:
                            break
                        print(f'could not find {picked_trigger_span} for {template["incident_type"]} event type in {ex["docid"]} in {split_name}')
                        messages.append({
                            "role": "user",
                            "content": format_trigger_not_in_text(picked_trigger_span)
                        })
                        response = openai_client.chat.completions.create(
                            model=gpt_version,
                            temperature=temperature,
                            messages=messages
                        )
                        picked_trigger_span = response.choices[0].message.content
                        messages.append({
                            "role": "assistant",
                            "content": picked_trigger_span
                        })

                    print(f'found {picked_trigger_span} for {template["incident_type"]} event type in {ex["docid"]} in {split_name}')
                    messages.append({
                        "role": "user",
                        "content": format_trigger_location(ex['doctext'], picked_trigger_span)
                    })
                    response = openai_client.chat.completions.create(
                        model=gpt_version,
                        temperature=temperature,
                        messages=messages
                    )
                    ex['Triggers'] = [[[picked_trigger_span, int(response.choices[0].message.content)]]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_annotations", type=str, required=True)
    parser.add_argument("--event_describer", type=str, required=True)
    parser.add_argument("--gpt_version", type=str, required=False, default="gpt-4")
    parser.add_argument("--gpt_temp", type=float, required=False, default=0.1)
    args = parser.parse_args()

    main(args.human_annotations, args.event_describer, args.gpt_version, args.gpt_temp)