import numpy as np
import argparse
import json
from scipy.optimize import linear_sum_assignment

def is_match(str1, str2, relax):
    if str1 == str2:
        return True

    if relax and (str1 in str2 or str2 in str1):
        return True

    return False

def entities_match(pred_entity, gold_entity, relax):
    if len(pred_entity) == 0:
        return len(gold_entity) == 0
    else:
        if len(gold_entity) == 0:
            return False
        
        matched = False
        for pred_mention in pred_entity:
            matched = matched or any(is_match(pred_mention, gold_mention[0], relax) for gold_mention in gold_entity)
        
        return matched

def compute_matches(pred_template, gold_template, relax):
    """
    Returns:
        1. whether or not event types match
        2. total number of matches per role type (as a dict)
        3. total number of predicted elements per role type (as a dict)
        4. total number of gold elements per role type (as a dict)
        5. total number of matches across all roles and including event type (but not triggers)
    """
    event_type_matches = pred_template['incident_type'] == gold_template['incident_type']
    matches_per_role = {}
    predicted_per_role = {}
    gold_per_role = {}
    total_matches = int(event_type_matches)
    for role_type, pred_entities in pred_template.items():
        if role_type != 'incident_type':
            gold_entities = gold_template[role_type]
            if role_type == 'Triggers':
                if len(gold_entities) == 0:
                    matches_per_role[role_type] = 0
                    predicted_per_role[role_type] = 0
                    gold_per_role[role_type] = 0
                else:
                    match = False
                    for pred_entity in pred_entities:
                        for gold_entity in gold_entities:
                            match = match or entities_match(pred_entity, gold_entity, relax)

                    matches_per_role[role_type] = int(match)
                    predicted_per_role[role_type] = 1
                    gold_per_role[role_type] = 1
            else:
                matr_size = max(len(gold_entities), len(pred_entities))
                cost_matr = np.zeros((matr_size, matr_size))

                for i, pred_entity in enumerate(pred_entities):
                    for j, gold_entity in enumerate(gold_entities):
                        cost_matr[i][j] = -1 * int(entities_match(pred_entity, gold_entity, relax))
                
                row_ind, col_ind = linear_sum_assignment(cost_matr)
                matches = int(-1 * cost_matr[row_ind, col_ind].sum())

                matches_per_role[role_type] = matches
                predicted_per_role[role_type] = len(pred_entities)
                gold_per_role[role_type] = len(gold_entities)
                if role_type != 'Triggers':
                    total_matches += matches

    return event_type_matches, matches_per_role, predicted_per_role, gold_per_role, total_matches

def non_zero(value):
    if value == 0:
        return 1
    
    return value

def calc_f1(precision, recall):
    return 2 * precision * recall / non_zero(precision + recall)

def main(config, in_file, relax_match, out_file, filter_lst):
    with open(config, 'r') as f:
        config = json.loads(f.read())
    
    with open(in_file, 'r') as f:
        predictions = json.loads(f.read())
    
    if filter_lst:
        with open(filter_lst, 'r') as f:
            filter_lst = json.loads(f.read())
    else:
        filter_lst = []
    
    num_matches, r_d, p_d = 0, 0, 0
    event_type_n, event_type_p_d, event_type_r_d = 0, 0, 0
    arg_n, arg_p_d, arg_r_d = 0, 0, 0
    event_type_per_class = {event_type : [0, 0, 0] for event_type in config['event_type_names']} # num, precision den, recall den
    arg_per_class = {role: [0, 0, 0] for role in config['role_names']}
    for example in predictions.values():
        gold_temps, pred_temps = example['gold_templates'], example['pred_templates']
        matr_size = max(len(gold_temps), len(pred_temps))
        cost_matr = np.zeros((matr_size, matr_size))

        match_info = {}
        for i, gold_temp in enumerate(gold_temps):
            for j, pred_temp in enumerate(pred_temps):
                event_type_matches, matches_per_role, predicted_per_role, gold_per_role, total_matches = compute_matches(
                    pred_temp, gold_temp, relax_match
                )
                cost_matr[i][j] = -1 * total_matches
                match_info[(i, j)] = {
                    "gold_event_type": gold_temp['incident_type'],
                    "pred_event_type": pred_temp['incident_type'],
                    "event_type_matches": event_type_matches,
                    "matches_per_role": matches_per_role,
                    "predicted_per_role": predicted_per_role,
                    "gold_per_role": gold_per_role
                }
        
        row_ind, col_ind = linear_sum_assignment(cost_matr)
        num_matches += int(-1 * cost_matr[row_ind, col_ind].sum())
        for i, j in zip(row_ind, col_ind):
            to_process = False
            if len(filter_lst):
                if isinstance(filter_lst[0], str):
                    to_process = example['docid'] in filter_lst
                else:
                    to_process = [example['docid'], i] in filter_lst
            else:
                to_process = True

            if to_process:
                if i < len(gold_temps) or j < len(pred_temps):
                    if i < len(gold_temps) and j < len(pred_temps): # matched gold and pred template
                        infos = match_info[(i, j)]
                        event_match_as_int = int(infos['event_type_matches'])
                        event_type_n += event_match_as_int
                        event_type_p_d += 1
                        event_type_r_d += 1
                        event_type_per_class[infos['pred_event_type']][0] += event_match_as_int
                        event_type_per_class[infos['gold_event_type']][2] += 1
                        event_type_per_class[infos['pred_event_type']][1] += 1
                        
                        total_predicted_args_num = sum(infos['predicted_per_role'].values()) - (infos['predicted_per_role']['Triggers'] if 'Triggers' in infos['predicted_per_role'] else 0)
                        total_gold_args_num = sum(infos['gold_per_role'].values()) - (infos['gold_per_role']['Triggers'] if 'Triggers' in infos['gold_per_role'] else 0)
                        arg_n += sum(infos['matches_per_role'].values()) - (infos['matches_per_role']['Triggers'] if 'Triggers' in infos['matches_per_role'] else 0)
                        arg_p_d += total_predicted_args_num
                        arg_r_d += total_gold_args_num
                        for arg_type in config['role_names']:
                            arg_per_class[arg_type][0] += infos['matches_per_role'][arg_type]
                            arg_per_class[arg_type][1] += infos['predicted_per_role'][arg_type]
                            arg_per_class[arg_type][2] += infos['gold_per_role'][arg_type]
                        
                        r_d += 1 + total_predicted_args_num
                        p_d += 1 + total_gold_args_num
                    
                    elif i >= len(gold_temps) and j < len(pred_temps): # extra pred template
                        event_type_p_d += 1
                        pred_temp = pred_temps[j]
                        event_type_per_class[pred_temp['incident_type']][1] += 1

                        total_predicted_args_num = 0
                        for arg_type in config['role_names']:
                            num_entities = len(pred_temp[arg_type])
                            arg_per_class[arg_type][1] += num_entities
                            if arg_type != 'Triggers':
                                total_predicted_args_num += num_entities
                        arg_p_d += total_predicted_args_num

                        p_d += 1 + total_predicted_args_num
                            
                    elif i < len(gold_temps) and j >= len(pred_temps): # extra gold template
                        event_type_r_d += 1
                        gold_temp = gold_temps[i]
                        event_type_per_class[gold_temp['incident_type']][2] += 1

                        total_gold_args_num = 0
                        for arg_type in config['role_names']:
                            num_entities = len(gold_temp[arg_type])
                            arg_per_class[arg_type][2] += num_entities
                            if arg_type != 'Triggers':
                                total_gold_args_num += num_entities
                        arg_r_d += total_gold_args_num

                        r_d += 1 + total_gold_args_num
            else:
                num_matches += cost_matr[i][j]
        
    overall_precision, overall_recall = num_matches / non_zero(p_d), num_matches / non_zero(r_d)
    overall_f1 = calc_f1(overall_precision, overall_recall)
    event_type_precision, event_type_recall = event_type_n / non_zero(event_type_p_d), event_type_n / non_zero(event_type_r_d)
    event_type_f1 = calc_f1(event_type_precision, event_type_recall)
    arg_precision, arg_recall = arg_n / non_zero(arg_p_d), arg_n / non_zero(arg_r_d)
    arg_f1 = calc_f1(arg_precision, arg_recall)
    event_type_class_precisions = {event_type : lst[0] / non_zero(lst[1]) for event_type, lst in event_type_per_class.items()}
    event_type_class_recalls = {event_type : lst[0] / non_zero(lst[2]) for event_type, lst in event_type_per_class.items()}
    event_type_class_f1 = {event_type: calc_f1(event_type_class_precisions[event_type], event_type_class_recalls[event_type]) for event_type in config['event_type_names']}
    arg_type_class_precisions = {arg_type : lst[0] / non_zero(lst[1]) for arg_type, lst in arg_per_class.items()}
    arg_type_class_recalls = {arg_type : lst[0] / non_zero(lst[2]) for arg_type, lst in arg_per_class.items()}
    arg_type_class_f1 = {arg_type: calc_f1(arg_type_class_precisions[arg_type], arg_type_class_recalls[arg_type]) for arg_type in config['role_names']}

    info_dic = {
        "overall_f1": overall_f1,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "event_type_f1": event_type_f1,
        "event_type_precision": event_type_precision,
        "event_type_recall": event_type_recall,
        "argument_f1": arg_f1,
        "argument_precision": arg_precision,
        "argument_recall": arg_recall,
        "event_type_metrics_per_class": {
            class_type : {
                "f1": event_type_class_f1[class_type],
                "precision": event_type_class_precisions[class_type],
                "recall": event_type_class_recalls[class_type]
            }
            for class_type in config['event_type_names']
        },
        "argument_type_metrics_per_class": {
            class_type : {
                "f1": arg_type_class_f1[class_type],
                "precision": arg_type_class_precisions[class_type],
                "recall": arg_type_class_recalls[class_type]
            }
            for class_type in config['role_names']
        }
    }

    with open(out_file, 'w') as f:
        f.write(json.dumps(info_dic))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    """
    config has format:
    {
        "event_type_names": [
            "attack",
            "bombing",
            ...
        ],
        "role_names": [
            "PerpInd",
            "PerpOrg",
            ...
        ]
    }
    """
    parser.add_argument('--relax_match', action='store_true') # set if relaxed matching of strings
    parser.add_argument('--in_file', type=str, required=True)
    """
    Expects --in_file to be JSON of form
    [
        {
            "pred_templates": [
                list of GTT style templates
            ],
            "gold_templates": [
                list of GTT style templates
            ]
        }
    ]
    """
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--filter_lst', type=str, required=False)
    args = parser.parse_args()

    main(args.config, args.in_file, args.relax_match, args.out_file, args.filter_lst)