# Code for TANL experiments

## Folders
`original_scripts`: Contains the file structure for running TANL; these are copied
everytime before running experiments

`experiments`: Contains all results for all experiments

`error_analysis`: Files for the scorer

`data`: Datasets formatted for TANL and for the scorer

## Conversion Script
`conversion.py` converts the processed data into the format required by TANL

`in_file`: JSON file in format:
```
{
    "<docid>": {
        "source": "<trigger source (human, keyword, etc)>",
        "templates": [
            {
                "incident_type": "<incident type 1>",
                "Role 1": [
                    [
                        [
                            "<1st event entity 1 mention 1 (as a string)>",
                            <character index of mention (as an int)>
                        ],
                        [
                            "<1st event entity 1 mention 2 (as a string)>",
                            <character index of mention (as an int)>
                        ]
                    ],
                    [
                        [
                            "<1st event entity 2 mention 1 (as a string)>",
                            <character index of mention (as an int)>
                        ],
                        [
                            "<1st event entity 2 mention 2 (as a string)>",
                            <character index of mention (as an int)>
                        ]
                    ]
                ],
                "Role 2": [
                    [
                        [
                            "<1st event entity 3 mention 1 (as a string)>",
                            <character index of mention (as an int)>
                        ],
                        [
                            "<1st event entity 3 mention 2 (as a string)>",
                            <character index of mention (as an int)>
                        ]
                    ]
                ],
                "Role 3": [], # no entities for this role
                ...
                "Triggers": [
                    [
                        [
                            "<trigger span for 1st event>",
                            <character index of span (as an int)>
                        ]
                    ]
                ]
            },
            {
                "incident_type": "<incident type 2>",
                ...,
            }
        ],
        "docid": "<docid>",
        "doctext": "abcdefg..."
    }
}
```

`out_train_trig`: (optional) output JSON containing training examples for trigger extraction

`out_train_arg`: (optional) output JSON containing training examples for argument extraction

`out_train_event`: (optional) output JSON containing training examples for event extraction (for evaluation or for single pass variants)

`out_dev_trig`: (optional) output JSON containing development examples for trigger extraction

`out_dev_arg`: (optional) output JSON containing development examples for argument extraction

`out_dev_event`: (optional) output JSON containing development examples for event extraction (for evaluation or for single pass variants)

`out_test_trig`: (optional) output JSON containing testing examples for trigger extraction

`out_test_arg`: (optional) output JSON containing testing examples for argument extraction

`out_test_event`: (optional) output JSON containing testing examples for event extraction (for evaluation or for single pass variants)

`num_trigs`: (optional) number of triggers per template (defaults to 1)

`out_train_gtt`: (optional) output JSON containing training examples in GTT format (for scorer use)

`out_dev_gtt`: (optional) output JSON containing development examples in GTT format (for scorer use)

`out_test_gtt`: (optional) output JSON containing testing examples in GTT format (for scorer use)

`span_selection`: (optional) how to pick the mention for each entity ("earliest" or "longest"); default is "earliest"

`trigger_selection`: (optional) what to order the triggers by ("position" or "popularity", where "popularity" is given by annotations for manual triggers and rating for GPT triggers); default is "position"

`dummy_trigs`: (flag) set if to use dummy triggers (for single pass with no trigger experiments)

`num_dummy_events`: number of dummy events to use (there must be at least as many dummy events as the max number of events in any document); default is 100

`dataset`: name of dataset ("MUC" or "WikiEvents")