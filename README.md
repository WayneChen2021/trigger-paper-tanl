# Code for TANL experiments

## Folders
`original_scripts`: Contains the file structure for running TANL

## Scripts
`conversion.py`: Converts data into the TANL format

**--in_file**: standardized MUC JSON file

**--out_train_trig**: output JSON containing training examples for trigger extraction

**--out_train_arg**: output JSON containing training examples for argument extraction

**--out_train_event**: output JSON containing training examples for event extraction (for evaluation)

**--out_test_trig**: output JSON containing testing examples for trigger extraction

**--out_test_arg**: output JSON containing testing examples for argument extraction

**--out_test_event**: output JSON containing testing examples for event extraction (for evaluation)

**--num_trigs**: number of triggers per template (defaults to 1)

**--span_selection**: how to pick the mention for each entity ("earliest" or "longest")

**--trigger_selection**: what to order the triggers by ("position" or "popularity")