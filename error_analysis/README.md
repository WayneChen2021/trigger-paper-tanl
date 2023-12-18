Get scorer results from theh experiments using `base_scorer.py`

## Usage
The script takes in a config file that has the format:
```
[
    {
        "experiment_dir": <experiment dir>, # should just contain dev_predictions, test_predictions folders
        "output_dir": <dir to output results>,
        "hungarian_config": <one of configs from scorer_configs, depending on dataset>,
        "types_mapping": <one of mappings from types_mapping, depending on dtaset>,
        "gtt_ref_dev": <dev dataset file formatted for scorer (in gtt folder in data folder)>,
        "tanl_ref_dev": <corresponding dev_event.json file in dataset folder in data folder>,
        "gtt_ref_test": <test dataset file formatted for scorer (in gtt folder in data folder)>,
        "tanl_ref_test": <corresponding test_event.json file in dataset folder in data folder>,
        "relax_match": <whether or not to compute relaxed matchings>,
        "dataset_name": <one of "muc" or "wikievents">
    }
]
```

See [base_score_config.json](base_score_config.json) for an example