Each dataset has its own folder and all but the `run` folder has split directories for each trigger source

## Overview of folders
`error_analysis`: scorer outputs for each experiment (for WikiEvents, this is split into `strict` and `non_strict`, where the former requires the event type predicted with an argument annotation to correspond to an existing argument role, event type pair and the latter does not require this)

`g2_environments`: directories where all experiments are run on G2

`g2_logs`: `.err` and `.out` files for experiment runs

`run`: files customized for each variant (default, no trigger, single pass) of TANL

`slurm`: slurm files to run experiments

## More specific details

`error_analysis`: there is a directory for each experiment with a bunch of folders `output_<n>` which correspond to the scorer results for each epoch

`g2_environments`: each experiment has 2 directories of text files (for dev and test) which are the raw model predictions at each epoch

## Running experiments

1. The slurm files assume a Conda environment is created with the name "TANL". Make this environment using `conda create --name TANL python=3.10`.

2. Navigate to the corresponding directory containing the slurm file, then `sbatch <slurm_file_name>`