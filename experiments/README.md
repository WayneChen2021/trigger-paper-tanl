# Experiment outputs and error analysis results

Each folder corresponds to a dataset. Each experiment has corresponding entries
in the `error_analysis`, `g2_environments`, `g2_logs`, and `slurm` folders.

## Overview of folders

`error_analysis`: contains error analysis outputs and training plots

`g2_environments`: contains folder in which G2 executes training/inference; most
files are not committed since files from [original_scripts](../original_scripts/)
are copied there

`g2_logs`: contains logs outputted by G2

`slurm`: contains slurm commands to execute each experiment on G2

There might be extra folders (for example [run](MUC/run)) that help streamline
setting up experiments

## Running experiments

1. If the experiment has already been run delete the folder containing the old
results or rename the folder

2. Call `sbatch <slurm_file_nam>` within the corresponding `slurm` folder; need
to do this via G2

3. Add entries in the [.gitignore](../.gitignore) (if they're not there already)
to only commit results necessary for error analysis (from G2)