Contains folders for each dataset

## Overview of folders
`gtt`: dev and test splits formatted like GTT for scorer use

All other folders correspond to different trigger sources, each folder (except `no_trig`) contains the dataset splits formatted for TANL for trigger extraction, argument extraction, or full event extraction. The trigger extraction and argument extraction ones are used for training regular pipelined TANL. The full event extraction files are used for evaluation of all variants of TANL and for training single pass TANL.