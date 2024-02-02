# Overview of Datasets

For each dataset, for each method of acquiring triggers, the datasets are split into train, dev, test following the format for GTT.

## Other notes
- Triggers are recorded just as another role filler type in the GTT format (it is formatted that same as any other role filler type).
- Human triggers for MUC is still incomplete. The templates without triggers have an empty list for their triggers.
- Keyword triggers are located in the `keywords.json` files. The keywords are stems (so a stem of "kill" will can match with an actual trigger span of "killed" or "killing")
- For WikiEvent, [all_roles_in_template](WikiEvent/all_roles_in_template/) has templates that contain all roles/entity types across the entire corpus; [event_roles_in_template](WikiEvent/event_roles_in_template/) contains only the roles/entity types of 1 event type in each teamplate [code4struct](WikiEvent/code4struct/) contains ACE style entity annotations and is meant for running Code4Struct
- For RAMs, [3_level_events](RAMs/3_levels_events/) consider 3 levels of classification for event types; [2_level_events](RAMs/2_levels_events/) contains only 2
- for [splits](splits), check README in directory