# Overview of splits

`bucketed` places the examples into 5 buckets with each bucket covering an equal range (so each bucket will likely not have the same number of examples); `raw` does not have bucketing

## Splits
1. `argument_spread`: How far all coreferrent mentions in a template are spread out from their mean location in the text
2. `doc_len`: Document length
3. `num_entities`: Number of entities present in each template
4. `num_templates`: Number of templates in document
5. `template_ordering`: The order of the template in the document (determined by average distance of the template arguments excluding the trigger); templates with no arguments are ordered last 