# Fine-tuning pretrained BERT model for Slovene classification tasks

This repository contains scripts and data for tuning multilingual models for solving Slovene classification tasks.
It contains adapters trained on 9 Slovene classification tasks.
Training data for the tasks was extracted from the ssj500k corpus. All the adapters were trained on this data.

Four different methods of tuning are available:
* Full model fine-tuning
* Last layer tuning
* Adapter tuning
* AdapterFusion tuning

## Instructions

Run model training on a classification task by running:

```bash
python3 run_classification.py training_config.json <tuning_mode>
```

Argument `<tuning_mode>` has 4 possible values:
* full-model
* last-layer
* adapter
* adapterfusion

We determine which classification task to train for, by setting the `"data_dir"` argument in `training_config.json` to the folder containing training data for the desired task.
We also set the `"labels"` argument to the .txt file containing the labels.
