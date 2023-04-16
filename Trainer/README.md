# Trainer

This directory contains all the training code to reproduce the experiments in the paper.

The trainer assumes that the files have been already converted to the WebDataset format.
Please see `../Unity3D/` for details regarding how to generate the dataset as well as convert it into the WebDataset format.

## Execution

Please check `trainer_slurm_v6.sh` to see the execution scripts used to produce our results.

## Plots

We provide plotting utilities in `log_parsers/` directory.
The log parsers parses the final output log in order to plot model performance as included in the paper.

Similarly, `viz_utils/` provides utilities to plot examples from the generated datasets.

Please check `gen_plots.sh` to see how to generate the final plots after training.

## License

MIT
