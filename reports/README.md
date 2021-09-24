# reports directory

Contents of this directory:

- `figures/`: these are figure components that I use when putting the figures
  together (e.g., a cartoon of preferred spatial frequency, legends that use
  images of the stimuli). These are cached in the github repo and are not
  updated or created by any analysis script.
  
- `paper_figures/`: these are the actual figures used in the paper, as created
  by the `snakemake` file. There are none in the github repo, see main repo
  README for details on how to create them.
  
- `figure_rules.txt`: this is a list of snakemake rules that create figures
  (rather than analyze the data). It can be used to limit snakemake's search of
  possible analysis paths. See main github README for more details.
