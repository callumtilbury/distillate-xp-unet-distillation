# unet-distillation

An autonomous ML experiment powered by [Distillate](https://github.com/rlacombe/distillate).

## What is Distillate?

Distillate is an open-source tool that helps scientists design, launch, and track autonomous ML experiments — with a paper library built in. Nicolas, the research alchemist, orchestrates Claude Code agents that iteratively improve your models.

## Reproducing this experiment

```bash
# Install Distillate
pip install distillate

# Clone and run
git clone https://github.com/$(gh api user -q .login)/distillate-xp-unet-distillation.git
cd distillate-xp-unet-distillation
distillate launch  # Resume the experiment
```

## Results

See `.distillate/runs.jsonl` for the full experiment history.
