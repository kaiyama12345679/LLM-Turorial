## LLM Train Tutorial for me.

### You need to set environment variables in `.venv` file.
- `WANDB_API_KEY`: wandb_api_token
- `DISCORD_WEBHOOK_URL`: url of discord webhook

### First, You have to build singularity image
```
module load singularitypro
singularity build --fakeroot llm-train.sif llm-train.def
```

### You can try single-node training or multi-node training.
1. Single-Node
```
qsub -g <your_ABCI_group> train.sh
```
2. Multi-Node
```
qsub -g <your_ABCI_group> multinode_train.sh
```