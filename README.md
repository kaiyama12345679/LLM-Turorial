## LLM Train Tutorial for me.
```
singularity build --fakeroot llm-train.sif llm-train.def
qsub -g <your_ABCI_GROUP> train.sh
```

### You need to set environment variables in `.venv` file.
- `WANDB_API_KEY`: wandb_api_token
- `DISCORD_WEBHOOK_URL`: url of discord webhook