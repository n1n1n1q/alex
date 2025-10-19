<img src="assets/alex.png" alt="ALEX Icon" width="40" align="left"/> 

# ALEX 1.0: Minecraft RL-micro and LLM-macro management AI agent
## Environment setup
1. Build docker with   
```bash
chmod +x docker/build.sh
./docker/build.sh
```

2. Put WANDB_API_KEY into `.env` file

3. Run the environment with
  
```bash
chmod +x docker/run.sh
./docker/run.sh {--reset/-r to reset the environment}
```
