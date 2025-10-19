<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <img src="assets/alex.png" alt="ALEX Icon" width="30"/>
  <h1 style="margin: 0;">ALEX 1.0: General Purpose Minecraft RL and LLM AI agent</h1>
</div>
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
