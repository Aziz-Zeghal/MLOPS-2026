# MLOPS-2026 Mini project 1
First train the model:
```bash
python train_model.py
```

## 2. Using a ML model in streamlit
Run the streamlit with this command:
```bash
streamlit run model_app.py
```

## 3. A machine learning model in a web service mini project

### Level 0 (on your local machine) 
Use this command:
```bash
fastapi run model_fastapi.py --port 9000
```

### Level 0.5 (on a remote machine)
Use the same command after ssh on a machine.

### Level 1 (on your local machine)

```bash
docker build --tag my_app .
docker run -p 9000:9000 my_app
```

### Level 2 - Deploy the container in a cloud virtual machine
Use the same commands after ssh on a cloud VM.
