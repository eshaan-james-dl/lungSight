# LungSight Streamlit Frontend

Run locally:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Defaults to `http://localhost:8000`.

To point at EC2:

```powershell
$env:LUNGSIGHT_API_URL="http://15.252.8.232:8000"
streamlit run app.py
```

Expected API:
- GET `/health`
- POST `/predict`
- POST `/gradcam?disease=<label>&explain=true|false`
