# yolo-frontend

Streamlit-based frontend for real-time moving object detection.
Detection logic ye repo me nahi hai — `motion_detector` package
(`yolo-model` repo) se aata hai, jo `requirements.txt` me GitHub URL
ke through install hota hai.

## Setup

1. Pehle `yolo-model` repo ko GitHub par upload karo (README wahan hai).
2. Is repo ke `requirements.txt` me `<your-username>` ko apne GitHub
   username se replace karo.
3. Install karo:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run streamlit_app.py
```

Browser khud khulega — usually **http://localhost:8501**. Camera
permission allow karna.

## GitHub par upload

```bash
cd yolo-frontend
git init
git add .
git commit -m "Streamlit frontend for real-time motion detection"
git branch -M main
git remote add origin https://github.com/<your-username>/yolo-frontend.git
git push -u origin main
```

## Streamlit Community Cloud par deploy (optional)

1. GitHub par push karne ke baad [share.streamlit.io](https://share.streamlit.io) pe jao
2. "New app" → is repo select karo → main file: `streamlit_app.py`
3. Deploy karo

Note: Cloud par deploy hone par browser hi khud user ka local webcam
use karta hai (webrtc ke through), server ka apna camera nahi chahiye
hota — isliye ye deployment ke liye bhi kaam karega.
