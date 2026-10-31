import os
import json
from typing import Dict, Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="BrainScan V2", page_icon="🧠", layout="wide")

st.title("🧠 BrainScan V2")
st.caption("MRI tumor classification and detection dashboard")

with st.sidebar:
    st.header("Model controls")
    st.info("API endpoint: %s" % API_BASE_URL)

uploaded_file = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded image")
        st.image(uploaded_file, caption="Original MRI", use_container_width=True)

    with col2:
        st.subheader("Classification")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/jpeg")}
        try:
            response = requests.post(f"{API_BASE_URL}/predict/classify", files=files, timeout=120)
            response.raise_for_status()
            result = response.json()
            st.json(result)
        except Exception as exc:
            st.error(f"Classification request failed: {exc}")

    st.subheader("Tumor detection")
    try:
        response = requests.post(f"{API_BASE_URL}/predict/detect", files=files, timeout=120)
        response.raise_for_status()
        result = response.json()
        if result.get("annotated_image"):
            import base64
            from io import BytesIO
            from PIL import Image

            decoded = base64.b64decode(result["annotated_image"])
            image = Image.open(BytesIO(decoded))
            st.image(image, caption="Detected MRI", use_container_width=True)
        st.json(result)
    except Exception as exc:
        st.error(f"Detection request failed: {exc}")
