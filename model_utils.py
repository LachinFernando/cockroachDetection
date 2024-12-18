from ultralytics import YOLO
import streamlit as st

MODEL_NAME = "cockroach_detection.pt"

@st.cache_resource
def load_yolo_model(image_path = MODEL_NAME):

    return YOLO(image_path)