"""
Real-Time Moving Object Detection — Streamlit app (single-repo version).

Detection logic isi repo ki detection.py file me hai (root me), isliye
seedha "from detection import ObjectDetector" use hota hai.

Run:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from detection import ObjectDetector

st.set_page_config(page_title="Real-Time Moving Object Detection", layout="wide")

st.title("🎯 Real-Time Moving Object Detection")
st.caption("YOLOv8 + Centroid Tracking — batata hai object Moving hai ya Static")


@st.cache_resource
def get_detector():
    return ObjectDetector()


detector = get_detector()

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Live Stats")
    total_placeholder = st.empty()
    moving_placeholder = st.empty()
    static_placeholder = st.empty()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    annotated_img, stats = detector.process_frame(img)

    st.session_state.total_objects = stats["total_objects"]
    st.session_state.moving = stats["moving"]
    st.session_state.static = stats["static"]

    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")


with col1:
    webrtc_streamer(
        key="motion-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    total_placeholder.metric("Total Objects", st.session_state.get("total_objects", 0))
    moving_placeholder.metric("Moving", st.session_state.get("moving", 0))
    static_placeholder.metric("Static", st.session_state.get("static", 0))
    st.info("🔴 Red box = Moving\n\n🟢 Green box = Static\n\n🟡 Yellow box = Detecting...")
