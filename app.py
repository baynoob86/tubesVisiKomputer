import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Deteksi Motor", page_icon="🛵")

st.sidebar.title("⚙️ Panel Kontrol")

# Pilihan Model
st.sidebar.subheader("1. Pilih Model Deteksi")
model_type = st.sidebar.selectbox(
    "Gunakan Model:",
    ("YOLOv8 Nano", "YOLOv11 Nano")
)

if model_type == "YOLOv11 Nano":
    model_path = 'yolo11n.pt' 
else:
    model_path = 'yolov8n.pt'

# Sensitivitas
st.sidebar.subheader("2. Sensitivitas")
conf_level = st.sidebar.slider("Confidence ", 0.0, 1.0, 0.25) 
iou_level = st.sidebar.slider("Overlap IoU ", 0.0, 1.0, 0.45) 

# 3. Pengaturan Gambar 
st.sidebar.subheader("3. Ukuran Gambar")
img_size = st.sidebar.number_input(
    "Masukkan ukuran input (piksel):", 
    min_value=320, 
    max_value=1280, 
    value=640, 
    step=32,
    help="Standar YOLO adalah 640. Gunakan kelipatan 32."
)


# MAIN PAGE
st.title("🛵 Sistem Deteksi Okupansi Sepeda Motor")
st.write("Silakan pilih model di sidebar kiri untuk membandingkan hasil deteksi.")

# Load Model Berdasarkan Pilihan
try:
    model = YOLO(model_path)
except Exception as e:
    if model_path == 'best.pt':
        st.error("⚠️ File 'best.pt' tidak ditemukan di folder ini! Pastikan kamu sudah menaruh file hasil trainingmu di sini.")
        st.stop()
    else:
        st.warning(f"Sedang mendownload {model_path} otomatis... (Pastikan ada internet)")
        model = YOLO(model_path)

# Fungsi Upload 
uploaded_file = st.file_uploader("Upload Gambar atau Video...", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Gambar
    if file_extension in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Gambar Asli', width=700)
        
        if st.button('Mulai Deteksi Gambar'):
            with st.spinner(f'Memproses menggunakan {model_path}...'):
                img_array = np.array(image)

                results = model.predict(img_array, conf=conf_level, iou=iou_level, imgsz=img_size)

                res_plotted = results[0].plot()
                jumlah_objek = len(results[0].boxes)
                
                if jumlah_objek > 0:
                    st.success(f"Selesai! Terdeteksi **{jumlah_objek}** objek.")
                else:
                    st.warning("Tidak ada objek yang terdeteksi. Coba turunkan 'Confidence'.")
                
                st.image(res_plotted, caption=f'Hasil Deteksi ({model_path})', width=700)
                
                res_image = Image.fromarray(res_plotted)
                import io
                buf = io.BytesIO()
                res_image.save(buf, format="JPEG")
                st.download_button(
                    label="Download Gambar Hasil",
                    data=buf.getvalue(),
                    file_name=f"hasil_{model_path}.jpg",
                    mime="image/jpeg"
                )

    # Video
    elif file_extension == 'mp4':
        st.video(uploaded_file)
        st.write("Video berhasil diupload.")
        
        if st.button('Mulai Deteksi Video'):
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break 

                results = model.predict(frame, conf=conf_level, iou=iou_level, imgsz=img_size, stream=True) 

                
                for result in results:
                    res_plotted = result.plot()
                    out.write(res_plotted)
                
                frame_count += 1
                if total_frames > 0:
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Memproses frame {frame_count}/{total_frames} dengan {model_path}...")

            cap.release()
            out.release()
            
            status_text.text("Pemrosesan Selesai!")
            progress_bar.progress(1.0)
            
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
                st.download_button(
                    label="Download Video Hasil",
                    data=video_bytes,
                    file_name=f"hasil_video_{model_path}.mp4",
                    mime="video/mp4"
                )
