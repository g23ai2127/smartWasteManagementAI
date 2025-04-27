import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import cv2
import tempfile
from io import BytesIO

# Load your trained model
model = tf.keras.models.load_model('garbage_classification_model_inception.h5')

# Define disposal info and dustbin colors
disposal_info = {
    'cardboard': "Place in the blue recycling bin.",
    'glass': "Place in the green recycling bin.",
    'metal': "Place in the yellow recycling bin.",
    'paper': "Place in the blue recycling bin.",
    'plastic': "Place in the red recycling bin.",
    'trash': "Place in the black general waste bin."
}

dustbin_colors = {
    'cardboard': 'blue',
    'glass': 'green',
    'metal': 'yellow',
    'paper': 'blue',
    'plastic': 'red',
    'trash': 'black'
}

# Helper function: Get dustbin image path
def get_dustbin_image_path(color):
    return f'dustbins/{color}.png'

# Prediction function for image
def predict_waste_category_from_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = image.resize((256, 256))  # Corrected resize
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        with st.spinner('🔍 Analyzing image... Please wait!'):
            prediction = model.predict(img_array)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Prediction function for video
def predict_waste_category_from_video(video_bytes):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_bytes)
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []
    success, frame = cap.read()

    while success:
        if frame_count % 30 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            prediction = predict_waste_category_from_image(frame)
            if prediction is not None:
                predicted_class_idx = np.argmax(prediction)
                predicted_class = list(disposal_info.keys())[predicted_class_idx]
                predictions.append(predicted_class)
        success, frame = cap.read()
        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        raise RuntimeError("No frames were captured from the video!")

    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction, frame

# Convert DataFrame to CSV
def convert_df(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data

# Main App
def main():
    st.title("♻️ Waste Management AI")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About This Project")
        st.markdown("""
        This Smart Waste Management AI system classifies waste into categories like **Plastic**, **Metal**, **Paper**, etc.

        - Built using **TensorFlow**, **Keras**, and **Streamlit**
        - Accepts **Image** and **Video** inputs
        - Shows **Waste Category**, **Disposal Info**, and **Recommended Dustbin**

        ♻️ Contributing towards a greener planet!
        """)

        st.header("👨‍💻 Developer Info")
        st.markdown("""
        **Developed by:** Sachin Patil  
        ✉️ Email: : m24de3067@iitj.ac.in  
        🌐 GitHub: https://github.com/g23ai2127/smartWasteManagementAI.git
        """)

    st.info("🌗 Tip: Switch your Streamlit Theme (Dark/Light) from the Settings ⚙️ Menu (Top-Right corner) for a better experience!")

    st.markdown("Upload an image or a video and let the AI classify the waste category.")

    uploaded_file = st.file_uploader("📂 Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

            prediction = predict_waste_category_from_image(image)
            if prediction is not None:
                predicted_class_idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                predicted_category = list(disposal_info.keys())[predicted_class_idx]

                st.balloons()

                st.success(f"🏷️ **Waste Category Predicted: {predicted_category}**")
                st.subheader("💚 Disposal Information:")
                disposal_text = disposal_info.get(predicted_category, "Disposal information not available.")
                st.info(disposal_text)

                dustbin_color = dustbin_colors.get(predicted_category, 'blue')
                dustbin_image_path = get_dustbin_image_path(dustbin_color)
                if dustbin_image_path and os.path.exists(dustbin_image_path):
                    dustbin_image = Image.open(dustbin_image_path)
                    st.image(dustbin_image, caption=f"💚 Dustbin for {predicted_category}", width=200)

                st.subheader("📄 Download Prediction Report")
                result_df = pd.DataFrame({
                    'Waste Category': [predicted_category],
                    'Confidence (%)': [f"{confidence:.2f}"],
                    'Disposal Info': [disposal_text]
                })
                csv = convert_df(result_df)
                st.download_button(
                    label="📥 Download Report as CSV",
                    data=csv,
                    file_name='prediction_report.csv',
                    mime='text/csv',
                )

        elif uploaded_file.type.startswith('video'):
            try:
                predicted_category, frame = predict_waste_category_from_video(uploaded_file.read())
                st.video(uploaded_file, start_time=0)
                st.balloons()

                st.success(f"🏷️ **Waste Category Predicted: {predicted_category}**")
                st.subheader("💚 Disposal Information:")
                disposal_text = disposal_info.get(predicted_category, "Disposal information not available.")
                st.info(disposal_text)

                dustbin_color = dustbin_colors.get(predicted_category, 'blue')
                dustbin_image_path = get_dustbin_image_path(dustbin_color)
                if dustbin_image_path and os.path.exists(dustbin_image_path):
                    dustbin_image = Image.open(dustbin_image_path)
                    st.image(dustbin_image, caption=f"💚 Dustbin for {predicted_category}", width=200)

            except RuntimeError as e:
                st.error(str(e))
        else:
            st.error("❌ Unsupported file format. Please upload an image (jpg, jpeg, png) or a video (mp4).")

if __name__ == "__main__":
    main()
