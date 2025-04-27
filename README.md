# ♻️ Smart Waste Management AI

A smart AI-based waste classification system built using **TensorFlow**, **Keras**, and **Streamlit**.
This application predicts the type of waste (Plastic, Metal, Glass, Paper, etc.) from uploaded images or videos, and recommends the correct dustbin for disposal.

---

## 🛠️ Features

- 📸 **Upload an Image** (jpg, jpeg, png)
- 🎥 **Upload a Video** (mp4)
- ⚡ **Real-Time Waste Category Prediction**
- 📂 **Download Prediction Report** (as CSV)
- 🖼️ **Shows Recommended Dustbin Image**
- 🎉 **Balloons Success Animation**
- 🌗 **Light/Dark Theme Switching (Streamlit settings)**

---

## 🚀 How to Run Locally

1. **Clone the Repository**

```bash
git clone https://github.com/yourgithub/garbage-classification-app.git
cd garbage-classification-app
```

2. **Install Required Libraries**

```bash
pip install -r requirements.txt
```

3. **Ensure these files are present:**
   - `app.py`
   - `garbage_classification_model_inception.h5` (your trained model)
   - `dustbins/` folder (with images like `blue.png`, `green.png`, etc.)

4. **Run the App**

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
garbage-classification-app/
│
├── app.py
├── garbage_classification_model_inception.h5
├── dustbins/
│   ├── blue.png
│   ├── green.png
│   ├── yellow.png
│   ├── red.png
│   └── black.png
├── requirements.txt
└── README.md
```

---

## 🛆 Deployment on Streamlit Cloud

- Push your project to a **public GitHub repository**.
- Go to [Streamlit Cloud](https://streamlit.io/cloud) → Deploy new app → Select your GitHub repo.
- Set `app.py` as the main file.
- 🌟 Done!

---

## 🧐 Model Information

- **Architecture:** Custom CNN Model
- **Input Size:** (224, 224, 3)
- **Trained On:** Garbage classification dataset
- **Classes:** Cardboard, Glass, Metal, Paper, Plastic, Trash

---

## 👨‍💻 Developer

- **Name:** Sachin Patil
- **Email:** [your.email@example.com](mailto:your.email@example.com)
- **GitHub:** [github.com/yourgithub](https://github.com/yourgithub)

---

# 🚀 Let's make waste management smarter together! ♻️
