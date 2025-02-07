# 🌍 City-scale Geolocation from Street-Level Images using Attention Nets, Object Localization, Colour-Space Embeddings, and Text Extraction 🚀

Welcome to the **Capstone MAI Project** for the **University of Canterbury (2024-2025)**! 🎓 This repo is all about building a *GeoGuessr Bot*... but a supercharged, AI-powered version. More on that in the business section. 📍


#### ⚡️ This README will be updated as the model is finalized. ⚡️

<p align="center"> 
  <img src="./docs/gitembeds/header1.png" width="200" height="200" /> 
  <img src="./docs/gitembeds/thinking.gif" width="200" height="200" /> 
  <img src="./docs/gitembeds/springfield_w_point.jpg" width="200" height="200" /> 
</p> 
<p align="center"><em>Image Features -> NN Model -> Output Location</em></p>

## 📚 Table of Contents
- [Background](#background)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Justifications and Deep Learning](#justifications-and-deep-learning)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [Contact Information](#contact-information)

## 🧭 Ever Played GeoGuessr? 🧭

GeoGuessr is an online geography game where you guess locations based on Google Street View images. 🌍 You analyze visual clues like road signs, landmarks, and vegetation to guess where in the world the photo was taken. Players earn points based on how close they are to the actual location. 🏆

This project takes that concept and turns it into a **GeoGuessr Bot**! 🚀 It uses **deep learning** to make geolocation predictions from street-level images. It's like turning **Google Street View** into a giant treasure map, and this bot is your trusty guide! 🗺







## 🌍 Hello, This is the Repo readme🚀  


So, what's inside? 

---

### 🏗️ 1. Procurement – Getting the Data  
We need street-level images from **Mapillary** and **Google Street View** to train our AI.  

🛠 Tools provided:  
- **StreetView**  
  - Generate points 📍  
  - Snap points to a shapefile 🗺️  
  - Download a shapefile 📂  
  - Generate a CSV of points along vectors in a shapefile  

- **Mapillary**  
  - `/mapillary/download.py` – Call it to download an image (just supply your API key 🔑)  
  - Other scripts include the **download client** and the **database client**  

👉 **Setup:** Drop your API keys into `KEYS.json` at the root, and you’re ready to roll.  

---

### 🎯 2. Ground Truth – Organizing the Data  
We're working with `.pkl` files here. You need to generate one from Mapillary's scripts to start processing.  

🔥 Tools available:  
- Check `chicago.pkl` for reference  
- Use `is_ingested` and `is_downloaded` to track progress  
- Scan directories for mismatches using scripts in `/scripts/tools`  

---

### 🤖 3. Machine Learning – The AI Brain 🧠  
This is where **the magic happens**!  

📂 **Core ML Modules**:  
- **OCR** (`/OCR`) – Wrappers for **EasyOCR** & **SentenceTransformer** models 📝  
- **Object Detection** (`/OBJ`) – Wrappers for **YOLO**, plus tools to train a custom model 🚗🔍  
- **Color Analysis** (`/Colorgram`) – Uses **HSV hist embeddings** 🎨  
- **Neural Network Training** (`/train`) –  
  - `OVERALL_train.py` trains the main network 🌍  
  - `SEGS_train.py` trains the signs network 🚏  
  - Both have utility scripts for support  

📌 **Clustering** (`/clustering`) – Where we analyze patterns in predictions and refine results.  

---

### 📊 4. Reports – Turning Numbers into Insights  
All report-generation tools are stored in `/reports`. This includes:  
- Scripts to analyze and visualize model results 📉  
- Four compressed files containing **all results** 📦  

📌 **Note:** The **151,510 image dataset & embeddings** are **not included** in the GitHub repo. Need access? **Drop me an email at** 📩 `twoofeverybug@gmail.com`, and I’ll hook you up.  

---

### 🎉 That’s a Wrap (For Now)  
There are **plenty** of other tools in this repo—most are well-named and easy to use. The README will keep evolving as the project progresses! 🚀  

> **#TODO:** Finish documenting all the tools and add more details where needed.  

In the meantime... **cheerio, happy punting! 🏉**  



(I used a generative AI to give readme.md flair....)