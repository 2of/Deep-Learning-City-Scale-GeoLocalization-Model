# 🌍 City-Scale Geolocation from Street-Level Images

### Using Attention Nets, Object Localization, Colour-Space Embeddings & Text Extraction 🚀

Welcome to the repository for my **Capstone Project** as part of the **Master of Artificial Intelligence** at the [University of Canterbury](https://www.canterbury.ac.nz/study/academic-study/qualifications/master-of-artificial-intelligence#accordion-3422d1b02a-item-bb8aa0463d-button).

## 📌 Project Overview

This fully self-directed project focuses on building a **multi-neighbourhood-scale GeoLocalization model**, capable of pinpointing locations from street-level imagery. To achieve this, I implemented a **pseudo-ensemble of machine learning models** trained on a **150,000-image dataset**, covering a **20 km²** urban area.
<p align="center"> 
  <img src="./docs/gitembeds/header1.png" width="200" height="200" /> 
  <img src="./docs/gitembeds/thinking.gif" width="200" height="200" /> 
  <img src="./docs/gitembeds/springfield_w_point.jpg" width="200" height="200" /> 
</p> 



## 🎓 The PDF's

<p align="center">
  <a href="https://github.com/2of/Deep-Learning-City-Scale-GeoLocalization-Model/blob/main/proposal.pdf" style="text-decoration: none; font-size: 2rem;">
    🚀 <strong>Proposal.pdf</strong>
  </a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/2of/Deep-Learning-City-Scale-GeoLocalization-Model/blob/main/THESIS_mini.pdf" style="text-decoration: none; font-size: 2rem;">
    🔥 <strong>Thesis.pdf</strong>
  </a>
</p>


## 🚀 Overview

- **Average Localization Error:** ~1.3 km
- **Candidate Area:** ~20 km^2
- **Dataset Size:** 150k+ images, 383k+ individual sign detections
- **Core Techniques Used:**
  - 🧠 **Attention-based Neural Networks** for feature learning
  - 🎯 **Object Localization / Detection** Fine tuned existing object detection models for detecting *GeoInformers*  
  - 🎨 **Colour-Space Embeddings** for scene analysis and getting a grip on the colourspace of signs and overall scenes
  - 🔍 **Text Extraction** from street signs and overall images
  - **Clustering Analysis** for validating, helping our ensemble models... 

## 🏆 Why This Matters

City-scale geolocation has diverse applications, including:

- **Augmenting navigation systems** with better landmark recognition
- **Enhancing autonomous vehicle localization** in urban environments
- **Improving local search accuracy** for mapping services


## 📓What does Google Notebook LM say about our paper:
*This research explores a novel method for geolocating street-level images using machine learning. The core concept involves extracting key features from images – text, objects and colour information – and using these to train neural networks. Different machine learning models were employed, including those incorporating attention mechanisms, to predict the location of an image within a limited geographical area. The effectiveness of these models was assessed, considering how factors such as data density and feature selection impact location accuracy. The outcomes reveal the potential and challenges of using image-based features for precise geolocation. The project successfully geolocates signs using sign-based features while still acknowledging the work required to generalise the outcomes.*



### Graded: A



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