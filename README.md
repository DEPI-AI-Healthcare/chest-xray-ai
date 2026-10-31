# 🩺 InnoTech AI Healthcare System

An advanced end-to-end AI-powered healthcare platform designed for automated Chest X-ray analysis, diagnostics, and clinical assistance, developed as part of the Digital Egypt Pioneers Initiative (DEPI) graduation project.

---

## 🚀 Project Overview & Architecture

InnoTech integrates computer vision, deep learning, and natural language processing to assist medical professionals in accurate diagnosis and report analysis:

1. **Phase 1: Preprocessing & Data Pipeline**
   - Merging projection and report datasets (Indiana University Chest X-ray Collection).
   - Image resizing, normalization, and generating clean structured splits (`cnn_data.csv` & `rag_data.csv`).
   - Dynamic environment path configuration (`config.py`) to seamlessly bridge local VS Code and Google Colab environments.

2. **Phase 2A: CNN Classification & Grad-CAM Interpretability**
   - Multi-model benchmarking (ResNet50, EfficientNet, DenseNet121) for accurate binary classification (Normal vs. Abnormal).
   - Grad-CAM heatmap visualization to highlight regions of interest for physicians.

3. **Phase 2B: U-Net Segmentation & RAG Chatbot**
   - U-Net lung mask extraction and segmentation.
   - Clinical report text cleaning and vectorization using LangChain and FAISS Vector Database for smart medical Q&A.

4. **Phase 3 & 4: Backend & Mobile App Deployment**
   - **FastAPI Backend:** Exposing high-performance REST APIs (`/predict` and `/chat`).
   - **Android Mobile App:** Developed in Kotlin with Retrofit integration for seamless medical staff interaction.
   - **Containerization:** Fully Dockerized architecture using Docker Compose.

---

## 📂 Repository Structure

```text
chest-xray-ai/
│
├── backend/          # FastAPI backend services & endpoints
├── cnn/              # CNN classification models & training scripts
├── data/             # Processed datasets (cnn_data.csv, rag_data.csv)
├── gradcam/          # Grad-CAM heatmap generation scripts
├── mobile/           # Android Kotlin mobile application
├── models/           # Saved model weights & checkpoints
├── preprocessing/    # Data cleaning, text & image preprocessing
├── rag/              # RAG pipeline, LangChain & FAISS implementation
├── unet/             # U-Net segmentation models
├── src/              # Core configuration files (config.py)
├── .gitignore        # Git ignore configurations
└── README.md         # Project documentation

👥 Team & Contributors
Tech Lead: Ereny Habib

AI & Deep Learning Engineers

NLP & RAG Engineers

Backend & Mobile Developers

---
## 📦 Full Dataset Access
You can access and download the complete normalized chest X-ray images and dataset files via our shared Google Drive link:
👉 [InnoTech Full Dataset (Google Drive)](https://drive.google.com/drive/folders/1scUXXUmXp3qK9Xfbn4icefYhZhrnq-J1?usp=sharing)
UI Application ->https://github.com/Moaz221/STELLARIS
