# 🔬 NeuroGuard AI: Clinical-Grade MRI Analysis Suite

[![NeuroGuard AI](https://img.shields.io/badge/AI-NeuroGuard-blueviolet?style=for-the-badge)](https://github.com/Jahnavigajjela213/brain-tumor-mri-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Deployed on Render](https://img.shields.io/badge/Backend-Render-006bd1?style=for-the-badge)](https://render.com)
[![Deployed on Vercel](https://img.shields.io/badge/Frontend-Vercel-black?style=for-the-badge)](https://vercel.com)

**NeuroGuard AI** is a state-of-the-art medical imaging dashboard designed for automated tumor segmentation and clinical survival projections. Built on the **BraTS 2020 dataset**, it utilizes a custom **Attention U-Net v2.0** for precise volumetric analysis and an ensemble **SurvivalNet** for prognosis estimation.

---

## 🌟 Key Features

- **Automated Tumor Segmentation**: Real-time identification of **Whole Tumor (WT)**, **Tumor Core (TC)**, and **Enhancing Tumor (ET)** using the BraTS 2020 clinical legend.
- **Survival Probability Projection**: Personalized survival estimates (in days) based on tumor heterogeneity, volume, and intensity ratios—mimicking institutional research standards.
- **Interactive Clinical Review**: A high-fidelity "Slide to Compare" interface allowing researchers to toggle between original MRI slices and clinical overlays.
- **Dynamic Inference Engine**: Adaptive thresholding and stochastic multimodal simulation for varied and realistic AI responses.
- **Institutional Research Prototype**: Designed for demonstration and research support in oncology and neurology domains.

---

## 🚀 Technology Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Deep Learning**: [PyTorch](https://pytorch.org/) (U-Net & SurvivalNet)
- **Image Processing**: OpenCV, Scikit-Image, NumPy
- **Server**: Uvicorn with PORT binding for cloud deployment

### Frontend
- **Library**: [React](https://reactjs.org/) (Vite + TypeScript)
- **Styling**: TailwindCSS & Lucide Icons
- **Deployment**: Vercel (CI/CD)
- **State Management**: React Hooks & Axios

---

## 📂 Project Structure

```bash
.
├── backend/               # FastAPI Deep Learning Engine
│   ├── app/
│   │   ├── models/        # PyTorch Model Architectures (U-Net, SurvivalNet)
│   │   ├── routes/        # API Endpoints (Inference, Dataset, Preprocessing)
│   │   └── utils/         # Image processing & clinical features
│   ├── dataset/           # BraTS 2020 sample images for explorer
│   ├── debug_render.py    # Deployment entry point
│   ├── requirements.txt
│   └── train.py           # Training pipeline for U-Net v2.0
├── frontend/              # React MRI Dashboard
│   ├── src/
│   │   ├── components/    # PredictionCard, SegmentationDisplay, Dashboard
│   │   ├── services/      # Axios API Client
│   │   └── App.tsx        # Main orchestration
│   └── tailwind.config.ts
├── render.yaml           # Infrastructure-as-Code for Render
└── README.md             # This file
```

---

## 🛠️ Local Development

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python debug_render.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The system will be available at `http://localhost:5173`. Make sure `VITE_API_URL` is set to `http://localhost:8001`.

---

## 🛰️ Deployment

The project is pre-configured for **Continuous Deployment**:
- **Backend (Render)**: Automatically builds using `backend/requirements.txt` and triggers via Git push.
- **Frontend (Vercel)**: Configured for Vite/React deployment. Ensure `VITE_API_URL` points to your Render backend link.

---

## ⚖️ Clinical Disclaimer

> [!CAUTION]
> **NOT FOR CLINICAL DIAGNOSIS.**
> This is a research prototype intended for institutional demonstration and educational purposes. The AI-generated segmentation and survival projections must not be used for medical decisions or patient care.

---

## 🖋️ License

This project is licensed under the **MIT License**.

---
*Developed by Jahnavigajjela213*
