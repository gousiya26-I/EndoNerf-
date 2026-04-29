# Endo-NeRF++

### Uncertainty-Aware Sampling and Multi-Resolution Encoding for Dynamic Surgical Scene Reconstruction

<p align="center">
  <img src="assassets/Diagram final.jpg" width="800"/>
</p>

---

## 📌 Overview
Endo-NeRF++ is a scalable framework for dynamic surgical scene reconstruction.  
It extends EndoNeRF by integrating:

- Multi-resolution hash encoding for compact scene representation  
- Feature blending across temporal dimensions  
- Uncertainty-aware modeling for robust reconstruction  

These improvements enable efficient learning and better performance in deformable and texture-sparse surgical environments.

---

## 🧠 Method
Our approach combines:
- Temporal deformation field (Direct Temporal NeRF)
- Multi-resolution hash encoding
- Temporal feature blending
- Uncertainty-aware learning

---

## 📊 Results
Endo-NeRF++ improves reconstruction quality and robustness compared to baseline EndoNeRF.

(Add your results images here later)

---

## 🔗 Project Page
https://github.com/gousiya26-I/EndoNerf-plusplus

---

## ⚙️ Installation
```bash
git clone https://github.com/gousiya26-I/EndoNerf-plusplus.git
cd EndoNerf-plusplus
pip install -r requirements.txt
