# 📝 Handwriting Recognition Web App (OCR)

A full-stack handwriting recognition application built with:

- 🔙 **Flask** backend using a trained CTC model (TensorFlow/Keras)
- 🔜 **React (Vite)** frontend styled with Tailwind CSS

It allows users to upload an image of handwritten **sentences**, splits them into **words**, and returns the **predicted sentence**.

---

## 🧠 Model Highlights

- CNN + RNN + CTC Loss-based architecture
- Greedy decoding with consistent vocabulary
- Trained on word-level handwriting dataset
- Sentence image → word images → predicted text

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/handwriting-ocr.git
cd handwriting-ocr
````

---

## 🔙 Backend Setup (Flask + TensorFlow)

### ✅ Create environment and install dependencies

```bash
cd backend
conda create -n ocr python=3.10
conda activate ocr
pip install -r requirements.txt
```

### ✅ Start the backend server

```bash
python run.py
```

This will start the Flask API at `http://localhost:5000`.

---

## 🔜 Frontend Setup (Vite + React + Tailwind CSS)

### ✅ Install dependencies

```bash
cd frontend
npm install
```

### ✅ Start the frontend dev server

```bash
npm run dev
```

Then open your browser to `http://localhost:5173`.

---

## 🧪 Usage

1. Open the frontend UI in the browser.
2. Upload a sentence-level handwriting image (`.png`, `.jpg`).
3. The app will display the predicted sentence after calling the Flask backend.

---

## 📦 Backend API

### Endpoint: `POST /predict`

**Request**:

* `multipart/form-data` with a field named `image`

**Response**:

```json
{
  "prediction": "This is an example"
}
```

---

## 📄 Vocabulary Note

Ensure `vocab.txt` in the backend `models/` folder **matches the vocabulary used during model training**. If mismatched, predictions will be incorrect.

---

## 📄 License

MIT License

---
Made with ❤️ by [Barnik Podder](https://github.com/Barnik-Podder)
