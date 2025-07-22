# 📜 Next Word Prediction with LSTM (Shakespeare Edition)🧙‍♂️

This project demonstrates a deep learning-based **Next Word Prediction** model using **LSTM (Long Short-Term Memory)** networks. The model is trained on Shakespearean texts to generate stylistic continuations of phrases, and is deployed using **Streamlit** for a lightweight and interactive web interface.

---

## 🧠 Model Architecture

- **Embedding Layer:** Converts words to dense vectors of fixed size.
- **2 LSTM Layers:** Learn temporal patterns from sequences.
- **Dropout:** Prevents overfitting.
- **Dense Softmax Output:** Outputs probability distribution across the vocabulary.


## 📁 Project Structure
```bash
.
├── app.py                            # Streamlit app
├── requirements.txt                  # Python dependencies
└── README.md
```

## ⚙️ Installation

### Install dependencies:

```bash
pip install -r requirements.txt
```
### Run the app:

```bash
streamlit run app.py
```
ENJOY!🎉