# Competitive Document Learning System

This project is a Python-based application where multiple machine learning algorithms compete to understand documents.  
Each algorithm is evaluated on three tasks:

- Text Classification
- Question Answering (QA)
- Document Summarization

The system compares algorithms based on **accuracy and execution time** and declares a winner.

---

## Features

- Compare multiple ML algorithms side-by-side
- Supports:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
- Automatic performance evaluation
- Clear explanation of how each algorithm works
- Failure point tracking for analysis
- Extensible design (easy to add new algorithms)

---

## Technologies Used

- Python 3.8+
- scikit-learn
- NumPy
- Streamlit (for UI, optional)
- NLTK (for basic NLP utilities)

---

## Project Structure

Learning_Algorithms/
│
├── app.py
├── competition_module.py
├── README.md
├── requirements.txt
│
├── pages/
│   ├── Algorithm_Builder.py
│   ├── Algorithm_Explorer.py
│   ├── Competition_Arena.py
│
└── __pycache__/
