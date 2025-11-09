# Next Word Prediction using Hamlet Dataset

A deep learning project that predicts the **next possible word** in a sentence using the **text of William Shakespeare’s *Hamlet***.  
The model learns the structure, style, and vocabulary of the play to generate context-aware word predictions.

---

## 📘 Overview

This project uses **Natural Language Processing (NLP)** and **deep learning (LSTM-based model)** to build a predictive language model trained on the *Hamlet* corpus.  
Given a sequence of words, the model predicts the **next most probable word** in Shakespearean language style.

---

## Features

- Preprocessed and tokenized the full *Hamlet* text corpus  
- Built word sequences for next-word prediction  
- Used **LSTM layers** to learn contextual word dependencies  
- Achieved smooth prediction for text continuation  
- Interactive **Streamlit app** for live text generation


## Tech Stack

- **Language:** Python  
- **Libraries:** TensorFlow / Keras, NumPy, Pandas, NLTK, Matplotlib  
- **Deployment:** Streamlit  
- **Dataset:** *Hamlet* by William Shakespeare (from Project Gutenberg)


## Model Architecture

- **Embedding Layer:** Converts words to dense vectors  
- **LSTM Layer(s):** Captures sequence and context relationships  
- **Dense Layer:** Outputs probability distribution over vocabulary  
- **Activation:** Softmax  

