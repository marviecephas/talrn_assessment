# Sentiment-Aligned AI Text Generator

## 📌 Project Overview
This project is an AI-powered application designed to demonstrate the integration of **Natural Language Processing (NLP)** models into a user-friendly web interface. It performs two distinct tasks:
1.  **Sentiment Analysis:** Classifies user input as Positive or Negative.
2.  **Conditional Text Generation:** Generates a paragraph of text that aligns with the detected sentiment.

The goal is to show how pre-trained ML models can be chained together in a functional frontend application.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Frontend:** Streamlit
* **ML Libraries:** Hugging Face Transformers, PyTorch/TensorFlow

## ⚙️ Methodology
The project utilizes the **Hugging Face Transformers** library to access state-of-the-art pre-trained models:

1.  **Sentiment Model:** Uses the default `distilbert-base-uncased-finetuned-sst-2-english`. This model was selected for its speed and high accuracy in binary sentiment classification.
2.  **Generation Model:** Uses `distilgpt2`. This lightweight version of GPT-2 was chosen to ensure the application runs efficiently on standard consumer hardware (CPUs) without requiring a GPU.
3.  **Logic:** The application uses "Prompt Engineering" to guide the text generator. Based on the sentiment detected by the first model, a specific tonal instruction is prepended to the user's prompt before generation.

## 🚀 Installation & Setup
To run this project locally on your machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/talrn_assessment.git](https://github.com/YOUR_USERNAME/talrn_assessment.git)
    cd talrn_assessment
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 🧠 Challenges & Reflections
* **Resource Management:** Running two transformer models simultaneously can be memory-intensive. I optimized this by using "Distil" versions of the models and implementing caching (`@st.cache_resource`) in Streamlit to prevent reloading models on every user interaction.
* **Sentiment Steering:** Ensuring the text generator strictly adheres to the detected sentiment is challenging without fine-tuning. I addressed this by dynamically altering the input prompt to include emotive keywords (e.g., "Write a happy story about...") which significantly improved the tonal alignment of the output.

## 🔮 Future Improvements
* Deploying the application to Streamlit Community Cloud.
* Adding a "Manual Override" feature to let users force a specific sentiment.
* Implementing a more advanced model (like GPT-2 Medium) for higher quality text generation.
