import streamlit as st
from transformers import pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sentiment-Aware Text Generator", layout="centered")

# --- LOAD AI MODELS (CACHED) ---
@st.cache_resource
def load_models():
    # Load Sentiment Analysis (DistilBERT is standard and fast)
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Load Text Generation (DistilGPT2 is lightweight and works on most laptops)
    generation_pipeline = pipeline("text-generation", model="distilgpt2")
    
    return sentiment_pipeline, generation_pipeline

# Load the models once when the app starts
try:
    sentiment_analyzer, text_generator = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- USER INTERFACE ---
st.title("🤖 AI Text Generator with Sentiment Analysis")
st.markdown("""
This application detects the sentiment of your prompt (Positive/Negative) 
and generates a short paragraph continuing that thought in a matching tone.
""")

# Input Area
user_input = st.text_area("Enter your prompt:", placeholder="e.g., The futuristic city was silent...", height=100)

# Sidebar Settings
st.sidebar.header("Generation Settings")
max_length = st.sidebar.slider("Max Length (Words)", 50, 200, 100)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.0, 0.7)

# --- MAIN LOGIC ---
if st.button("Generate Text"):
    if user_input.strip():
        with st.spinner('Analyzing sentiment and writing...'):
            
            # Step 1: Analyze Sentiment
            sentiment_result = sentiment_analyzer(user_input)[0]
            label = sentiment_result['label']
            score = sentiment_result['score']
            
            # Step 2: Display Sentiment to User
            st.info(f"Detected Sentiment: **{label}** ({score:.2f} confidence)")
            
            # Step 3: Construct the Prompt
            # We add a tonal cue to guide the AI based on the detected sentiment
            if label == "POSITIVE":
                tone_instruction = f"Write a happy and optimistic story about: {user_input}"
            else:
                tone_instruction = f"Write a dark or serious story about: {user_input}"
            
            # Step 4: Generate Content
            generated = text_generator(
                tone_instruction, 
                max_length=max_length, 
                num_return_sequences=1,
                temperature=temperature
            )
            
            result_text = generated[0]['generated_text']
            
            # (Optional) Clean the output to remove the instruction line if needed
            # For this MVP, we display the raw generation to show the AI's work
            
            st.subheader("Generated Output:")
            st.write(result_text)
            
    else:
        st.warning("Please enter a prompt first.")
