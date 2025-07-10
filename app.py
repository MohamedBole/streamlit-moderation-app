import os
import requests
import json
import torch
import streamlit as st
from PIL import Image # For image processing
# Removed: from google.colab import userdata # Not available on Streamlit Community Cloud
# Removed: from google.colab import drive # Not available on Streamlit Community Cloud
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chains import LLMChain
from typing import Dict, Any

# --- Configuration and Setup ---

# Define the device for PyTorch models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.info(f"Using device: {device}")

# --- Load Fine-tuned DistilBert Model (Soft Classifier) ---
# IMPORTANT: You MUST replace these placeholders with the actual Hugging Face Model IDs
# where you have uploaded your fine-tuned model and tokenizer.
# Example: "your-huggingface-username/my-fine-tuned-distilbert-lora"
FINE_TUNED_MODEL_ID = "your-huggingface-org/your-fine-tuned-distilbert-model" # <<< REPLACE THIS PLACEHOLDER
FINE_TUNED_TOKENIZER_ID = "your-huggingface-org/your-fine-tuned-distilbert-tokenizer" # <<< REPLACE THIS PLACEHOLDER (often same as base model or fine-tuned model)

@st.cache_resource # Cache the model loading for performance
def load_soft_classifier_model():
    st.write("Loading soft classifier model (DistilBert)... This may take a moment.")
    try:
        # Load tokenizer from Hugging Face Model Hub
        tokenizer = DistilBertTokenizer.from_pretrained(FINE_TUNED_TOKENIZER_ID)
        num_labels = 9 # Adjust this if your number of classes is different
        base_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', # Load the original base model first
            num_labels=num_labels
        )
        # Load the PEFT (LoRA) adapters onto the base model from Hugging Face Model Hub
        model = PeftModel.from_pretrained(base_model, FINE_TUNED_MODEL_ID)
        model.to(device)
        model.eval()
        st.success("Soft classifier model and tokenizer loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading soft classifier model. Please ensure '{FINE_TUNED_MODEL_ID}' and '{FINE_TUNED_TOKENIZER_ID}' are correct and accessible on Hugging Face Model Hub, or adjust paths if models are in your repo: {e}")
        st.stop() # Stop Streamlit app if model cannot be loaded

tokenizer, soft_classifier_model = load_soft_classifier_model()

# Define your label mapping for the soft classifier (must match your training)
label_map = {
    0: 'Child Sexual Exploitation',
    1: 'Elections',
    2: 'Non-Violent Crimes',
    3: 'Safe',
    4: 'Sex-Related Crimes',
    5: 'Suicide & Self-Harm',
    6: 'Unknown S-Type',
    7: 'Violent Crimes',
    8: 'Unsafe'
}

# --- Load BLIP Vision-Language Model ---
@st.cache_resource # Cache the model loading for performance
def load_blip_model():
    st.write("Loading BLIP image captioning model... This may take a moment.")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.to(device)
        blip_model.eval()
        st.success("BLIP model loaded successfully!")
        return processor, blip_model
    except Exception as e:
        st.error(f"Error loading BLIP model: {e}")
        st.stop() # Stop Streamlit app if model cannot be loaded

blip_processor, blip_model = load_blip_model()


# --- API Key Setup (for OpenRouter) ---

# Retrieve API Keys from Streamlit Secrets
# Use st.secrets["KEY_NAME"] for direct access, or st.secrets.get("KEY_NAME", default_value)
try:
    openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
    # It's crucial to strip any whitespace/newline characters that might be appended
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key.strip()
    st.success("OpenRouter API Key loaded from Streamlit Secrets.")
except KeyError:
    st.error("`OPENROUTER_API_KEY` not found in Streamlit Secrets. Please add it to your app's secrets.")
    st.stop()

os.environ["OPENROUTER_BASE_URL"] = "https://openrouter.ai/api/v1" # Standard OpenRouter base URL
# Optional: Set these for OpenRouter analytics if you have a site URL/name
# Use .get() with a default value to prevent errors if secrets are not set
os.environ["YOUR_SITE_URL"] = st.secrets.get("YOUR_SITE_URL", "https://streamlit.app") # Default for Streamlit Cloud
os.environ["YOUR_SITE_NAME"] = st.secrets.get("YOUR_SITE_NAME", "Streamlit Moderation App")


# --- Common LLM Initialization Function for OpenRouter ---
def initialize_openrouter_llm(model_name: str, temperature: float = 0.7):
    return ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        model_name=model_name,
        temperature=temperature,
        model_kwargs={
            "headers": {
                "HTTP-Referer": os.getenv("YOUR_SITE_URL"),
                "X-Title": os.getenv("YOUR_SITE_NAME"),
            }
        },
    )

# --- Stage 1: Hard Filter (Llama Guard API Integration via OpenRouter) ---

def call_llama_guard_api(text: str) -> Dict[str, Any]:
    """
    Calls the actual Llama Guard API via OpenRouter to moderate the given text.
    """
    openrouter_api_key_clean = os.getenv("OPENROUTER_API_KEY")
    your_site_url = os.getenv("YOUR_SITE_URL")
    your_site_name = os.getenv("YOUR_SITE_NAME")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key_clean}",
        "Content-Type": "application/json",
        "HTTP-Referer": your_site_url,
        "X-Title": your_site_name,
    }
    data = {
        "model": "meta-llama/llama-guard-3-8b", # The specific Llama Guard model on OpenRouter
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.0 # Guard models should typically have low temperature for deterministic output
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()

        guard_output = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if guard_output.lower().startswith("unsafe"):
            category_detail = guard_output[len("unsafe "):].strip() if len(guard_output) > len("unsafe") else "General Unsafe"
            return {"flagged": True, "category": f"Llama Guard: {category_detail}"}
        elif guard_output.lower().startswith("safe"):
            return {"flagged": False, "category": "Llama Guard: Safe"}
        else:
            return {"flagged": True, "category": f"Llama Guard: Unexpected output - '{guard_output}' (Raw: {response_json})"}

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling OpenRouter Llama Guard API: {e}")
        return {"flagged": True, "category": f"Llama Guard API Error: {e}"}
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response from OpenRouter Llama Guard API: {e}")
        return {"flagged": True, "category": f"Llama Guard API JSON Error: {e}"}

def hard_filter_stage(input_data: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_data["question"]
    st.write(f"**Stage 1: Running Hard Filter (Llama Guard via OpenRouter)**")
    moderation_result = call_llama_guard_api(user_input)

    if moderation_result.get("flagged"):
        st.error(f"**[Hard Filter] Input flagged as UNSAFE:** {moderation_result.get('category')}. Halting processing.")
        raise ValueError(f"Content blocked by hard moderation filter: {moderation_result.get('category')}")
    else:
        st.success(f"**[Hard Filter] Input deemed SAFE.** Proceeding to soft classifier.")
        return input_data

# --- Stage 2: Soft Classifier (Fine-tuned DistilBert) ---

def soft_classifier_stage(input_data: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_data["question"]
    st.write(f"**Stage 2: Running Soft Classifier (Fine-tuned DistilBert)**")

    encoded_input = tokenizer(
        user_input,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = soft_classifier_model(**encoded_input)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_label_id = logits.argmax(-1).item()

    predicted_category = label_map[predicted_label_id]
    confidence = probabilities[predicted_label_id].item()

    st.info(f"**[Soft Classifier] Predicted Category:** `{predicted_category}` (Confidence: `{confidence:.4f}`)")

    input_data["soft_classification"] = {
        "category": predicted_category,
        "confidence": confidence,
        "all_probabilities": {label_map[i]: prob.item() for i, prob in enumerate(probabilities)}
    }
    return input_data

# --- BLIP Captioning Function ---
def generate_caption(image: Image.Image) -> str:
    st.write("Generating image caption...")
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = blip_processor(images=image, text="", return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "Error: Could not generate caption."

# --- Main LLM Chain (for answering the question) ---

llm_main_qa = initialize_openrouter_llm(model_name="mistralai/mistral-7b-instruct", temperature=0.7)

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm_main_qa)

# --- Dual-Stage Moderation Workflow + Main LLM Chain ---
full_moderation_and_qa_chain = (
    RunnablePassthrough.assign(question=lambda x: x["question"])
    | RunnableLambda(hard_filter_stage)
    | RunnableLambda(soft_classifier_stage)
    | llm_chain
)

# --- Streamlit UI ---

st.title("🛡️ Dual-Stage Content Moderation & Q&A Pipeline")
st.markdown("This application moderates user input (text or image captions) in two stages: a hard filter (Llama Guard via OpenRouter) and a soft classifier (fine-tuned DistilBert). If content passes, it's sent to a Q&A LLM.")

user_input_text = st.text_area("✍️ Enter text here:", height=100, placeholder="Type your message...")
uploaded_file = st.file_uploader("🖼️ Or upload an image:", type=["png", "jpg", "jpeg"])

content_to_moderate = None
processing_image = False

if uploaded_file is not None:
    processing_image = True
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    with st.spinner("Generating caption from image..."):
        image = Image.open(uploaded_file)
        caption = generate_caption(image)
        st.write(f"**Generated Caption:** `{caption}`")
        content_to_moderate = caption
elif user_input_text:
    content_to_moderate = user_input_text

if st.button("🚀 Process Input") and content_to_moderate:
    st.subheader("Moderation Results:")
    try:
        with st.spinner("Running moderation pipeline..."):
            final_result = full_moderation_and_qa_chain.invoke({"question": content_to_moderate})

        st.success("Content passed all moderation stages!")
        st.write("---")
        st.subheader("Soft Classifier Details:")
        st.json(final_result["soft_classification"])
        st.write("---")
        st.subheader("LLM Answer:")
        st.info(final_result["text"])

    except ValueError as e:
        st.error(f"**🚫 Moderation Blocked:** {e}")
    except Exception as e:
        st.exception(f"An unexpected error occurred during processing: {e}")
elif st.button("🚀 Process Input") and not content_to_moderate:
    st.warning("Please enter some text or upload an image to process.")

