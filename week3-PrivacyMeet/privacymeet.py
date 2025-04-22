# -*- coding: utf-8 -*-
"""
 PrivacyMeet: Privacy-First Meeting Minutes Generator
"""


# First, installing dependencies
!pip install -q openai-whisper transformers accelerate bitsandbytes gradio
!apt-get -qq update && apt-get -qq install -y ffmpeg

# Standard imports
import os
import json
import time
import datetime
import logging
from pathlib import Path

# Google Drive integration
from google.colab import drive
import gradio as gr
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PrivacyMeet')

# Constants
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Gated model that you have access to

# ============================================================================
# Authentication and Setup
# ============================================================================

# Log in to Hugging Face - use the token you've saved in Colab secrets
from google.colab import userdata
hf_token = userdata.get('TOKEN')  # Make sure this matches your secret name
if hf_token:
    login(token=hf_token)
    logger.info("Successfully logged in to Hugging Face")
else:
    logger.warning("No Hugging Face token found. Attempting to proceed without authentication.")

# Mount Google Drive
drive.mount('/content/drive')

# ============================================================================
# Audio Processing
# ============================================================================

def setup_whisper():
    """Set up the Whisper model for transcription"""
    logger.info(f"Loading Whisper {WHISPER_MODEL_SIZE} model...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    logger.info("Whisper model loaded")
    return model

def transcribe_audio(audio_path, whisper_model):
    """Transcribe audio using Whisper"""
    logger.info(f"Transcribing audio: {audio_path}")
    start_time = time.time()

    result = whisper_model.transcribe(audio_path)
    transcription = result["text"]

    duration = time.time() - start_time
    logger.info(f"Transcription completed in {duration:.2f} seconds")

    return transcription

# ============================================================================
# LLM Setup and Minutes Generation
# ============================================================================

def setup_llm():
    """Set up the LLM for minutes generation"""
    logger.info(f"Loading LLM model: {LLM_MODEL}")

    # Quantization configuration for efficient loading
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        quantization_config=quant_config
    )

    logger.info("LLM model loaded successfully")
    return model, tokenizer

def generate_minutes(transcription, audio_filename, model, tokenizer):
    """Generate meeting minutes from transcription with improved title and attendee detection"""
    logger.info(f"Generating minutes for {audio_filename}")

    # Create system message emphasizing the title and attendee extraction
    system_message = """You are a professional meeting minutes generator.
    Create detailed, well-formatted meeting minutes from the transcript.

    IMPORTANT:
    1. Create a meaningful, specific title for the meeting based on the content (e.g., "Minutes of the Denver City Council Meeting" rather than "Meeting Minutes")
    2. Extract the full names and titles of all attendees mentioned in the transcript (e.g., "Councilman Lopez", "Councilwoman Ortega")
    3. Format the attendees as a bulleted list
    4. Include a  summary, key discussion points, decisions made, and action items
    5. Format output in clear markdown with proper headings and structure"""

    # Create user prompt
    base_name = os.path.splitext(os.path.basename(audio_filename))[0].replace("_", " ")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    user_prompt = f"""Based on this transcript, generate professional meeting minutes.

The audio file was named: {base_name}
Today's date: {date_str}

Remember to:
1. Create a proper title reflecting the meeting's purpose
2. List all attendees with their titles/positions
3. Summarize the key points and decisions
4. Highlight any action items or follow-ups with owners

Transcript:
{transcription}"""

    # Prepare input for the model
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    # Generate minutes
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(inputs, max_new_tokens=2000)
    response = tokenizer.decode(outputs[0])

    # Clean up the response
    response = response.split("<|end_header_id|>")[-1].strip()
    response = response.replace("<|eot_id|>", "")

    logger.info(f"Minutes generation complete")

    return response

# ============================================================================
# File Handling
# ============================================================================

def list_audio_files(base_dir="/content/drive/MyDrive", extensions=(".mp3", ".wav", ".m4a")):
    """List audio files in Google Drive"""
    audio_files = []

    # Recursively search for audio files (limit depth and number for performance)
    max_depth = 3
    max_files = 20
    count = 0

    def search_dir(dir_path, current_depth=0):
        nonlocal count
        if current_depth > max_depth or count >= max_files:
            return

        try:
            for item in os.listdir(dir_path):
                if count >= max_files:
                    return

                full_path = os.path.join(dir_path, item)
                if os.path.isfile(full_path) and full_path.lower().endswith(extensions):
                    audio_files.append(full_path)
                    count += 1
                elif os.path.isdir(full_path):
                    search_dir(full_path, current_depth + 1)
        except (PermissionError, FileNotFoundError):
            pass  # Skip directories we can't access

    search_dir(base_dir)
    return audio_files

def save_minutes_to_file(minutes_content, audio_path):
    """Save minutes to a file next to the audio file"""
    try:
        base_path = os.path.splitext(audio_path)[0]
        minutes_path = f"{base_path}_minutes.md"

        with open(minutes_path, 'w') as f:
            f.write(minutes_content)

        return minutes_path
    except Exception as e:
        logger.error(f"Error saving minutes: {str(e)}")
        return None

# ============================================================================
# Main Processing Function
# ============================================================================

def process_audio_file(audio_path, whisper_model, llm_model, llm_tokenizer):
    """Process an audio file to generate meeting minutes"""
    try:
        # Transcribe the audio
        transcription = transcribe_audio(audio_path, whisper_model)

        # Generate minutes with better title and attendee extraction
        minutes = generate_minutes(transcription, audio_path, llm_model, llm_tokenizer)

        return minutes

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return f"Error processing audio: {str(e)}"

# ============================================================================
# Gradio UI
# ============================================================================

def create_gradio_interface():
    """Create Gradio interface with simpler, more reliable implementation"""
    # Set up models
    whisper_model = setup_whisper()
    llm_model, llm_tokenizer = setup_llm()

    # Get available audio files
    audio_files = list_audio_files()

    with gr.Blocks(title="PrivacyMeet - Private Meeting Minutes") as app:
        gr.Markdown("# PrivacyMeet")
        gr.Markdown("Generate meeting minutes from audio recordings without sharing your data with third parties.")

        with gr.Row():
            with gr.Column(scale=1):
                # Input options
                input_type = gr.Radio(
                    ["Select from Drive", "Upload File"],
                    label="Input Method",
                    value="Select from Drive"
                )

                # Drive file selection
                file_dropdown = gr.Dropdown(
                    label="Select Audio File from Drive",
                    choices=audio_files,
                    visible=True
                )

                # File upload
                file_upload = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    visible=False
                )

                refresh_btn = gr.Button("Refresh File List")
                process_btn = gr.Button("Generate Minutes", variant="primary")

            with gr.Column(scale=2):
                # Status and output
                status_output = gr.Textbox(label="Status", interactive=False)
                minutes_output = gr.Markdown(label="Generated Minutes")
                save_btn = gr.Button("Save to Drive")
                download_btn = gr.Button("Download as Markdown")
                saved_path = gr.Textbox(label="Saved Location", visible=True)

        # Simple file download component
        download_file = gr.File(label="Download Minutes", visible=False)

        # Handle input type change
        def toggle_input_visibility(input_type_value):
            return {
                file_dropdown: gr.update(visible=input_type_value == "Select from Drive"),
                file_upload: gr.update(visible=input_type_value == "Upload File")
            }

        input_type.change(
            toggle_input_visibility,
            inputs=[input_type],
            outputs=[file_dropdown, file_upload]
        )

        # Handle refreshing the file list
        def refresh_files():
            new_files = list_audio_files()
            return gr.Dropdown(choices=new_files)

        refresh_btn.click(
            refresh_files,
            inputs=[],
            outputs=[file_dropdown]
        )

        # Handle processing the selected file
        def process_selected_file(input_type_value, file_path_dropdown, file_path_upload):
            # Determine which file path to use
            file_path = file_path_dropdown if input_type_value == "Select from Drive" else file_path_upload

            if not file_path:
                return "Please select an audio file", ""

            try:
                # Process the file
                minutes = process_audio_file(
                    file_path,
                    whisper_model,
                    llm_model,
                    llm_tokenizer
                )

                return "Minutes generated successfully!", minutes
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, ""

        process_btn.click(
            process_selected_file,
            inputs=[input_type, file_dropdown, file_upload],
            outputs=[status_output, minutes_output]
        )

        # Handle saving minutes to a file
        def save_minutes(input_type_value, file_path_dropdown, file_path_upload, minutes_content):
            if not minutes_content:
                return "Please generate minutes first"

            # Determine which file path to use
            file_path = file_path_dropdown if input_type_value == "Select from Drive" else file_path_upload

            if not file_path:
                return "No audio file selected"

            saved_path = save_minutes_to_file(minutes_content, file_path)

            if saved_path:
                return f"Minutes saved to {saved_path}"
            else:
                return "Error saving minutes"

        save_btn.click(
            save_minutes,
            inputs=[input_type, file_dropdown, file_upload, minutes_output],
            outputs=[saved_path]
        )

        # Handle downloading minutes
        def create_download_file(minutes_content):
            if not minutes_content:
                return None

            temp_path = "/tmp/meeting_minutes.md"
            with open(temp_path, 'w') as f:
                f.write(minutes_content)

            return temp_path

        download_btn.click(
            create_download_file,
            inputs=[minutes_output],
            outputs=[download_file]
        )

    return app

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""
    logger.info("Starting PrivacyMeet application")

    # Create and launch the Gradio interface
    app = create_gradio_interface()
    app.launch(inline=True, share=True)

if __name__ == "__main__":
    main()
