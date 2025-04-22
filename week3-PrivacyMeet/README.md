# PrivacyMeet: Privacy-First Meeting Minutes Generator

![Privacy First](https://img.shields.io/badge/Privacy-First-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🔒 Why Privacy Matters in Meeting Assistants

Every time an employee connects a third-party AI assistant to your organization's meetings, they're effectively inviting an outsider to listen in on confidential discussions, product roadmaps, and strategic planning.

**Common Privacy Issues with Commercial Meeting Assistants:**
- Audio data processed on external servers
- Your sensitive business data potentially becoming part of training datasets
- Limited transparency about data retention and access policies
- Potential regulatory compliance issues in healthcare, finance, and legal sectors

## 📋 Project Overview

PrivacyMeet is an open-source, privacy-focused meeting assistant that allows you to generate professional meeting minutes from audio recordings **without sending any data to third-party services**. Everything runs locally, keeping your sensitive conversations private.

### Key Features

- **🛡️ Complete Privacy**: All processing happens locally on your machine
- **🔊 Local Transcription**: Uses Whisper for on-device speech-to-text
- **📝 AI-Generated Minutes**: Leverages Llama models for intelligent summarization
- **💾 Google Drive Integration**: Seamlessly access and save from your Drive
- **📱 User-Friendly Interface**: Simple Gradio UI for easy interaction
- **📊 Structured Output**: Clean, professional markdown format for minutes

## 🔍 Privacy Comparison

| Feature | PrivacyMeet | Commercial Solutions |
|---------|------------|---------------------|
| Data Processing Location | Local only | Cloud servers |
| Internet Required for Processing | No¹ | Yes |
| Data Retention | User controlled | Typically 30-90+ days |
| Data Used for Model Training | No | Often yes (opt-out may be available) |
| Compliance Self-Assessment | Full control | Limited control |
| Source Code Transparency | 100% open source | Proprietary |

¹ *Internet required only for initial model download*

## 🛠️ Technical Implementation

PrivacyMeet is built with:

- **Whisper** (Base/Small model): For accurate speech-to-text transcription
- **Llama 3.1 (8B)**: For intelligent meeting minutes generation
- **BitsAndBytes**: For model quantization to run efficiently on consumer hardware
- **Gradio**: For building the user interface
- **PyTorch**: For machine learning operations
- **Transformers**: For working with language models

## 📦 Installation

### Prerequisites

- Python 3.8+
- Google Colab (or local environment with GPU)
- Access to HuggingFace gated models (for Llama 3.1)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/LLM_Engineering.git
cd LLM_Engineering/week3-PrivacyMeet
```

2. Install dependencies:
```bash
pip install openai-whisper transformers accelerate bitsandbytes gradio torch
apt-get update && apt-get install -y ffmpeg
```

3. Set up authentication:
- Create a HuggingFace token with appropriate permissions
- Store it securely in your environment

## 🚀 Usage

### In Google Colab

1. Open the notebook in Google Colab
2. Mount your Google Drive
3. Store your HuggingFace token in Colab secrets
4. Run all cells
5. Use the Gradio interface to select or upload audio files
6. Generate and save meeting minutes

### Local Deployment

For organizations with sensitive information, we recommend running PrivacyMeet in a secure, local environment:

1. Adapt the code for local file handling
2. Set up environment variables for authentication
3. Run the application on an air-gapped machine for maximum security

## 📊 Data Flow Diagram

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Audio Input  │────▶│  Whisper    │────▶│  Transcript  │────▶│   Llama 3.1  │
│ (Local File) │     │ Transcription│     │ (Text Data)  │     │   Model      │
└──────────────┘     └─────────────┘     └──────────────┘     └──────┬───────┘
                                                                     │
                                                                     ▼
                                                              ┌──────────────┐
                                                              │   Meeting    │
                                                              │   Minutes    │
                                                              └──────────────┘
```

**No data leaves your local environment at any point**

## ⚠️ Security Considerations

- Always ensure you have permission to record and transcribe meetings
- Consider implementing local encryption for stored transcripts
- For highly sensitive meetings, run on machines with no internet access
- Regularly update dependencies to patch security vulnerabilities

## 🤔 Why Not Just Use Commercial Solutions?

While commercial solutions offer convenience, they come with significant privacy tradeoffs:

1. **Data Control**: Your sensitive conversations may be stored on third-party servers
2. **Potential Training Data**: Your data might be used to train future AI models
3. **Compliance Challenges**: Difficult to ensure compliance with industry regulations
4. **Vendor Lock-in**: Dependency on external services for critical business functions
5. **Cost**: Subscription fees for services vs. one-time setup of open-source solution

## 🤝 Contributing

Contributions to improve PrivacyMeet are welcome! Some areas for enhancement:

- Adding local encryption for stored transcripts
- Implementing speaker diarization
- Supporting additional languages
- Creating standalone desktop application
- Adding automated action item tracking

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for providing access to state-of-the-art models
- The Whisper and Llama teams for their incredible work
- Open-source community for making privacy-first AI accessible to everyone
