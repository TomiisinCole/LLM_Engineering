# Week 2: Data Science Mentor AI Assistant

## Project Overview
This is my Week 2 project implementation from the "Mastering Generative AI and LLMs: An 8-Week Hands-On Journey" course. The Data Science Mentor is an AI-powered assistant that helps data scientists understand complex programming concepts, algorithms, and libraries.

## Features
- Interactive chat interface built with Gradio
- Support for multiple LLM models:
  - OpenAI models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
  - Claude models (Claude 3 Opus, Claude 3.7 Sonnet, Claude 3.5 Haiku)
- Python code execution tool to run and test data science code snippets
- Documentation lookup tool for pandas, numpy, scikit-learn, and other libraries
- Voice response capability with text-to-speech summarization
- Multi-modal interaction allowing text and audio outputs

## Technical Capabilities
The Data Science Mentor can:
- Explain complex programming concepts with clear examples
- Debug Python code for data science tasks
- Provide documentation for data science libraries and functions
- Execute Python code snippets with data science libraries
- Generate voice explanations of complex topics

## Technologies Used
- Python
- Gradio for the web interface
- OpenAI API (GPT models and text-to-speech)
- Anthropic API (Claude models)
- Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn integrations

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Anthropic API key (optional, for Claude models)

### Installation

1. Navigate to this project directory:
```bash
cd week2-ds-mentor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

4. Run the application:
```bash
python ds_mentor.py
```
Or open and run the Jupyter notebook:
```bash
jupyter notebook Project_1_DS_mentor.ipynb
```

## Usage Examples
- "Explain the difference between .loc and .iloc in pandas"
- "How do I handle missing values in a DataFrame?"
- "Can you help me understand how Random Forest works?"
- "What does sklearn.preprocessing.StandardScaler do?"


## Next Steps
For the full course journey, check out the [main repository README](../README.md).

## License
[MIT License](../LICENSE)
