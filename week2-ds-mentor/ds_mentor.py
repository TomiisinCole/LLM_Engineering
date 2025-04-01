"""
Data Science Mentor AI Assistant
Week 2 Project Implementation
Mastering Generative AI and LLMs: An 8-Week Hands-On Journey
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import anthropic
from IPython.display import Markdown, display, update_display

# Initialization
# Load environment variables from .env file
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Print API key status for debugging
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")
if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

# Default to gpt-4o-mini, but allow switching
DEFAULT_MODEL = "gpt-4o-mini"
openai = OpenAI()
claude = anthropic.Anthropic()

# System message that defines the AI assistant's role and capabilities
system_message = """You are a helpful Data Science Programming Mentor. You specialize in helping
data scientists understand complex programming concepts, algorithms, and libraries.
Your expertise includes:
- Python programming for data science
- Data manipulation with pandas and numpy
- Machine learning with scikit-learn
- Data visualization with matplotlib and seaborn
- Statistical analysis and techniques
Provide clear, concise explanations with practical examples. When explaining code, break down
the logic and syntax in a way that's easy to understand. Consider the user's proficiency level
and adapt your responses accordingly.
When appropriate, suggest best practices, potential optimizations, and common pitfalls to avoid.
"""

# Simple tool to execute Python code
def execute_code(code_snippet):
    """
    Execute Python code snippet in a safe environment with data science libraries
    available and return the results.
    """
    print(f"Tool execute_code called with code: {code_snippet[:50]}...")
    import sys
    from io import StringIO
    import traceback
    
    # Create string buffers to capture stdout and stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    # Prepare the return dictionary
    result = {
        "execution_successful": False,
        "output": "",
        "error": ""
    }
    
    # Save the original stdout and stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_capture, stderr_capture
    
    try:
        # Execute the code with common data science libraries available
        exec(code_snippet, {
            "__builtins__": __builtins__,
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "plt": __import__("matplotlib.pyplot"),
            "sns": __import__("seaborn")
        })
        result["execution_successful"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Get the output
        result["output"] = stdout_capture.getvalue()
        if stderr_capture.getvalue():
            result["error"] += stderr_capture.getvalue()
        
        # Restore stdout and stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr
    
    return result

# Define the code execution tool for OpenAI
code_tool = {
    "name": "execute_code",
    "description": "Execute Python code for data science tasks and return the results",
    "parameters": {
        "type": "object",
        "properties": {
            "code_snippet": {
                "type": "string",
                "description": "Valid Python code to execute"
            }
        },
        "required": ["code_snippet"]
    }
}

# Add all tools to a list
tools = [{"type": "function", "function": code_tool}]

# Documentation lookup tool
def lookup_documentation(query, library=None):
    """
    Search for documentation on data science libraries, functions, and methods.
    """
    print(f"Tool lookup_documentation called for {query} in {library or 'all libraries'}")
    import importlib
    import inspect
    
    # Define libraries we can search documentation for
    available_libraries = {
        "pandas": "pd",
        "numpy": "np",
        "matplotlib.pyplot": "plt",
        "seaborn": "sns",
        "sklearn": "sklearn"
    }
    
    result = {
        "found": False,
        "documentation": "",
        "signature": "",
        "source_link": ""
    }
    
    try:
        # If library is specified, only search there
        libraries_to_search = [library] if library and library in available_libraries else available_libraries.keys()
        
        for lib_name in libraries_to_search:
            try:
                # Try to import the library
                module = importlib.import_module(lib_name)
                
                # Parse the query to get attribute path
                parts = query.split('.')
                # Start with the base module
                obj = module
                
                for part in parts[1:] if lib_name in parts[0] else parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                
                # Get documentation
                doc = inspect.getdoc(obj)
                if doc:
                    result["found"] = True
                    result["documentation"] = doc
                    
                    # Get function signature if applicable
                    if callable(obj):
                        result["signature"] = str(inspect.signature(obj))
                    
                    # Add link to official documentation
                    if lib_name == "pandas":
                        result["source_link"] = f"https://pandas.pydata.org/docs/reference/api/{query}.html"
                    elif lib_name == "numpy":
                        result["source_link"] = f"https://numpy.org/doc/stable/reference/{query}.html"
                    
                    break
            except (ImportError, AttributeError) as e:
                continue
        
        if not result["found"]:
            result["documentation"] = f"Documentation for '{query}' not found. Please check the spelling or try a different query."
    except Exception as e:
        import traceback
        result["documentation"] = f"Error looking up documentation: {str(e)}\n{traceback.format_exc()}"
    
    return result

# Define the documentation tool for OpenAI
doc_tool = {
    "name": "lookup_documentation",
    "description": "Search for documentation on data science libraries, functions, and methods",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The function, class, or method to look up (e.g., 'pandas.DataFrame.groupby')"
            },
            "library": {
                "type": "string",
                "description": "Optional specific library to search in",
                "enum": ["pandas", "numpy", "matplotlib.pyplot", "seaborn", "sklearn"]
            }
        },
        "required": ["query"]
    }
}

# Add to our tools list
tools.append({"type": "function", "function": doc_tool})

def handle_tool_call(message):
    """
    Handle tool calls from the OpenAI API response.
    """
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    result = None
    if function_name == "execute_code":
        result = execute_code(arguments.get("code_snippet", ""))
    elif function_name == "lookup_documentation":
        result = lookup_documentation(
            arguments.get("query", ""),
            arguments.get("library", None)
        )
    
    response = {
        "role": "tool",
        "content": json.dumps(result),
        "tool_call_id": tool_call.id
    }
    
    return response

# Function to determine if a model is from OpenAI or Claude
def is_claude_model(model_name):
    """Check if the model is a Claude model."""
    return model_name.startswith("claude")

# Chat function to handle both OpenAI and Claude models
def chat(message, history, model_choice):
    """
    Process chat messages with either OpenAI or Claude models.
    """
    messages = [{"role": "system", "content": system_message}]
    
    # Convert history to the format expected by the API
    for entry in history:
        messages.append({"role": "user", "content": entry[0]})
        if entry[1]:
            messages.append({"role": "assistant", "content": entry[1]})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    # Check if we're using a Claude model
    if is_claude_model(model_choice):
        # Format messages for Claude API
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Claude uses system prompt differently - we'll prepend it to the first user message
                system_content = msg["content"]
            elif msg["role"] == "user":
                claude_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                claude_messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "tool":
                # Convert tool responses to a format Claude can understand
                claude_messages.append({"role": "assistant", "content": f"Tool result: {msg['content']}"})
        
        # Adjust the first user message to include the system prompt if needed
        if claude_messages and claude_messages[0]["role"] == "user":
            claude_messages[0]["content"] = f"System instructions: {system_content}\n\nUser message: {claude_messages[0]['content']}"
        
        # Make the API call with the selected Claude model
        response = claude.messages.create(
            model=model_choice,
            messages=claude_messages,
            max_tokens=1024,
            stream=True
        )
        
        # For streaming responses with Claude
        collected_messages = []
        
        # Stream the response
        for chunk in response:
            if chunk.type == "content_block_delta" and chunk.delta.type == "text":
                collected_messages.append(chunk.delta.text)
            yield "".join(collected_messages)
        
        # If no chunks were collected, yield a default message
        if not collected_messages:
            yield "Sorry, I encountered an issue processing your request with Claude. Please try again."
    else:
        # Original OpenAI implementation
        response = openai.chat.completions.create(
            model=model_choice,
            messages=messages,
            tools=tools,
            stream=True
        )
        
        # For streaming responses
        collected_messages = []
        collected_chunks = []
        
        # Stream the response
        for chunk in response:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                chunk_message = chunk.choices[0]
                content = chunk_message.delta.content
                
                # If it's a tool call, handle it differently
                if chunk_message.finish_reason == "tool_calls" or (hasattr(chunk_message.delta, 'tool_calls') and chunk_message.delta.tool_calls):
                    # Need to collect all chunks to process the tool call properly
                    collected_chunks.append(chunk)
                    continue
                
                # Regular content streaming
                if content:
                    collected_messages.append(content)
                yield "".join(collected_messages)
        
        # If we collected tool call chunks, we need to process them
        if collected_chunks:
            # Reconstruct the full message from chunks
            full_message = openai.chat.completions.create(
                model=model_choice,
                messages=messages,
                tools=tools
            ).choices[0].message
            
            # Handle the tool call
            tool_response = handle_tool_call(full_message)
            
            # Add the tool call and its response to messages
            messages.append(full_message)
            messages.append(tool_response)
            
            # Get a new response that incorporates the tool result
            final_response = openai.chat.completions.create(
                model=model_choice,
                messages=messages
            )
            
            yield final_response.choices[0].message.content
        
        # If no chunks were collected but the response was not streamed, yield the collected messages
        if not collected_chunks and not collected_messages:
            yield "Sorry, I encountered an issue processing your request. Please try again."

def generate_voice_summary(full_response, max_chars=1200):
    """
    Generate a summarized, conversational version of the response for voice
    """
    # Create a system message that instructs the model to create a summary
    summary_system_message = """Create a brief, summary of the following technical explanation.
    Keep it under 100 words. Use a conversational tone but avoid unnecessary fillers.
    Include only the most essential information and one simple example if relevant.
    The summary must be complete and not end abruptly."""
    
    try:
        # Use Claude for the summary
        summary_response = claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=summary_system_message,
            max_tokens=100,  # Adjusted for Claude's token counting
            messages=[
                {"role": "user", "content": f"Please summarize this explanation: {full_response}"}
            ]
        )
        
        summary = summary_response.content[0].text
        
        # Print for debugging/inspection
        print(f"Claude summary ({len(summary)} chars):")
        print("-" * 50)
        print(summary)
        print("-" * 50)
        
        # Ensure we don't exceed max_chars
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."
            
        return summary
    except Exception as e:
        print(f"Error generating summary with Claude: {e}")
        
        # Fallback to OpenAI if Claude fails
        try:
            openai_summary_message = """Create a brief, friendly summary of the following technical explanation.
            Keep it under 100 words. Use a conversational, engaging tone.
            The summary must be complete and not end abruptly."""
            
            summary_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": openai_summary_message},
                    {"role": "user", "content": f"Please summarize this explanation: {full_response}"}
                ],
                max_tokens=100
            )
            
            summary = summary_response.choices[0].message.content
            
            # Print for debugging/inspection
            print(f"OpenAI fallback summary ({len(summary)} chars):")
            print("-" * 50)
            print(summary)
            print("-" * 50)
            
            return summary
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            
            # Last resort fallback
            return full_response[:max_chars] + ("..." if len(full_response) > max_chars else "")

def text_to_speech(text, filename="response_audio.mp3"):
    """
    Convert text to speech using OpenAI's TTS API and save to a file
    """
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        with open(filename, "wb") as file:
            file.write(response.content)
            
        return filename
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return None

def user_input(user_message, history, model_choice):
    """
    Process user input and add it to the conversation history.
    """
    return "", history + [{"role": "user", "content": user_message}]

def on_submit(history, model_choice, enable_voice):
    """
    Handler for when the user submits a message.
    """
    if not history:
        return history, None
        
    user_message = history[-1]["content"]
    bot_message = None
    audio_path = None
    
    # Format messages for API
    formatted_messages = [{"role": "system", "content": system_message}]
    for msg in history[:-1]:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
    formatted_messages.append({"role": "user", "content": user_message})
    
    # Check if we're using a Claude model
    if is_claude_model(model_choice):
        # Format messages for Claude API
        claude_messages = []
        system_content = ""
        
        for msg in formatted_messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                claude_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                claude_messages.append({"role": "assistant", "content": msg["content"]})
        
        # Adjust the first user message to include the system prompt
        if claude_messages and claude_messages[0]["role"] == "user":
            claude_messages[0]["content"] = f"System instructions: {system_content}\n\nUser message: {claude_messages[0]['content']}"
        
        # Make the API call with the selected Claude model
        response = claude.messages.create(
            model=model_choice,
            messages=claude_messages,
            max_tokens=1024
        )
        
        bot_message = response.content[0].text
    else:
        # Original OpenAI implementation
        response = openai.chat.completions.create(
            model=model_choice,
            messages=formatted_messages,
            tools=tools
        )
        
        # Process response
        openai_message = response.choices[0].message
        
        if response.choices[0].finish_reason == "tool_calls":
            # Handle tool calls
            tool_response = handle_tool_call(openai_message)
            
            formatted_messages.append({"role": "assistant", "content": openai_message.content, "tool_calls": openai_message.tool_calls})
            formatted_messages.append(tool_response)
            
            final_response = openai.chat.completions.create(
                model=model_choice,
                messages=formatted_messages
            )
            
            bot_message = final_response.choices[0].message.content
        else:
            bot_message = openai_message.content
    
    # Generate a voice summary if voice is enabled
    if enable_voice:
        try:
            # Create a friendlier, summarized version for voice
            voice_summary = generate_voice_summary(bot_message)
            
            # Make sure the last sentence is complete to avoid abrupt endings
            sentences = voice_summary.split('.')
            if len(sentences) > 1:
                # Ensure we have complete sentences by removing incomplete ones
                complete_summary = '.'.join(sentences[:-1]) + '.'
                if len(complete_summary) < 20:  # If too short, use original
                    complete_summary = voice_summary
            else:
                complete_summary = voice_summary
                
            print("Final audio text:", complete_summary)
            audio_path = text_to_speech(complete_summary)
            
            # Adding a note to the chat indicating a voice summary is available
            if audio_path:
                history.append({"role": "assistant", "content": bot_message})
                history.append({"role": "assistant", "content": "📢 *Voice summary available! Click play to listen.*"})
            return history, audio_path
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    history.append({"role": "assistant", "content": bot_message})
    return history, audio_path

def clear_chat():
    """
    Clear the chat history.
    """
    return [], None

def update_audio_visibility(enable_voice):
    """
    Update the visibility of the audio player based on the voice toggle.
    """
    return gr.update(visible=enable_voice)

def main():
    """
    Main function to create and launch the Gradio interface.
    """
    # Create the Gradio interface with all event handlers
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Data Science Mentor AI")
        gr.Markdown("Ask questions about Python, pandas, numpy, scikit-learn, or any data science topic.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Model selector
                model_dropdown = gr.Dropdown(
                    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-opus-20240229", "claude-3-7-sonnet-latest", "claude-3-5-haiku-20240307"],
                    label="Select AI Model",
                    value=DEFAULT_MODEL
                )
                
                # Main chat interface
                chatbot = gr.Chatbot(height=500, type="messages")
                msg = gr.Textbox(label="Ask your data science question:")
                
                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.Button("Clear")
                
                # Voice toggle and audio player
                with gr.Row():
                    voice_output = gr.Checkbox(label="Enable Voice Response", value=True)
                    audio_player = gr.Audio(label="Response Audio", visible=True)
                
                # Examples
                gr.Examples(
                    examples=[
                        "Explain the difference between .loc and .iloc in pandas",
                        "How do I handle missing values in a DataFrame?",
                        "Can you help me understand how Random Forest works?",
                        "What does sklearn.preprocessing.StandardScaler do?"
                    ],
                    inputs=msg
                )
        
        # Connect the event handlers
        msg.submit(user_input, [msg, chatbot, model_dropdown], [msg, chatbot]).then(
            on_submit, [chatbot, model_dropdown, voice_output], [chatbot, audio_player]
        )
        
        submit_btn.click(user_input, [msg, chatbot, model_dropdown], [msg, chatbot]).then(
            on_submit, [chatbot, model_dropdown, voice_output], [chatbot, audio_player]
        )
        
        voice_output.change(
            update_audio_visibility,
            [voice_output],
            [audio_player]
        )
        
        clear_btn.click(clear_chat, None, [chatbot, audio_player])
        
        # Launch the demo
        demo.launch(share=True)

if __name__ == "__main__":
    main()
