# import os
# import re
# import gradio as gr
# from dotenv import load_dotenv

# from src.rag_engine import RAGEngine
# from optimized_config import OPTIMIZED_CONFIG  # Optimized parameters

# # Load environment variables
# load_dotenv()

# # Initialize the RAG engine
# rag_engine = RAGEngine()  # Using enhanced defaults

# def md_to_html_bold(text: str) -> str:
#     """
#     Convert markdown-style **bold** to HTML <strong> tags.
#     """
#     return re.sub(r"\*\*(.+?)\*\*", r"<strong style='font-weight:700; color:#101828;'>\1</strong>", text)


# def style_citations(text: str) -> str:
#     """
#     Wrap any [citation] in a styled <span>.
#     """
#     return re.sub(r"\[([^\]]+)\]", r"<span style='color:#344054; font-weight:500;'>[\1]</span>", text)


# def format_answer_html(answer: str) -> str:
#     """
#     Format the raw answer into high-contrast HTML with lists, paragraphs,
#     bold text, and styled citations. Ensure full opacity and black text.
#     """
#     # Normalize lines
#     lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
#     numbered = any(re.match(rf"{i}\.\s+", ln) for i in range(1, 10) for ln in lines)

#     if numbered:
#         items = []
#         for ln in lines:
#             m = re.match(r"^(\d+)\.\s+(.*)", ln)
#             if m:
#                 items.append(m.group(2))
#             else:
#                 if items:
#                     items[-1] += " " + ln
#         html = "<ol style='margin:16px 0; padding-left:24px; color:#101828;'>\n"
#         for item in items:
#             html += f"  <li style='margin-bottom:12px; color:#101828; line-height:1.6;'>{item}</li>\n"
#         html += "</ol>"
#     else:
#         paras = [p.strip() for p in answer.split("\n\n") if p.strip()]
#         html = "\n".join(
#             f"<p style='margin:16px 0; color:#101828; line-height:1.8;'>{p}</p>"
#             for p in paras
#         )

#     # Apply bold and citation styling
#     html = md_to_html_bold(html)
#     html = style_citations(html)
#     return html


# def process_query(query: str) -> str:
#     """
#     Process a user query and return styled HTML answer with sources, with full visibility.
#     """
#     result = rag_engine.answer_question(query)
#     answer = result.get("answer", "")

#     # Generate the formatted answer HTML
#     formatted_answer = format_answer_html(answer)

#     # Build the sources panel with darker text
#     sources_html = (
#         "<div style=\"margin-top:24px; padding:20px; background:#f0f1f3; "
#         "border-radius:12px; border:1px solid #d0d5dd;\">"
#         "<h3 style=\"margin:0 0 16px; color:#101828; font-size:18px; font-weight:600;\">📚 Source Attribution</h3>"
#         "<div style=\"space-y:12px;\">"
#     )
    
#     for i, src in enumerate(result.get("sources", [])):
#         if src.get("similarity", 0) < 0.05:
#             continue
#         title = src.get("title", "Untitled")
#         url = src.get("url")
#         section = src.get("section")
        
#         # Icon based on source type
#         icon = "📄"
#         if "handbook" in title.lower():
#             icon = "📘"
#         elif "report" in title.lower():
#             icon = "📊"
#         elif "knowledge" in title.lower():
#             icon = "🧠"

#         link_html = (
#             f"<a href='{url}' target='_blank' style='color:#0066cc; text-decoration:none; font-weight:600;'>{title}</a>"
#             if url else f"<span style='font-weight:600; color:#101828;'>{title}</span>"
#         )
#         section_html = f"<div style='color:#344054; font-size:14px; margin-top:4px;'>{section}</div>" if section else ""
        
#         sources_html += (
#             f"<div style='padding:12px; background:white; border-radius:8px; margin-bottom:8px; "
#             f"border:1px solid #d0d5dd;'>"
#             f"<div style='display:flex; align-items:start;'>"
#             f"<span style='margin-right:8px; font-size:16px;'>{icon}</span>"
#             f"<div style='flex:1; color:#101828;'>{link_html}{section_html}</div>"
#             f"</div></div>"
#         )

#     sources_html += "</div></div>"

#     # Wrap with a clean container with stronger text
#     final_html = (
#         "<div style=\"background:white; padding:24px; border-radius:12px; "
#         "box-shadow:0 1px 3px rgba(0,0,0,0.1);\">"
#         f"<div style='line-height:1.8; font-size:16px; color:#101828; font-weight:400;'>{formatted_answer}</div>"
#         + sources_html + "</div>"
#     )

#     return final_html


# # Simplified and focused CSS
# custom_css = """
# /* Base styling for the app */
# .gradio-container {
#     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
#     background-color: #f8f9fa;
# }

# /* Main container */
# .main-container {
#     max-width: 900px;
#     margin: 0 auto;
#     padding: 40px 20px;
# }

# /* Header styling */
# .header-section {
#     text-align: center;
#     margin-bottom: 48px;
# }

# .header-title {
#     font-size: 32px;
#     font-weight: 700;
#     color: #101828;
#     margin-bottom: 8px;
# }

# .header-subtitle {
#     font-size: 18px;
#     color: #344054;
#     font-weight: 400;
# }

# /* Input section */
# .input-section {
#     background: white;
#     border-radius: 16px;
#     padding: 32px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.08);
#     margin-bottom: 32px;
# }

# /* Critical fix for text input */
# .gradio-container textarea,
# .gradio-container input[type="text"] {
#     background-color: white !important;
#     color: #101828 !important;
#     border: 2px solid #e4e7ec !important;
#     border-radius: 8px !important;
#     padding: 12px 16px !important;
#     font-size: 16px !important;
#     font-family: inherit !important;
#     width: 100% !important;
#     box-sizing: border-box !important;
# }

# .gradio-container textarea:focus,
# .gradio-container input[type="text"]:focus {
#     border-color: #0066cc !important;
#     outline: none !important;
#     box-shadow: 0 0 0 3px rgba(0,102,204,0.1) !important;
#     background-color: white !important;
#     color: #101828 !important;
# }

# .gradio-container textarea::placeholder,
# .gradio-container input[type="text"]::placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# /* Remove any conflicting styles on text inputs */
# .gradio-container .gr-text-input input,
# .gradio-container .gr-text-input textarea {
#     background: white !important;
#     color: #101828 !important;
#     -webkit-text-fill-color: #101828 !important;
# }

# /* Button styling */
# .gradio-container .gr-button-primary {
#     background-color: #0066cc !important;
#     border: none !important;
#     color: white !important;
#     font-weight: 500 !important;
#     padding: 10px 24px !important;
#     font-size: 16px !important;
#     border-radius: 8px !important;
#     cursor: pointer !important;
# }

# .gradio-container .gr-button-primary:hover {
#     background-color: #0052a3 !important;
# }

# /* Example questions */
# .example-header {
#     font-size: 20px;
#     font-weight: 600;
#     color: #101828;
#     margin-bottom: 16px;
# }

# .example-grid {
#     display: grid;
#     grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
#     gap: 16px;
# }

# .example-card {
#     background: white;
#     border: 1px solid #e4e7ec;
#     border-radius: 12px;
#     padding: 20px;
#     cursor: pointer;
#     transition: all 0.2s ease;
#     display: flex;
#     align-items: center;
#     gap: 12px;
# }

# .example-card:hover {
#     border-color: #0066cc;
#     box-shadow: 0 4px 12px rgba(0,102,204,0.15);
#     transform: translateY(-2px);
# }

# .example-icon {
#     font-size: 24px;
#     width: 40px;
#     height: 40px;
#     background: #eff8ff;
#     border-radius: 8px;
#     display: flex;
#     align-items: center;
#     justify-content: center;
# }

# .example-text {
#     flex: 1;
#     color: #101828;
#     font-size: 15px;
#     font-weight: 500;
# }

# /* Output section */
# .output-section {
#     background: white;
#     border-radius: 16px;
#     padding: 32px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.08);
#     min-height: 200px;
# }

# .empty-state {
#     text-align: center;
#     color: #667085;
#     padding: 60px 20px;
#     font-size: 16px;
# }

# /* Remove Gradio's default styling that might interfere */
# .gradio-container .gr-form,
# .gradio-container .gr-box,
# .gradio-container .gr-panel {
#     border: none !important;
#     background: transparent !important;
#     padding: 0 !important;
# }

# /* Ensure all text is visible */
# .gradio-container label {
#     display: none;
# }

# /* Override any dark mode styles */
# .gradio-container * {
#     opacity: 1;
# }
# """

# # Example questions
# example_questions = [
#     {"icon": "📋", "text": "What is Technova's mission statement?"},
#     {"icon": "📊", "text": "What were last quarter's key metrics?"},
#     {"icon": "👥", "text": "what is the policy on Employee Dating & Relationships "},
#     {"icon": "📦", "text": "What are our core products?"},
#     {"icon": "🏠", "text": "What is our remote work policy?"},
#     {"icon": "💼", "text": "How does our code review process work?"},
# ]

# # Build the interface using Blocks
# with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
#     with gr.Column(elem_classes="main-container"):
#         # Header
#         gr.HTML("""
#             <div class="header-section">
#                 <h1 class="header-title">🏢 TechNova Knowledge Assistant</h1>
#                 <p class="header-subtitle">Get instant answers from our company knowledge base</p>
#             </div>
#         """)
        
#         # Input Section
#         with gr.Column(elem_classes="input-section"):
#             query_input = gr.Textbox(
#                 placeholder="Type your question here...",
#                 show_label=False,
#                 lines=1,
#                 max_lines=3,
#                 elem_id="query-input"
#             )
#             submit_btn = gr.Button("Ask Question →", variant="primary", scale=0)
        
#         # Example Questions
#         gr.HTML('<h2 class="example-header">Example Questions</h2>')
        
#         # Create example buttons
#         example_html = '<div class="example-grid">'
#         for ex in example_questions:
#             example_html += f'''
#                 <div class="example-card" onclick="
#                     const input = document.getElementById('query-input').querySelector('textarea') || document.getElementById('query-input').querySelector('input');
#                     if(input) {{
#                         input.value = '{ex["text"]}';
#                         input.dispatchEvent(new Event('input', {{bubbles: true}}));
#                     }}
#                 ">
#                     <div class="example-icon">{ex["icon"]}</div>
#                     <div class="example-text">{ex["text"]}</div>
#                 </div>
#             '''
#         example_html += '</div>'
#         gr.HTML(example_html)
        
#         # Output Section
#         with gr.Column(elem_classes="output-section"):
#             output = gr.HTML(
#                 value='<div class="empty-state">💬 Ask a question to get started</div>',
#                 show_label=False
#             )
        
#         # Event handlers
#         submit_btn.click(
#             fn=process_query,
#             inputs=[query_input],
#             outputs=[output]
#         )
        
#         query_input.submit(
#             fn=process_query,
#             inputs=[query_input],
#             outputs=[output]
#         )

# if __name__ == "__main__":
#     demo.launch(share=True)  # Set share=False in production



import os
import re
import gradio as gr
from dotenv import load_dotenv

from src.rag_engine import RAGEngine
from optimized_config import OPTIMIZED_CONFIG  # Optimized parameters

# Load environment variables
load_dotenv()

# Initialize the RAG engine
rag_engine = RAGEngine()  # Using enhanced defaults

def md_to_html_bold(text: str) -> str:
    """
    Convert markdown-style **bold** to HTML <strong> tags.
    """
    return re.sub(r"\*\*(.+?)\*\*", r"<strong style='font-weight:700; color:#101828;'>\1</strong>", text)


def style_citations(text: str) -> str:
    """
    Wrap any [citation] in a styled <span>.
    """
    return re.sub(r"\[([^\]]+)\]", r"<span style='color:#344054; font-weight:500;'>[\1]</span>", text)


def format_answer_html(answer: str) -> str:
    """
    Format the raw answer into high-contrast HTML with lists, paragraphs,
    bold text, and styled citations. Ensure full opacity and black text.
    """
    # Normalize lines
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    numbered = any(re.match(rf"{i}\.\s+", ln) for i in range(1, 10) for ln in lines)

    if numbered:
        items = []
        for ln in lines:
            m = re.match(r"^(\d+)\.\s+(.*)", ln)
            if m:
                items.append(m.group(2))
            else:
                if items:
                    items[-1] += " " + ln
        html = "<ol style='margin:16px 0; padding-left:24px; color:#101828;'>\n"
        for item in items:
            html += f"  <li style='margin-bottom:12px; color:#101828; line-height:1.6;'>{item}</li>\n"
        html += "</ol>"
    else:
        paras = [p.strip() for p in answer.split("\n\n") if p.strip()]
        html = "\n".join(
            f"<p style='margin:16px 0; color:#101828; line-height:1.8;'>{p}</p>"
            for p in paras
        )

    # Apply bold and citation styling
    html = md_to_html_bold(html)
    html = style_citations(html)
    return html


def process_query(query: str) -> str:
    """
    Process a user query and return styled HTML answer with sources, with full visibility.
    """
    result = rag_engine.answer_question(query)
    answer = result.get("answer", "")

    # Generate the formatted answer HTML
    formatted_answer = format_answer_html(answer)

    # Build the sources panel with darker text
    sources_html = (
        "<div style=\"margin-top:24px; padding:20px; background:#f0f1f3; "
        "border-radius:12px; border:1px solid #d0d5dd;\">"
        "<h3 style=\"margin:0 0 16px; color:#101828; font-size:18px; font-weight:600;\">📚 Source Attribution</h3>"
        "<div style=\"space-y:12px;\">"
    )
    
    for i, src in enumerate(result.get("sources", [])):
        if src.get("similarity", 0) < 0.05:
            continue
        title = src.get("title", "Untitled")
        url = src.get("url")
        section = src.get("section")
        
        # Icon based on source type
        icon = "📄"
        if "handbook" in title.lower():
            icon = "📘"
        elif "report" in title.lower():
            icon = "📊"
        elif "knowledge" in title.lower():
            icon = "🧠"

        link_html = (
            f"<a href='{url}' target='_blank' style='color:#0066cc; text-decoration:none; font-weight:600;'>{title}</a>"
            if url else f"<span style='font-weight:600; color:#101828;'>{title}</span>"
        )
        section_html = f"<div style='color:#344054; font-size:14px; margin-top:4px;'>{section}</div>" if section else ""
        
        sources_html += (
            f"<div style='padding:12px; background:white; border-radius:8px; margin-bottom:8px; "
            f"border:1px solid #d0d5dd;'>"
            f"<div style='display:flex; align-items:start;'>"
            f"<span style='margin-right:8px; font-size:16px;'>{icon}</span>"
            f"<div style='flex:1; color:#101828;'>{link_html}{section_html}</div>"
            f"</div></div>"
        )

    sources_html += "</div></div>"

    # Wrap with a clean container with stronger text
    final_html = (
        "<div style=\"background:white; padding:24px; border-radius:12px; "
        "box-shadow:0 1px 3px rgba(0,0,0,0.1);\">"
        f"<div style='line-height:1.8; font-size:16px; color:#101828; font-weight:400;'>{formatted_answer}</div>"
        + sources_html + "</div>"
    )

    return final_html


# Fixed CSS with proper text visibility
custom_css = """
/* Force all text to be visible */
.gradio-container * {
    opacity: 1 !important;
}

/* Base styling for the app */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: #f8f9fa;
}

/* Force all text elements to be dark and visible */
.gradio-container h1,
.gradio-container h2, 
.gradio-container h3,
.gradio-container p,
.gradio-container span,
.gradio-container div {
    color: #101828 !important;
    opacity: 1 !important;
}

/* Main container */
.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px;
}

/* Header styling */
.header-section {
    text-align: center;
    margin-bottom: 48px;
}

.header-title {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #101828 !important;
    margin-bottom: 8px !important;
    opacity: 1 !important;
}

.header-subtitle {
    font-size: 18px !important;
    color: #344054 !important;
    font-weight: 400 !important;
    opacity: 1 !important;
}

/* Input section */
.input-section {
    background: white;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 32px;
}

/* Critical fix for text input */
.gradio-container textarea,
.gradio-container input[type="text"] {
    background-color: white !important;
    color: #101828 !important;
    border: 2px solid #e4e7ec !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 16px !important;
    font-family: inherit !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: #0066cc !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,102,204,0.1) !important;
    background-color: white !important;
    color: #101828 !important;
}

.gradio-container textarea::placeholder,
.gradio-container input[type="text"]::placeholder {
    color: #667085 !important;
    opacity: 1 !important;
}

/* Remove any conflicting styles on text inputs */
.gradio-container .gr-text-input input,
.gradio-container .gr-text-input textarea {
    background: white !important;
    color: #101828 !important;
    -webkit-text-fill-color: #101828 !important;
}

/* Button styling */
.gradio-container .gr-button-primary {
    background-color: #0066cc !important;
    border: none !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
}

.gradio-container .gr-button-primary:hover {
    background-color: #0052a3 !important;
}

/* Example questions */
.example-header {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #101828 !important;
    margin-bottom: 16px !important;
    opacity: 1 !important;
}

.example-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
}

.example-card {
    background: white !important;
    border: 1px solid #e4e7ec !important;
    border-radius: 12px !important;
    padding: 20px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    opacity: 1 !important;
}

.example-card:hover {
    border-color: #0066cc !important;
    box-shadow: 0 4px 12px rgba(0,102,204,0.15) !important;
    transform: translateY(-2px) !important;
}

.example-icon {
    font-size: 24px !important;
    width: 40px !important;
    height: 40px !important;
    background: #eff8ff !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    opacity: 1 !important;
}

.example-text {
    flex: 1 !important;
    color: #101828 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

/* Output section */
.output-section {
    background: white;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    min-height: 200px;
}

.output-section * {
    color: #101828 !important;
    opacity: 1 !important;
}

.empty-state {
    text-align: center !important;
    color: #667085 !important;
    padding: 60px 20px !important;
    font-size: 16px !important;
    opacity: 1 !important;
}

/* Remove Gradio's default styling that might interfere */
.gradio-container .gr-form,
.gradio-container .gr-box,
.gradio-container .gr-panel {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Ensure all text is visible */
.gradio-container label {
    display: none;
}

/* Override any dark mode styles */
.gradio-container * {
    opacity: 1;
}
"""

# Example questions
example_questions = [
    {"icon": "📋", "text": "What is Technova's mission statement?"},
    {"icon": "📊", "text": "What were last quarter's key metrics?"},
    {"icon": "👥", "text": "what is the policy on Employee Dating & Relationships "},
    {"icon": "📦", "text": "What are our core products?"},
    {"icon": "🏠", "text": "What is our remote work policy?"},
    {"icon": "💼", "text": "How does our code review process work?"},
]

# Build the interface using Blocks
with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    with gr.Column(elem_classes="main-container"):
        # Header
        gr.HTML("""
            <div class="header-section">
                <h1 class="header-title" style="color: #101828 !important; opacity: 1 !important; font-size: 32px !important; font-weight: 700 !important; margin-bottom: 8px !important;">🏢 TechNova Knowledge Assistant</h1>
                <p class="header-subtitle" style="color: #344054 !important; opacity: 1 !important; font-size: 18px !important; font-weight: 400 !important;">Get instant answers from our company knowledge base</p>
            </div>
        """)
        
        # Input Section
        with gr.Column(elem_classes="input-section"):
            query_input = gr.Textbox(
                placeholder="Type your question here...",
                show_label=False,
                lines=1,
                max_lines=3,
                elem_id="query-input"
            )
            submit_btn = gr.Button("Ask Question →", variant="primary", scale=0)
        
        # Example Questions
        gr.HTML('<h2 class="example-header" style="color: #101828 !important; opacity: 1 !important; font-size: 20px !important; font-weight: 600 !important; margin-bottom: 16px !important;">Example Questions</h2>')
        
        # Create example buttons
        example_html = '<div class="example-grid">'
        for ex in example_questions:
            example_html += f'''
                <div class="example-card" style="background: white !important; border: 1px solid #e4e7ec !important; border-radius: 12px !important; padding: 20px !important; cursor: pointer !important; display: flex !important; align-items: center !important; gap: 12px !important; opacity: 1 !important;" onclick="
                    const input = document.getElementById('query-input').querySelector('textarea') || document.getElementById('query-input').querySelector('input');
                    if(input) {{
                        input.value = '{ex["text"]}';
                        input.dispatchEvent(new Event('input', {{bubbles: true}}));
                    }}
                ">
                    <div class="example-icon" style="font-size: 24px !important; width: 40px !important; height: 40px !important; background: #eff8ff !important; border-radius: 8px !important; display: flex !important; align-items: center !important; justify-content: center !important; opacity: 1 !important;">{ex["icon"]}</div>
                    <div class="example-text" style="flex: 1 !important; color: #101828 !important; font-size: 15px !important; font-weight: 500 !important; opacity: 1 !important;">{ex["text"]}</div>
                </div>
            '''
        example_html += '</div>'
        gr.HTML(example_html)
        
        # Output Section
        with gr.Column(elem_classes="output-section"):
            output = gr.HTML(
                value='<div class="empty-state" style="text-align: center !important; color: #667085 !important; padding: 60px 20px !important; font-size: 16px !important; opacity: 1 !important;">💬 Ask a question to get started</div>',
                show_label=False
            )
        
        # Event handlers
        submit_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[output]
        )
        
        query_input.submit(
            fn=process_query,
            inputs=[query_input],
            outputs=[output]
        )

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False in production