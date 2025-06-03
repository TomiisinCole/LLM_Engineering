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
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def style_citations(text: str) -> str:
    """
    Wrap any [citation] in a styled <span>.
    """
    return re.sub(r"\[([^\]]+)\]", r"<span style='color:#555 !important;'>[\1]</span>", text)


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
        html = "<ol style='margin:12px 0; color:#000 !important; opacity:1 !important;'>\n"
        for item in items:
            html += f"  <li style='margin-bottom:8px; color:#000 !important; opacity:1 !important;'>{item}</li>\n"
        html += "</ol>"
    else:
        paras = [p.strip() for p in answer.split("\n\n") if p.strip()]
        html = "\n".join(
            f"<p style='margin:12px 0; color:#000 !important; opacity:1 !important;'>{p}</p>"
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
    # Add custom CSS to ensure visibility
    custom_css = """
    <style>
    .answer-content * {
        color: #000 !important;
        opacity: 1 !important;
        font-weight: normal !important;
    }
    .answer-content strong {
        font-weight: bold !important;
    }
    </style>
    """
    
    result = rag_engine.answer_question(query)
    answer = result.get("answer", "")

    # Generate the formatted answer HTML
    formatted_answer = format_answer_html(answer)

    # Build the sources panel
    sources_html = (
        "<div style=\"margin-top:20px; padding:15px; background:#f0f4f8; "
        "border-left:4px solid #3498db; border-radius:8px; opacity:1 !important;\">"
        "<h3 style=\"margin:0 0 12px; color:#2c3e50; opacity:1 !important;\">Sources</h3>"
        "<ol style=\"padding-left:20px; margin:0;\">"
    )
    for src in result.get("sources", []):
        if src.get("similarity", 0) < 0.05:
            continue
        title = src.get("title", "Untitled")
        url = src.get("url")
        section = src.get("section")

        link_html = (
            f"<a href='{url}' target='_blank' style='color:#3498db !important; text-decoration:none; opacity:1 !important;'>{title}</a>"
            if url else f"{title}"
        )
        section_html = f" - <span style='color:#7f8c8d !important; opacity:1 !important;'>{section}</span>" if section else ""
        sources_html += (
            f"<li style='margin-bottom:8px; opacity:1 !important;'><strong>{link_html}</strong>{section_html}</li>"
        )

    sources_html += "</ol></div>"

    # Wrap with a class that we can target with CSS plus inline styles
    final_html = custom_css + (
        "<div class='answer-content' style=\"background:#fff; padding:20px; border-radius:8px;"
        " box-shadow:0 2px 5px rgba(0,0,0,0.1);\">"
        f"<div style='line-height:1.6; font-size:16px; color:#000 !important; opacity:1 !important;'>{formatted_answer}</div>"
        + sources_html + "</div>"
    )

    

    return final_html


# Gradio interface setup
demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(
        placeholder="Ask a question about TechNova...",
        label="Your Question"
    ),
    outputs=gr.HTML(label="Answer"),
    title="TechNova Knowledge Navigator",
    description="Ask questions about TechNova and get answers based on our Notion workspace.",
    examples=[
        ["What is our remote work policy?"],
        ["What are TechNova's core values?"],
        ["How does our code review process work?"],
        ["what is the policy on Employee Dating & Relationships"],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False in production
# # What are TechNova's core values? please explain each to me



## option 2


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


# # Custom CSS for professional look
# custom_css = """
# /* CSS Reset - Force all elements to be fully visible */
# *, *::before, *::after {
#     opacity: 1 !important;
#     -webkit-opacity: 1 !important;
#     -moz-opacity: 1 !important;
#     -o-opacity: 1 !important;
# }

# /* Force text color on common elements */
# p, h1, h2, h3, h4, h5, h6, span, div, li, a, input, textarea, button {
#     opacity: 1 !important;
# }

# /* Force Gradio specific elements */
# .gr-box, .gr-form, .gr-input-label, .gr-panel, .gr-text-input, .gr-button {
#     opacity: 1 !important;
# }

# .gradio-container {
#     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
#     background-color: #f8f9fa !important;
# }

# .gradio-container * {
#     opacity: 1 !important;
# }

# .main-container {
#     max-width: 900px !important;
#     margin: 0 auto !important;
#     padding: 40px 20px !important;
# }

# .header-section {
#     text-align: center;
#     margin-bottom: 48px;
# }

# .header-title {
#     font-size: 32px !important;
#     font-weight: 700 !important;
#     color: #101828 !important;
#     margin-bottom: 8px !important;
#     opacity: 1 !important;
# }

# .header-subtitle {
#     font-size: 18px !important;
#     color: #344054 !important;
#     font-weight: 400 !important;
#     opacity: 1 !important;
# }

# .input-section {
#     background: white;
#     border-radius: 16px;
#     padding: 32px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.08);
#     margin-bottom: 32px;
# }

# .example-section {
#     margin-bottom: 32px;
# }

# .example-header {
#     font-size: 20px !important;
#     font-weight: 600 !important;
#     color: #101828 !important;
#     margin-bottom: 16px !important;
#     opacity: 1 !important;
# }

# .example-grid {
#     display: grid;
#     grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
#     gap: 16px;
# }

# .example-card {
#     background: white !important;
#     border: 1px solid #e4e7ec !important;
#     border-radius: 12px !important;
#     padding: 20px !important;
#     cursor: pointer !important;
#     transition: all 0.2s ease !important;
#     display: flex !important;
#     align-items: center !important;
#     gap: 12px !important;
#     opacity: 1 !important;
# }

# .example-card:hover {
#     border-color: #0066cc !important;
#     box-shadow: 0 4px 12px rgba(0,102,204,0.15) !important;
#     transform: translateY(-2px) !important;
# }

# .example-icon {
#     font-size: 24px !important;
#     width: 40px !important;
#     height: 40px !important;
#     background: #eff8ff !important;
#     border-radius: 8px !important;
#     display: flex !important;
#     align-items: center !important;
#     justify-content: center !important;
#     opacity: 1 !important;
# }

# .example-text {
#     flex: 1 !important;
#     color: #101828 !important;
#     font-size: 15px !important;
#     font-weight: 500 !important;
#     opacity: 1 !important;
# }

# .output-section {
#     background: white;
#     border-radius: 16px;
#     padding: 32px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.08);
#     min-height: 200px;
# }

# .output-section * {
#     color: #101828 !important;
#     opacity: 1 !important;
# }

# .empty-state {
#     text-align: center !important;
#     color: #667085 !important;
#     padding: 60px 20px !important;
#     font-size: 16px !important;
#     opacity: 1 !important;
# }

# /* Gradio component overrides */
# .dark-button {
#     opacity: 1 !important;
# }

# .gr-button-primary {
#     background-color: #0066cc !important;
#     border: none !important;
#     font-weight: 500 !important;
#     padding: 10px 24px !important;
#     font-size: 16px !important;
#     border-radius: 8px !important;
#     transition: all 0.2s ease !important;
#     opacity: 1 !important;
#     color: white !important;
# }

# .gr-button-primary:hover {
#     background-color: #0052a3 !important;
#     transform: translateY(-1px) !important;
#     box-shadow: 0 4px 12px rgba(0,102,204,0.25) !important;
# }

# .gr-form {
#     border: none !important;
#     background: transparent !important;
#     padding: 0 !important;
# }

# .gr-box {
#     border: none !important;
#     background: transparent !important;
# }

# .gr-padded {
#     padding: 0 !important;
# }

# .gr-panel {
#     border: none !important;
#     background: transparent !important;
# }

# /* Input field styling */
# .dark-input textarea {
#     color: #101828 !important;
#     background-color: white !important;
#     opacity: 1 !important;
# }

# textarea, input[type="text"] {
#     border: 2px solid #e4e7ec !important;
#     border-radius: 8px !important;
#     padding: 12px 16px !important;
#     font-size: 16px !important;
#     transition: all 0.2s ease !important;
#     color: #101828 !important;
#     background-color: white !important;
#     opacity: 1 !important;
# }

# textarea:focus, input[type="text"]:focus {
#     border-color: #0066cc !important;
#     outline: none !important;
#     box-shadow: 0 0 0 3px rgba(0,102,204,0.1) !important;
#     color: #101828 !important;
# }

# textarea::placeholder, input[type="text"]::placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# label {
#     display: none !important;
# }

# /* Force all output text to be visible */
# .output-section p,
# .output-section li,
# .output-section span,
# .output-section div,
# .output-section strong,
# .output-section a {
#     opacity: 1 !important;
#     color: #101828 !important;
# }

# .output-section a {
#     color: #0066cc !important;
# }

# /* Override any Gradio default styles that might cause fading */
# .gr-prose * {
#     opacity: 1 !important;
# }

# .dark .gr-prose * {
#     opacity: 1 !important;
# }

# /* Ensure all text elements are visible */
# h1, h2, h3, h4, h5, h6, p, span, div, li, a {
#     opacity: 1 !important;
# }

# /* Force dark text color on all text elements */
# .gradio-container h1,
# .gradio-container h2,
# .gradio-container h3,
# .gradio-container p,
# .gradio-container span,
# .gradio-container div {
#     color: #101828 !important;
#     opacity: 1 !important;
# }

# /* Override any theme-specific opacity */
# .light *, .dark * {
#     opacity: 1 !important;
# }

# /* Soft theme specific overrides */
# .soft {
#     --body-text-color: #101828 !important;
#     --color-accent-soft: #101828 !important;
#     --body-text-color-subdued: #344054 !important;
# }

# .soft * {
#     opacity: 1 !important;
# }

# /* Final overrides to ensure visibility */
# .gradio-container * {
#     opacity: 1 !important;
# }

# .gr-text-input {
#     opacity: 1 !important;
# }

# .gr-input {
#     opacity: 1 !important;
# }

# .gr-button {
#     opacity: 1 !important;
# }

# /* Ensure example cards are fully visible */
# .example-card * {
#     opacity: 1 !important;
# }

# /* Force all text to be dark */
# body, .gradio-container {
#     color: #101828 !important;
# }

# /* Anti-gray-out measures */
# .gradio-container :not(button) {
#     color: #101828 !important;
# }

# .gr-prose {
#     color: #101828 !important;
# }

# .gr-padded {
#     color: #101828 !important;
# }

# /* Ensure no disabled styles are applied */
# :disabled {
#     opacity: 1 !important;
#     color: #101828 !important;
# }

# /* Remove any backdrop filters */
# * {
#     backdrop-filter: none !important;
#     -webkit-backdrop-filter: none !important;
# }

# /* Handle Gradio pending/processing states */
# .pending {
#     opacity: 1 !important;
# }

# .processing {
#     opacity: 1 !important;
# }

# /* Ensure all states are visible */
# *[style*="opacity"] {
#     opacity: 1 !important;
# }

# /* Override inline styles */
# [style*="color: rgb"] {
#     color: #101828 !important;
# }

# /* Final nuclear option - use CSS variables */
# :root {
#     --tw-text-opacity: 1 !important;
#     --text-opacity: 1 !important;
# }

# /* Target Gradio's label and text elements */
# .gr-input-label, .gr-button, .gr-check-radio, .gr-checkbox-label {
#     opacity: 1 !important;
#     color: #101828 !important;
# }

# /* Ensure placeholder text is visible */
# ::placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# ::-webkit-input-placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# ::-moz-placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# :-ms-input-placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# /* Override dark mode if active */
# .dark {
#     filter: none !important;
# }

# .dark * {
#     opacity: 1 !important;
#     filter: none !important;
# }

# /* Remove any theme-based opacity */
# [class*="opacity-"] {
#     opacity: 1 !important;
# }

# /* Ensure Gradio containers are visible */
# .container {
#     opacity: 1 !important;
# }

# .gr-panel {
#     opacity: 1 !important;
# }

# /* Ultimate override - target all elements with any opacity */
# *:not(img):not(svg) {
#     opacity: 1 !important;
#     min-opacity: 1 !important;
#     max-opacity: 1 !important;
# }

# /* Force text color on text elements only */
# h1, h2, h3, h4, h5, h6, p, label, input, textarea {
#     color: #101828 !important;
#     -webkit-text-fill-color: #101828 !important;
# }

# /* Specific overrides for headers and example text */
# .header-title, .header-subtitle, .example-header, .example-text {
#     color: #101828 !important;
#     -webkit-text-fill-color: #101828 !important;
#     opacity: 1 !important;
# }

# /* Ensure spans and divs with text are visible */
# span, div {
#     opacity: 1 !important;
# }

# /* But keep their color inheritance unless specified */
# .main-container span, .main-container div {
#     color: inherit;
# }

# /* Remove any gray text colors */
# [style*="color: gray"], [style*="color: #999"], [style*="color: #666"], [style*="color: #ccc"] {
#     color: #101828 !important;
# }

# /* Final insurance - no element should have low opacity or gray color */
# * {
#     min-opacity: 0.99;
# }

# /* Specific fix for Gradio text inputs */
# .gr-text-input input, .gr-text-input textarea {
#     color: #101828 !important;
#     -webkit-text-fill-color: #101828 !important;
#     opacity: 1 !important;
# }
# """

# # Example questions
# example_questions = [
#     {"icon": "📋", "text": "What is Technova's mission statement?"},
#     {"icon": "📊", "text": "What were last quarter's key metrics?"}, #Brand Voice & Tone
#     {"icon": "👥", "text": "what is the policy on Employee Dating & Relationships "},
#     {"icon": "📦", "text": "What are our core products?"},
#     {"icon": "🏠", "text": "What is our remote work policy?"},
#     {"icon": "💼", "text": "How does our code review process work?"},
# ]

# # Build the interface using Blocks
# with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
#     # Add inline styles to force visibility
#     gr.HTML("""
#         <style>
#         /* Global reset to force visibility */
#         * {
#             opacity: 1 !important;
#             -webkit-opacity: 1 !important;
#         }
        
#         /* Force all Gradio elements to be visible */
#         .gradio-container, .gradio-container * {
#             opacity: 1 !important;
#         }
        
#         /* Force specific text elements */
#         .main-container *, 
#         .header-title, 
#         .header-subtitle,
#         .example-header,
#         .example-text,
#         .example-card,
#         h1, h2, h3, p, span, div, input, textarea, button {
#             opacity: 1 !important;
#             color: inherit !important;
#         }
        
#         /* Ensure dark text where needed */
#         .header-title { color: #101828 !important; }
#         .header-subtitle { color: #344054 !important; }
#         .example-header { color: #101828 !important; }
#         .example-text { color: #101828 !important; }
        
#         /* Force input text to be dark */
#         textarea, input {
#             color: #101828 !important;
#             opacity: 1 !important;
#             -webkit-text-fill-color: #101828 !important;
#         }
        
#         /* Remove any filters that might cause graying */
#         * {
#             filter: none !important;
#             -webkit-filter: none !important;
#         }
#         </style>
#         <script>
#         // Comprehensive visibility enforcement
#         (function() {
#             // Function to force visibility on all elements
#             const forceVisibility = function() {
#                 // Get all elements
#                 const allElements = document.querySelectorAll('*');
#                 allElements.forEach(element => {
#                     // Force opacity
#                     element.style.setProperty('opacity', '1', 'important');
#                     element.style.setProperty('filter', 'none', 'important');
                    
#                     // Check computed style and override if needed
#                     const computed = window.getComputedStyle(element);
#                     if (computed.opacity !== '1') {
#                         element.style.cssText += 'opacity: 1 !important;';
#                     }
#                 });
                
#                 // Specific fixes for text elements
#                 const textElements = document.querySelectorAll('h1, h2, h3, p, span, div, input, textarea, label');
#                 textElements.forEach(element => {
#                     if (!element.closest('button')) {  // Don't change button text
#                         element.style.setProperty('color', '#101828', 'important');
#                         element.style.setProperty('-webkit-text-fill-color', '#101828', 'important');
#                     }
#                 });
                
#                 // Fix placeholders
#                 const inputs = document.querySelectorAll('input, textarea');
#                 inputs.forEach(input => {
#                     input.style.setProperty('color', '#101828', 'important');
#                     input.style.setProperty('-webkit-text-fill-color', '#101828', 'important');
#                 });
#             };
            
#             // Run on various events
#             window.addEventListener('DOMContentLoaded', forceVisibility);
#             window.addEventListener('load', forceVisibility);
            
#             // Run periodically
#             setTimeout(forceVisibility, 100);
#             setTimeout(forceVisibility, 500);
#             setTimeout(forceVisibility, 1000);
#             setTimeout(forceVisibility, 2000);
            
#             // Monitor for DOM changes
#             const observer = new MutationObserver(function(mutations) {
#                 forceVisibility();
#             });
            
#             // Start observing when ready
#             window.addEventListener('DOMContentLoaded', function() {
#                 observer.observe(document.body, {
#                     childList: true,
#                     subtree: true,
#                     attributes: true,
#                     attributeFilter: ['style', 'class']
#                 });
#             });
#         })();
#         </script>
#     """)
    
#     with gr.Column(elem_classes="main-container"):
#         # Header
#         gr.HTML("""
#             <div class="header-section">
#                 <h1 class="header-title" style="color: #101828 !important; opacity: 1 !important; -webkit-text-fill-color: #101828 !important;">🏢 TechNova Knowledge Assistant</h1>
#                 <p class="header-subtitle" style="color: #344054 !important; opacity: 1 !important; -webkit-text-fill-color: #344054 !important;">Get instant answers from our company knowledge base</p>
#             </div>
#         """)
        
#         # Input Section
#         with gr.Column(elem_classes="input-section"):
#             query_input = gr.Textbox(
#                 placeholder="Type your question here...",
#                 show_label=False,
#                 lines=1,
#                 max_lines=3,
#                 elem_id="query-input",
#                 elem_classes="dark-input"
#             )
#             submit_btn = gr.Button("Ask Question →", variant="primary", scale=0, elem_classes="dark-button")
        
#         # Example Questions
#         gr.HTML('<h2 class="example-header" style="color: #101828 !important; opacity: 1 !important; -webkit-text-fill-color: #101828 !important;">Example Questions</h2>')
        
#         # Create example buttons
#         example_html = '<div class="example-grid">'
#         for ex in example_questions:
#             example_html += f'''
#                 <div class="example-card" style="opacity: 1 !important; background: white !important;" onclick="document.getElementById('query-input').querySelector('textarea').value='{ex["text"]}'; 
#                      document.getElementById('query-input').querySelector('textarea').dispatchEvent(new Event('input', {{bubbles: true}}));">
#                     <div class="example-icon" style="opacity: 1 !important;">{ex["icon"]}</div>
#                     <div class="example-text" style="color: #101828 !important; opacity: 1 !important; font-weight: 500 !important; -webkit-text-fill-color: #101828 !important;">{ex["text"]}</div>
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
#     # Launch with custom theme colors
#     demo.launch(share=True)  # Set share=False in production


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


# # Custom CSS for professional look
# custom_css = """
# /* CSS Reset - Force all elements to be fully visible */
# *, *::before, *::after {
#     opacity: 1 !important;
#     -webkit-opacity: 1 !important;
#     -moz-opacity: 1 !important;
#     -o-opacity: 1 !important;
# }

# /* Force text color on common elements */
# p, h1, h2, h3, h4, h5, h6, span, div, li, a, input, textarea, button {
#     opacity: 1 !important;
# }

# /* Force Gradio specific elements */
# .gr-box, .gr-form, .gr-input-label, .gr-panel, .gr-text-input, .gr-button {
#     opacity: 1 !important;
# }

# .gradio-container {
#     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
#     background-color: #f8f9fa !important;
# }

# .gradio-container * {
#     opacity: 1 !important;
# }

# .main-container {
#     max-width: 900px !important;
#     margin: 0 auto !important;
#     padding: 40px 20px !important;
# }

# .header-section {
#     text-align: center;
#     margin-bottom: 48px;
# }

# .header-title {
#     font-size: 32px !important;
#     font-weight: 700 !important;
#     color: #101828 !important;
#     margin-bottom: 8px !important;
#     opacity: 1 !important;
# }

# .header-subtitle {
#     font-size: 18px !important;
#     color: #344054 !important;
#     font-weight: 400 !important;
#     opacity: 1 !important;
# }

# .input-section {
#     background: white;
#     border-radius: 16px;
#     padding: 32px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.08);
#     margin-bottom: 32px;
# }

# .example-section {
#     margin-bottom: 32px;
# }

# .example-header {
#     font-size: 20px !important;
#     font-weight: 600 !important;
#     color: #101828 !important;
#     margin-bottom: 16px !important;
#     opacity: 1 !important;
# }

# .example-grid {
#     display: grid;
#     grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
#     gap: 16px;
# }

# .example-card {
#     background: white !important;
#     border: 1px solid #e4e7ec !important;
#     border-radius: 12px !important;
#     padding: 20px !important;
#     cursor: pointer !important;
#     transition: all 0.2s ease !important;
#     display: flex !important;
#     align-items: center !important;
#     gap: 12px !important;
#     opacity: 1 !important;
# }

# .example-card:hover {
#     border-color: #0066cc !important;
#     box-shadow: 0 4px 12px rgba(0,102,204,0.15) !important;
#     transform: translateY(-2px) !important;
# }

# .example-icon {
#     font-size: 24px !important;
#     width: 40px !important;
#     height: 40px !important;
#     background: #eff8ff !important;
#     border-radius: 8px !important;
#     display: flex !important;
#     align-items: center !important;
#     justify-content: center !important;
#     opacity: 1 !important;
# }

# .example-text {
#     flex: 1 !important;
#     color: #101828 !important;
#     font-size: 15px !important;
#     font-weight: 500 !important;
#     opacity: 1 !important;
# }

# .output-section {
#     background: white;
#     border-radius: 16px;
#     padding: 32px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.08);
#     min-height: 200px;
# }

# .output-section * {
#     color: #101828 !important;
#     opacity: 1 !important;
# }

# .empty-state {
#     text-align: center !important;
#     color: #667085 !important;
#     padding: 60px 20px !important;
#     font-size: 16px !important;
#     opacity: 1 !important;
# }

# /* Gradio component overrides */
# .dark-button {
#     opacity: 1 !important;
# }

# .gr-button-primary {
#     background-color: #0066cc !important;
#     border: none !important;
#     font-weight: 500 !important;
#     padding: 10px 24px !important;
#     font-size: 16px !important;
#     border-radius: 8px !important;
#     transition: all 0.2s ease !important;
#     opacity: 1 !important;
#     color: white !important;
# }

# .gr-button-primary:hover {
#     background-color: #0052a3 !important;
#     transform: translateY(-1px) !important;
#     box-shadow: 0 4px 12px rgba(0,102,204,0.25) !important;
# }

# .gr-form {
#     border: none !important;
#     background: transparent !important;
#     padding: 0 !important;
# }

# .gr-box {
#     border: none !important;
#     background: transparent !important;
# }

# .gr-padded {
#     padding: 0 !important;
# }

# .gr-panel {
#     border: none !important;
#     background: transparent !important;
# }

# /* Input field styling */
# .dark-input textarea {
#     color: #101828 !important;
#     background-color: white !important;
#     opacity: 1 !important;
#     pointer-events: auto !important;
#     z-index: 10 !important;
# }

# textarea, input[type="text"] {
#     border: 2px solid #e4e7ec !important;
#     border-radius: 8px !important;
#     padding: 12px 16px !important;
#     font-size: 16px !important;
#     transition: all 0.2s ease !important;
#     color: #101828 !important;
#     background-color: white !important;
#     opacity: 1 !important;
#     pointer-events: auto !important;
#     z-index: 10 !important;
# }

# textarea:focus, input[type="text"]:focus {
#     border-color: #0066cc !important;
#     outline: none !important;
#     box-shadow: 0 0 0 3px rgba(0,102,204,0.1) !important;
#     color: #101828 !important;
# }

# textarea::placeholder, input[type="text"]::placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# label {
#     display: none !important;
# }

# /* Force all output text to be visible */
# .output-section p,
# .output-section li,
# .output-section span,
# .output-section div,
# .output-section strong,
# .output-section a {
#     opacity: 1 !important;
#     color: #101828 !important;
# }

# .output-section a {
#     color: #0066cc !important;
# }

# /* Override any Gradio default styles that might cause fading */
# .gr-prose * {
#     opacity: 1 !important;
# }

# .dark .gr-prose * {
#     opacity: 1 !important;
# }

# /* Ensure all text elements are visible */
# h1, h2, h3, h4, h5, h6, p, span, div, li, a {
#     opacity: 1 !important;
# }

# /* Force dark text color on all text elements */
# .gradio-container h1,
# .gradio-container h2,
# .gradio-container h3,
# .gradio-container p,
# .gradio-container span,
# .gradio-container div {
#     color: #101828 !important;
#     opacity: 1 !important;
# }

# /* Override any theme-specific opacity */
# .light *, .dark * {
#     opacity: 1 !important;
# }

# /* Soft theme specific overrides */
# .soft {
#     --body-text-color: #101828 !important;
#     --color-accent-soft: #101828 !important;
#     --body-text-color-subdued: #344054 !important;
# }

# .soft * {
#     opacity: 1 !important;
# }

# /* Final overrides to ensure visibility */
# .gradio-container * {
#     opacity: 1 !important;
# }

# .gr-text-input {
#     opacity: 1 !important;
# }

# .gr-input {
#     opacity: 1 !important;
# }

# .gr-button {
#     opacity: 1 !important;
# }

# /* Ensure example cards are fully visible */
# .example-card * {
#     opacity: 1 !important;
# }

# /* Force all text to be dark */
# body, .gradio-container {
#     color: #101828 !important;
# }

# /* Anti-gray-out measures */
# .gradio-container :not(button) {
#     color: #101828 !important;
# }

# .gr-prose {
#     color: #101828 !important;
# }

# .gr-padded {
#     color: #101828 !important;
# }

# /* Ensure no disabled styles are applied */
# :disabled {
#     opacity: 1 !important;
#     color: #101828 !important;
# }

# /* Remove any backdrop filters */
# * {
#     backdrop-filter: none !important;
#     -webkit-backdrop-filter: none !important;
# }

# /* Handle Gradio pending/processing states */
# .pending {
#     opacity: 1 !important;
# }

# .processing {
#     opacity: 1 !important;
# }

# /* Ensure all states are visible */
# *[style*="opacity"] {
#     opacity: 1 !important;
# }

# /* Override inline styles */
# [style*="color: rgb"] {
#     color: #101828 !important;
# }

# /* Final nuclear option - use CSS variables */
# :root {
#     --tw-text-opacity: 1 !important;
#     --text-opacity: 1 !important;
# }

# /* Target Gradio's label and text elements */
# .gr-input-label, .gr-button, .gr-check-radio, .gr-checkbox-label {
#     opacity: 1 !important;
#     color: #101828 !important;
# }

# /* Ensure placeholder text is visible */
# ::placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# ::-webkit-input-placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# ::-moz-placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# :-ms-input-placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }

# /* Override dark mode if active */
# .dark {
#     filter: none !important;
# }

# .dark * {
#     opacity: 1 !important;
#     filter: none !important;
# }

# /* Remove any theme-based opacity */
# [class*="opacity-"] {
#     opacity: 1 !important;
# }

# /* Ensure Gradio containers are visible */
# .container {
#     opacity: 1 !important;
# }

# .gr-panel {
#     opacity: 1 !important;
# }

# /* Ultimate override - target all elements with any opacity */
# *:not(img):not(svg) {
#     opacity: 1 !important;
#     min-opacity: 1 !important;
#     max-opacity: 1 !important;
# }

# /* Force text color on text elements only */
# h1, h2, h3, h4, h5, h6, p, label, input, textarea {
#     color: #101828 !important;
#     -webkit-text-fill-color: #101828 !important;
# }

# /* Specific overrides for headers and example text */
# .header-title, .header-subtitle, .example-header, .example-text {
#     color: #101828 !important;
#     -webkit-text-fill-color: #101828 !important;
#     opacity: 1 !important;
# }

# /* Ensure spans and divs with text are visible */
# span, div {
#     opacity: 1 !important;
# }

# /* But keep their color inheritance unless specified */
# .main-container span, .main-container div {
#     color: inherit;
# }

# /* Remove any gray text colors */
# [style*="color: gray"], [style*="color: #999"], [style*="color: #666"], [style*="color: #ccc"] {
#     color: #101828 !important;
# }

# /* Final insurance - no element should have low opacity or gray color */
# * {
#     min-opacity: 0.99;
# }

# /* Specific fix for Gradio text inputs */
# .gr-text-input input, .gr-text-input textarea {
#     color: #101828 !important;
#     -webkit-text-fill-color: #101828 !important;
#     opacity: 1 !important;
# }

# /* === Fix: bring the Ask‐Question textarea to the front and re‐enable clicking === */
# #query-input textarea {
#     position: relative !important;
#     z-index: 9999     !important;
#     pointer-events: auto !important;
# }
# """

# custom_css += """
# /* Critical input field fixes */
# #query-input textarea {
#     background-color: white !important;
#     color: #101828 !important;
#     -webkit-text-fill-color: #101828 !important;
#     caret-color: #101828 !important;
#     border: 2px solid #e4e7ec !important;
# }

# #query-input textarea:focus {
#     background-color: white !important;
#     color: #101828 !important;
#     border-color: #0066cc !important;
#     box-shadow: 0 0 0 3px rgba(0,102,204,0.1) !important;
# }

# #query-input::placeholder {
#     color: #667085 !important;
#     opacity: 1 !important;
# }
# """


# # Example questions
# example_questions = [
#     {"icon": "📋", "text": "What is Technova's mission statement?"},
#     {"icon": "📊", "text": "What were last quarter's key metrics?"},
#     {"icon": "👥", "text": "What is the policy on Employee Dating & Relationships"},
#     {"icon": "📦", "text": "What are our core products?"},
#     {"icon": "🏠", "text": "What is our remote work policy?"},
#     {"icon": "💼", "text": "How does our code review process work?"},
# ]

# # Build the interface using Blocks
# with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
#     # Add inline styles to force visibility
#     gr.HTML("""
#         <style>
#         /* Global reset to force visibility */
#         * {
#             opacity: 1 !important;
#             -webkit-opacity: 1 !important;
#         }
        
#         /* Force all Gradio elements to be visible */
#         .gradio-container, .gradio-container * {
#             opacity: 1 !important;
#         }
        
#         /* Force specific text elements */
#         .main-container *, 
#         .header-title, 
#         .header-subtitle,
#         .example-header,
#         .example-text,
#         .example-card,
#         h1, h2, h3, p, span, div, input, textarea, button {
#             opacity: 1 !important;
#             color: inherit !important;
#         }
        
#         /* Ensure dark text where needed */
#         .header-title { color: #101828 !important; }
#         .header-subtitle { color: #344054 !important; }
#         .example-header { color: #101828 !important; }
#         .example-text { color: #101828 !important; }
        
#         /* Critical fix for input text visibility */
#         #query-input textarea {
#             background-color: white !important;
#             color: #101828 !important;
#             opacity: 1 !important;
#             -webkit-text-fill-color: #101828 !important;
#             caret-color: #101828 !important;
#         }
        
#         /* Remove any filters that might cause graying */
#         * {
#             filter: none !important;
#             -webkit-filter: none !important;
#         }
#         </style>
#         <script>
#         // Comprehensive visibility enforcement
#         (function() {
#             // Function to force visibility on all elements
#             const forceVisibility = function() {
#                 // Get all elements
#                 const allElements = document.querySelectorAll('*');
#                 allElements.forEach(element => {
#                     // Force opacity
#                     element.style.setProperty('opacity', '1', 'important');
#                     element.style.setProperty('filter', 'none', 'important');
                    
#                     // Check computed style and override if needed
#                     const computed = window.getComputedStyle(element);
#                     if (computed.opacity !== '1') {
#                         element.style.cssText += 'opacity: 1 !important;';
#                     }
#                 });
                
#                 // Specific fixes for text elements
#                 const textElements = document.querySelectorAll('h1, h2, h3, p, span, div, input, textarea, label');
#                 textElements.forEach(element => {
#                     if (!element.closest('button')) {  // Don't change button text
#                         element.style.setProperty('color', '#101828', 'important');
#                         element.style.setProperty('-webkit-text-fill-color', '#101828', 'important');
#                     }
#                 });
                
#                 // Fix placeholders
#                 const inputs = document.querySelectorAll('input, textarea');
#                 inputs.forEach(input => {
#                     input.style.setProperty('color', '#101828', 'important');
#                     input.style.setProperty('-webkit-text-fill-color', '#101828', 'important');
#                 });
                
#                 // CRITICAL FIX: Force the query input to have white background and dark text
#                 const queryInput = document.querySelector('#query-input textarea');
#                 if (queryInput) {
#                     queryInput.style.cssText += 'background-color: white !important; color: #101828 !important; -webkit-text-fill-color: #101828 !important; caret-color: #101828 !important;';
#                 }
#             };
            
#             // Run on various events
#             window.addEventListener('DOMContentLoaded', forceVisibility);
#             window.addEventListener('load', forceVisibility);
            
#             // Run periodically to ensure styles stick
#             setInterval(forceVisibility, 500);
            
#             // Monitor for DOM changes
#             const observer = new MutationObserver(function(mutations) {
#                 forceVisibility();
                
#                 // Special handling for the query input
#                 const queryInput = document.querySelector('#query-input textarea');
#                 if (queryInput) {
#                     queryInput.style.cssText += 'background-color: white !important; color: #101828 !important; -webkit-text-fill-color: #101828 !important; caret-color: #101828 !important;';
#                 }
#             });
            
#             // Start observing when ready
#             window.addEventListener('DOMContentLoaded', function() {
#                 observer.observe(document.body, {
#                     childList: true,
#                     subtree: true,
#                     attributes: true,
#                     attributeFilter: ['style', 'class']
#                 });
                
#                 // Direct fix for the textarea
#                 setTimeout(() => {
#                     const queryTextarea = document.querySelector('#query-input textarea');
#                     if (queryTextarea) {
#                         queryTextarea.style.cssText += 'background-color: white !important; color: #101828 !important; -webkit-text-fill-color: #101828 !important; caret-color: #101828 !important;';
                        
#                         // Add input event listener to ensure style persists during typing
#                         queryTextarea.addEventListener('focus', () => {
#                             queryTextarea.style.cssText += 'background-color: white !important; color: #101828 !important; -webkit-text-fill-color: #101828 !important; caret-color: #101828 !important;';
#                         });
                        
#                         queryTextarea.addEventListener('input', () => {
#                             queryTextarea.style.cssText += 'background-color: white !important; color: #101828 !important; -webkit-text-fill-color: #101828 !important; caret-color: #101828 !important;';
#                         });
#                     }
#                 }, 500);
#             });
#         })();
#         </script>
# """)
    
#     with gr.Column(elem_classes="main-container"):
#         # Header
#         gr.HTML("""
#             <div class="header-section">
#                 <h1 class="header-title" style="color: #101828 !important; opacity: 1 !important; -webkit-text-fill-color: #101828 !important;">🏢 TechNova Knowledge Assistant</h1>
#                 <p class="header-subtitle" style="color: #344054 !important; opacity: 1 !important; -webkit-text-fill-color: #344054 !important;">Get instant answers from our company knowledge base</p>
#             </div>
#         """)
        
#         # Input Section
#         with gr.Column(elem_classes="input-section"):
#             query_input = gr.Textbox(
#                 placeholder="Type your question here...",
#                 show_label=False,
#                 lines=1,
#                 max_lines=3,
#                 elem_id="query-input",
#                 elem_classes="dark-input",
#                 interactive=True,
#                 render=False
#             )

#             query_input = query_input.render()
#             query_input.root.style = "background-color: white !important; color: #101828 !important;"



#             submit_btn = gr.Button("Ask Question →", variant="primary", scale=0, elem_classes="dark-button")
        
#         # Example Questions
#         gr.HTML('<h2 class="example-header" style="color: #101828 !important; opacity: 1 !important; -webkit-text-fill-color: #101828 !important;">Example Questions</h2>')
        
#         # Create example buttons
#         example_html = '<div class="example-grid">'
#         for ex in example_questions:
#             example_html += f'''
#                 <div class="example-card" style="opacity: 1 !important; background: white !important;" onclick="document.getElementById('query-input').querySelector('textarea').value='{ex["text"]}'; 
#                      document.getElementById('query-input').querySelector('textarea').dispatchEvent(new Event('input', {{bubbles: true}}));">
#                     <div class="example-icon" style="opacity: 1 !important;">{ex["icon"]}</div>
#                     <div class="example-text" style="color: #101828 !important; opacity: 1 !important; font-weight: 500 !important; -webkit-text-fill-color: #101828 !important;">{ex["text"]}</div>
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
#     # Launch with custom theme colors
#     demo.launch(share=True)  # Set share=False in production
