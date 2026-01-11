# from ytbot import *
# import gradio as gr

# with gr.Blocks() as interface:
#     # Input field for YouTube URL
#     video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
#     # Outputs for summary and answer
#     summary_output = gr.Textbox(label="Video Summary", lines=5)
#     question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
#     answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

#     # Buttons for selecting functionalities after fetching transcript
#     summarize_btn = gr.Button("Summarize Video")
#     question_btn = gr.Button("Ask a Question")

#     # Display status message for transcript fetch
#     transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

#     # Set up button actions
#     summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
#     question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# # Launch the app with specified server name and port
# interface.launch(server_name="0.0.0.0", server_port=7000)

from ytbot import *
import gradio as gr


def summarize_wrapper(video_url):
    if not video_url:
        return "Please enter a valid YouTube URL."
    return summarize_video(video_url)


def qa_wrapper(video_url, question):
    if not video_url or not question:
        return "Please enter both a YouTube URL and a question."
    return answer_question(video_url, question)


with gr.Blocks(title="AI-Powered YouTube Summarizer & Q&A") as interface:
    gr.Markdown("## 🎥 AI-Powered YouTube Summarizer & Q&A")

    # Input field for YouTube URL
    video_url = gr.Textbox(
        label="YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    # Buttons
    summarize_btn = gr.Button("📄 Summarize Video")
    question_btn = gr.Button("❓ Ask a Question")

    # Outputs
    summary_output = gr.Textbox(
        label="Video Summary",
        lines=8
    )

    question_input = gr.Textbox(
        label="Ask a Question About the Video",
        placeholder="What is the video about?"
    )

    answer_output = gr.Textbox(
        label="Answer",
        lines=6
    )

    # Button actions
    summarize_btn.click(
        summarize_wrapper,
        inputs=video_url,
        outputs=summary_output
    )

    question_btn.click(
        qa_wrapper,
        inputs=[video_url, question_input],
        outputs=answer_output
    )


# Launch the app
interface.launch(
    server_name="0.0.0.0",
    server_port=7000,
    show_error=True
)
