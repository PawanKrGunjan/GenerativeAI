# Import necessary libraries for the YouTube bot
#import gradio as gr
import re  #For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain_ollama import OllamaLLM, OllamaEmbeddings
#from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate

class RAGConfig:
    # Models
    TINY_LLM_MODEL = "llama3.2:1b"          # ~2–3.5 GB
    FULL_LLM_MODEL = "llama3.2:latest"      # only if 16+ GB RAM
    EMBEDDING_MODEL = "nomic-embed-text"
    HF_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    @staticmethod
    def get_llm(tiny: bool = True) -> OllamaLLM:
        model = RAGConfig.TINY_LLM_MODEL if tiny else RAGConfig.FULL_LLM_MODEL
        return OllamaLLM(
            model=model,
            temperature=0.1,
            max_tokens=300
        )

    @staticmethod
    def get_embeddings() -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=RAGConfig.EMBEDDING_MODEL
        )



def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)
    
    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()
    
    # Fetch the list of available transcripts for the given YouTube video
    transcripts = ytt_api.list(video_id)
    
    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
    
    return transcript if transcript else None

def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""
    
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    # Return the processed transcript as a single string
    return txt

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks

def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
    
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """
    # Use the FAISS library to create an index from the provided text chunks
    return FAISS.from_texts(chunks, embedding_model)

def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    :return: PromptTemplate object
    """
    # Define the template for the summary prompt
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    
    return prompt

def create_summary_chain(llm, prompt):
    """
    Create a runnable chain for generating summaries
    using modern LangChain (LCEL).

    :param llm: LLM instance (e.g., Ollama)
    :param prompt: PromptTemplate instance
    :return: Runnable chain
    """
    return prompt | llm

def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context

def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.

    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
    
    # Define the template string
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context}

    Based on the above context, please answer the following question:
    Question: {question}
    """

    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    return prompt_template

def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    
    #return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)
    return prompt_template | llm

def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)
    # Generate answer using the QA chain
    context_text = "\n".join([doc.page_content for doc in relevant_context])

    answer = qa_chain.invoke({
        "context": context_text,
        "question": question
    })

    return answer

# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # Step 1: Set up IBM Watson credentials
        #model_id, credentials, client, project_id = setup_credentials()

        # Step 2: Initialize WatsonX LLM for summarization
        llm = RAGConfig.get_llm(tiny=True)

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.invoke({
            "transcript": processed_transcript
        })

        return summary
    else:
        return "No transcript available. Please fetch the transcript first."
    
def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up IBM Watson credentials
        #model_id, credentials, client, project_id = setup_credentials()

        # Step 3: Initialize WatsonX LLM for Q&A
        llm = RAGConfig.get_llm(tiny=True)

        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = RAGConfig.get_embeddings()
        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."
   
if __name__ == "__main__":

    video_url = 'https://www.youtube.com/watch?v=9WpcgTpJkSw'
    video_url = "https://www.youtube.com/watch?v=n0fRyDcG1BM"
    video_url = 'https://www.youtube.com/watch?v=noDGpTtGhhI'

    print("-" * 100)

    # Get video ID
    video_id = get_video_id(video_url)
    print("VIDEO ID:", video_id)
    print("-" * 100)

    # Fetch transcript
    transcript = get_transcript(video_url)
    print("RAW TRANSCRIPTION:")
    print(transcript)
    print("-" * 100)

    # Test QA prompt formatting
    qa_prompt_template = create_qa_prompt_template()

    context = "Attention is all you need"
    question = "What are the key principles discussed in the video?"

    generated_prompt = qa_prompt_template.format(
        context=context,
        question=question
    )

    print("Generated Prompt:")
    print(generated_prompt)
    print("-" * 100)

    # ✅ Run video summarization
    video_summary = summarize_video(video_url)
    print("Video Summary:")
    print(video_summary)
    print("-" * 100)

    # ✅ Run question answering
    user_question = "Who is the speaker"
    answer = answer_question(video_url, user_question)
    print("Answer:")
    print(answer)
    print("-" * 100)
