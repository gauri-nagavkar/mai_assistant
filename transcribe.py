import sys
import whisper
from transformers import pipeline

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes the given audio file using the Whisper model.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def chunk_text(text: str, tokenizer, max_tokens: int) -> list:
    """
    Splits text into chunks that do not exceed max_tokens.
    Uses a tokenizer to properly split and then decodes each chunk.
    """
    # Get token IDs without special tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk)
    return chunks

def summarize_text(text: str) -> str:
    """
    Summarizes the text using a summarization pipeline.
    If the text exceeds the model's max input tokens (with a safety buffer),
    it is split into manageable chunks.
    """
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    tokenizer = summarizer.tokenizer
    max_input_length = tokenizer.model_max_length  # typically 1024 tokens for BART
    # Leave a safety buffer to avoid hitting the absolute maximum.
    safe_max_length = max_input_length - 10

    # Tokenize the input without special tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > safe_max_length:
        print(f"Input is too long ({len(tokens)} tokens). Splitting into chunks...")
        chunks = chunk_text(text, tokenizer, safe_max_length)
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            summary_chunk = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary_chunk[0]['summary_text'])
        # Combine the summaries from all chunks.
        combined_summary = " ".join(summaries)
        print("Summarizing the combined summary...")
        final_summary = summarizer(combined_summary, max_length=150, min_length=30, do_sample=False)
        return final_summary[0]['summary_text']
    else:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_summarize.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"Transcribing audio file: {audio_file}...")
    transcript = transcribe_audio(audio_file)
    print("\nTranscript:")
    print(transcript)
    
    print("\nSummarizing transcript...")
    summary = summarize_text(transcript)
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()
