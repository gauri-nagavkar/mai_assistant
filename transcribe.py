import sys
import whisper
import nltk
import ollama
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes the given audio file using the Whisper model.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def better_chunk_text(text: str, tokenizer, max_tokens: int) -> list:
    """
    Splits text into chunks without breaking sentences.
    Uses NLTK's sent_tokenize to split the transcript into sentences,
    then aggregates sentences until the token count reaches max_tokens.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        candidate_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        tokens = tokenizer(candidate_chunk)

        if len(tokens) <= max_tokens:
            current_chunk = candidate_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def summarize_text(text: str) -> str:
    """
    Summarizes the text using a locally running LLM via Ollama (Mistral-7B).
    """
    prompt = (
        "Summarize the following transcript concisely. "
        "Preserve key decisions, discussions, and important points but omit general chatter.\n\n"
        f"Transcript:\n{text}"
    )

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

def extract_action_items(text: str) -> list:
    """
    Extracts action items from the transcript using a local AI model (Mistral-7B via Ollama).
    If the transcript is too long, it is chunked before processing.
    """
    max_tokens = 2048  # Mistral-7B token limit per request

    def tokenize(t): return ollama.chat(model="mistral", messages=[{"role": "user", "content": t}])['message']['content'].split()

    tokens = tokenize(text)
    action_items = []

    if len(tokens) > max_tokens:
        print(f"Transcript is too long ({len(tokens)} tokens). Splitting into chunks...")
        chunks = better_chunk_text(text, tokenize, max_tokens)

        for i, chunk in enumerate(chunks):
            print(f"Extracting action items from chunk {i+1}/{len(chunks)}...")
            prompt = (
                "Extract the past, current and future action items from the following meeting transcript. "
                "Only include key tasks (new, assigned, pending and completed), decisions, or follow-ups, ignoring general discussion.\n\n"
                f"Transcript:\n{chunk}"
            )
            response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            items = [item.strip() for item in response['message']['content'].split("\n") if item.strip()]
            action_items.extend(items)
    else:
        prompt = (
            "Extract the past, current and future action items from the following meeting transcript. "
            "Only include key tasks (new, assigned, pending and completed), decisions, or follow-ups, ignoring general discussion.\n\n"
            f"Transcript:\n{text}"
        )
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        action_items = [item.strip() for item in response['message']['content'].split("\n") if item.strip()]

    # Remove duplicates
    return list(dict.fromkeys(action_items))

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

    print("\nExtracting action items from transcript...")
    action_items = extract_action_items(transcript)
    print("\nAction Items:")
    for i, item in enumerate(action_items, start=1):
        print(f"{i}. {item}")

if __name__ == "__main__":
    main()
