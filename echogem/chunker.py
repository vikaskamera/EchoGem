"""
Transcript chunking module using Google Gemini for intelligent segmentation.
"""

import os
import re
import json
import google.generativeai as genai
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .models import Chunk


class Chunker:
    """
    Intelligent transcript chunking using LLM-based semantic analysis
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: str = "all-MiniLM-L6-v2",
        max_tokens: int = 2000,
        similarity_threshold: float = 0.82,
        coherence_threshold: float = 0.75,
    ):
        """
        Initialize the chunker
        
        Args:
            api_key: Google API key for Gemini
            embed_model: Path to sentence transformer model or model name
            max_tokens: Maximum tokens per chunk
            similarity_threshold: Threshold for semantic similarity
            coherence_threshold: Threshold for coherence
        """
        # Initialize Google Gemini
        if api_key:
            genai.configure(api_key=api_key)
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")
        
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize sentence transformer
        try:
            self.embedder = SentenceTransformer(embed_model)
        except Exception as e:
            print(f"Warning: Could not load model {embed_model}: {e}")
            print("Using default model")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.max_tokens = max_tokens
        self.sim_threshold = similarity_threshold
        self.coh_threshold = coherence_threshold

    def load_transcript(self, file_path: str) -> str:
        """
        Load transcript text from file
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            Transcript text content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read()
            print(f"Transcript loaded ({len(transcript)} characters)")
            return transcript
        except FileNotFoundError:
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading transcript: {str(e)}")

    def chunk_transcript(self, transcript: str) -> List[Chunk]:
        """
        Chunk transcript using LLM-based semantic analysis
        
        Args:
            transcript: Transcript text to chunk
            
        Returns:
            List of Chunk objects
        """
        try:
            # Create chunking prompt
            prompt = self._create_prompt(transcript)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse response
            chunks = self._parse_chunk_response(response.text)
            
            print(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"Error during chunking: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(transcript)

    def _create_prompt(self, transcript: str) -> str:
        """Create the chunking prompt"""
        return f"""
        **SYSTEM PROMPT**
        You are a transcript processing expert. The following transcript needs to be chunked very intelligently and logically. Ensure sensible segments and structure to be later provided as context to answer questions.

        **INSTRUCTIONS**
        1. Create as many or as few chunks as needed
        2. Each chunk should contain consecutive sentences
        3. For each chunk provide:
          - title: 2-5 word summary
          - content: exact sentences
          - keywords: 3-5 important terms
          - named_entities: any mentioned names
          - timestamp_range: estimate like "00:00-01:30"

        **TRANSCRIPT**
        {transcript[:5000]}...

        **OUTPUT FORMAT**
        You must output ONLY valid JSON in this exact format:
        {{
          "chunks": [
            {{
              "title": "Summary",
              "content": "Actual sentences",
              "keywords": ["term1", "term2"],
              "named_entities": ["Name"],
              "timestamp_range": "00:00-01:30"
            }}
          ]
        }}
        """

    def _parse_chunk_response(self, response_text: str) -> List[Chunk]:
        """Parse the LLM response into Chunk objects.

        This method is robust to extra explanatory text before/after the JSON
        by scanning for the first valid JSON object or array using
        json.JSONDecoder.raw_decode.
        """
        if not isinstance(response_text, str):
            raise ValueError("Expected string response from LLM")

        decoder = json.JSONDecoder()
        data = None
        last_error: Optional[Exception] = None

        # First, try to parse the whole response as JSON
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            last_error = e

        # If that fails, scan for the first JSON object/array within the text
        if data is None:
            for idx, ch in enumerate(response_text):
                if ch in "{[":
                    try:
                        data, end = decoder.raw_decode(response_text[idx:])
                        break
                    except json.JSONDecodeError as e:
                        last_error = e
                        continue

        if data is None:
            raise ValueError(f"Could not parse JSON from response: {last_error}")

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object with 'chunks' field, got {type(data).__name__}")

        raw_chunks = data.get("chunks") or []
        if not isinstance(raw_chunks, list):
            raise ValueError("Expected 'chunks' to be a list in JSON response")

        chunks: List[Chunk] = []
        for chunk_data in raw_chunks:
            if not isinstance(chunk_data, dict):
                continue

            content = (chunk_data.get("content") or "").strip()
            if not content:
                continue

            chunk = Chunk(
                title=(chunk_data.get("title") or "Untitled").strip() or "Untitled",
                content=content,
                keywords=chunk_data.get("keywords") or [],
                named_entities=chunk_data.get("named_entities") or [],
                timestamp_range=chunk_data.get("timestamp_range") or "",
                chunk_id=f"chunk_{len(chunks)}",
            )
            chunks.append(chunk)

        if not chunks:
            raise ValueError("No valid chunks parsed from JSON response")

        return chunks

    def _fallback_chunking(self, transcript: str) -> List[Chunk]:
        """Fallback chunking that preserves semantic boundaries where possible.

        Instead of naive fixed-word chunks, this method:
        - Splits on paragraphs and sentences when possible
        - Packs consecutive sentences into chunks up to an approximate
          max token/word budget
        - Only falls back to word-based splitting for extremely long sentences
        """
        text = transcript.strip()
        if not text:
            return []

        max_words = max(self.max_tokens, 1)
        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_word_count = 0

        def flush_chunk() -> None:
            nonlocal current_sentences, current_word_count
            if not current_sentences:
                return
            chunk_text = " ".join(current_sentences).strip()
            if not chunk_text:
                current_sentences = []
                current_word_count = 0
                return

            chunk = Chunk(
                title=f"Chunk {len(chunks) + 1}",
                content=chunk_text,
                keywords=[],
                named_entities=[],
                timestamp_range="",
                chunk_id=f"fallback_chunk_{len(chunks)}",
            )
            chunks.append(chunk)
            current_sentences = []
            current_word_count = 0

        # Split into paragraphs first to respect larger structural breaks
        paragraphs = re.split(r"\n\s*\n+", text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Split paragraph into sentences using simple punctuation heuristic
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                words_in_sentence = sentence.split()
                sentence_word_count = len(words_in_sentence)

                # Handle sentences longer than max_words by word-based splitting
                if sentence_word_count > max_words:
                    # Finish any current chunk before splitting this long sentence
                    flush_chunk()

                    for i in range(0, sentence_word_count, max_words):
                        chunk_words = words_in_sentence[i : i + max_words]
                        chunk_text = " ".join(chunk_words).strip()
                        if not chunk_text:
                            continue
                        chunk = Chunk(
                            title=f"Chunk {len(chunks) + 1}",
                            content=chunk_text,
                            keywords=[],
                            named_entities=[],
                            timestamp_range="",
                            chunk_id=f"fallback_chunk_{len(chunks)}",
                        )
                        chunks.append(chunk)
                    continue

                # If adding this sentence would exceed the limit, start a new chunk
                if current_word_count + sentence_word_count > max_words:
                    flush_chunk()

                current_sentences.append(sentence)
                current_word_count += sentence_word_count

        # Flush any remaining content
        flush_chunk()

        return chunks

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using sentence transformer"""
        try:
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return [0.0] * 384  # Default dimension
