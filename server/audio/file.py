import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from pyannote.audio import Pipeline
from scipy.stats import zscore


class HybridVoiceRiskEngine:
    def __init__(self, device=None, hf_auth_token=None):
        """
        Initialize the Hybrid Voice Risk Engine
        
        Args:
            device: 'cuda' or 'cpu' (auto-detected if None)
            hf_auth_token: HuggingFace auth token for pyannote model
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_auth_token = hf_auth_token
        
        print(f"\nRunning on: {self.device.upper()}\n")
        
        # -------------------------
        # Load Models
        # -------------------------
        self._load_models()
        
    def _load_models(self):
        """Load all required models"""
        try:
            print("Loading diarization model...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=self.hf_auth_token
            )
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            print("Please provide a valid HuggingFace auth token")
            sys.exit(1)

        print("Loading emotion model...")
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True
        )

        print("Loading sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            device=0 if self.device == "cuda" else -1
        )

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        
        print("All models loaded successfully.\n")

    # ==========================================================
    # 1️⃣ Speaker Diarization
    # ==========================================================

    def diarize_audio(self, audio_path):
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of segments with start, end, and speaker label
        """
        diarization = self.diarization_pipeline(audio_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments

    # ==========================================================
    # 2️⃣ Identify Banker vs Customer
    # ==========================================================

    def identify_customer(self, transcript_segments):
        """
        Identify which speaker is the customer based on question patterns
        
        Args:
            transcript_segments: List of transcribed segments with text
            
        Returns:
            Speaker label of the banker (customer is the other speaker)
        """
        speaker_stats = defaultdict(lambda: {"questions": 0, "length": 0})

        for seg in transcript_segments:
            text = seg["text"]
            speaker = seg["speaker"]

            speaker_stats[speaker]["length"] += len(text.split())
            speaker_stats[speaker]["questions"] += text.count("?")

        # Banker typically asks more questions
        banker = max(speaker_stats.items(), key=lambda x: x[1]["questions"])[0]
        
        print(f"Identified banker: {banker}")
        
        return banker

    # ==========================================================
    # 3️⃣ Extract Acoustic Features
    # ==========================================================

    def extract_acoustic_features(self, audio_path, segments, customer_label):
        """
        Extract acoustic features (stress, pitch volatility) for customer segments
        
        Args:
            audio_path: Path to audio file
            segments: Diarized segments
            customer_label: Speaker label of customer
            
        Returns:
            stress_score: Normalized stress level
            volatility_score: Pitch volatility
        """
        y, sr = librosa.load(audio_path, sr=None)

        stress_scores = []
        pitch_vars = []

        for seg in segments:
            if seg["speaker"] != customer_label:
                continue

            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) < sr * 0.5:  # Skip segments shorter than 0.5 seconds
                continue

            # Extract pitch features
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]

            if len(pitch_values) > 0:
                pitch_vars.append(np.var(pitch_values))

            # Extract energy features
            rms = librosa.feature.rms(y=segment_audio)
            energy = np.mean(rms)
            stress_scores.append(energy)

        # Calculate normalized stress score
        if len(stress_scores) > 1:
            stress_score = float(np.mean(zscore(stress_scores)))
        else:
            stress_score = 0.0
            
        # Calculate pitch volatility
        volatility_score = float(np.std(pitch_vars)) if pitch_vars else 0.0

        return stress_score, volatility_score

    # ==========================================================
    # 4️⃣ NLP Analysis
    # ==========================================================

    def analyze_text(self, transcript_segments, customer_label):
        """
        Perform NLP analysis on customer's speech
        
        Args:
            transcript_segments: List of transcribed segments
            customer_label: Speaker label of customer
            
        Returns:
            Dictionary of NLP-based scores
        """
        # Extract only customer's speech
        customer_texts = [
            seg["text"] for seg in transcript_segments
            if seg["speaker"] == customer_label
        ]

        if not customer_texts:
            print("Warning: No customer text found")
            return {
                "sentiment_stability_score": 0.0,
                "emotional_volatility_score_nlp": 0.0,
                "financial_literacy_score": 0.0,
                "planning_orientation_score": 0.0
            }

        full_text = " ".join(customer_texts)

        # Sentiment analysis
        sentiment = self.sentiment_pipeline(full_text)[0]
        sentiment_score = sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"]

        # Emotion analysis
        emotion_scores = self.emotion_pipeline(full_text)[0]
        emotion_probs = {e["label"]: e["score"] for e in emotion_scores}

        # Calculate emotional volatility (variance across emotion probabilities)
        emotion_variance = float(np.std(list(emotion_probs.values())))

        # Financial literacy heuristic (domain-specific term matching)
        financial_terms = ["interest", "emi", "apr", "collateral", "repayment", "principal", "rate", "term"]
        financial_score = sum(term in full_text.lower() for term in financial_terms) / len(financial_terms)

        # Planning orientation heuristic
        planning_terms = ["future", "plan", "next year", "long term", "budget", "save", "investment"]
        planning_score = sum(term in full_text.lower() for term in planning_terms) / len(planning_terms)

        return {
            "sentiment_stability_score": sentiment_score,
            "emotional_volatility_score_nlp": emotion_variance,
            "financial_literacy_score": financial_score,
            "planning_orientation_score": planning_score
        }

    # ==========================================================
    # 5️⃣ Embedding-based Consistency
    # ==========================================================

    def consistency_score(self, transcript_segments, customer_label):
        """
        Calculate consistency of customer's responses using sentence embeddings
        
        Args:
            transcript_segments: List of transcribed segments
            customer_label: Speaker label of customer
            
        Returns:
            Consistency score between 0 and 1
        """
        answers = [
            seg["text"] for seg in transcript_segments
            if seg["speaker"] == customer_label
        ]

        if len(answers) < 2:
            return 1.0  # Perfect consistency if only one response

        # Generate embeddings
        embeddings = self.embedding_model.encode(answers, convert_to_tensor=True)
        
        # Calculate cosine similarity matrix
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        
        # Remove self-similarities (diagonal) and average
        mask = torch.ones_like(sim_matrix) - torch.eye(len(answers), device=sim_matrix.device)
        avg_similarity = torch.sum(sim_matrix * mask) / (len(answers) * (len(answers) - 1))

        return float(avg_similarity.item())

    # ==========================================================
    # 6️⃣ Main Pipeline
    # ==========================================================

    def analyze(self, audio_path, transcript_segments):
        """
        Main analysis pipeline
        
        Args:
            audio_path: Path to audio file
            transcript_segments: List of transcribed segments with text
            
        Returns:
            Dictionary of all behavioral scores
        """
        print("=" * 50)
        print("Starting behavioral analysis...")
        print("=" * 50)
        
        # Step 1: Diarization
        print("\n1️⃣ Running speaker diarization...")
        diarized_segments = self.diarize_audio(audio_path)
        print(f"   Found {len(diarized_segments)} speech segments")

        # Step 2: Attach speaker labels to transcript
        print("\n2️⃣ Aligning transcripts with speakers...")
        for i, seg in enumerate(transcript_segments):
            if i < len(diarized_segments):
                seg["speaker"] = diarized_segments[i]["speaker"]
            else:
                print(f"   Warning: More transcripts ({len(transcript_segments)}) than diarized segments ({len(diarized_segments)})")
                seg["speaker"] = "UNKNOWN"

        # Step 3: Identify roles
        print("\n3️⃣ Identifying roles...")
        banker_label = self.identify_customer(transcript_segments)
        # Customer is the other speaker
        all_speakers = set(seg["speaker"] for seg in diarized_segments)
        customer_label = [s for s in all_speakers if s != banker_label][0] if len(all_speakers) > 1 else banker_label
        print(f"   Banker: {banker_label}, Customer: {customer_label}")

        # Step 4: Acoustic analysis
        print("\n4️⃣ Extracting acoustic features...")
        stress_score, volatility_audio = self.extract_acoustic_features(
            audio_path, diarized_segments, customer_label
        )
        print(f"   Stress score: {stress_score:.4f}")
        print(f"   Audio volatility: {volatility_audio:.4f}")

        # Step 5: NLP analysis
        print("\n5️⃣ Running NLP analysis...")
        nlp_scores = self.analyze_text(transcript_segments, customer_label)
        print(f"   Sentiment: {nlp_scores['sentiment_stability_score']:.4f}")
        print(f"   Financial literacy: {nlp_scores['financial_literacy_score']:.4f}")

        # Step 6: Consistency analysis
        print("\n6️⃣ Calculating consistency...")
        consistency = self.consistency_score(transcript_segments, customer_label)
        print(f"   Consistency score: {consistency:.4f}")

        # Step 7: Compile final results
        print("\n7️⃣ Compiling final scores...")
        
        # Check for early repayment mention
        all_text = " ".join([seg["text"] for seg in transcript_segments if seg["speaker"] == customer_label])
        early_repayment_flag = 1 if "early repayment" in all_text.lower() else 0

        result = {
            "confidence_level_score": 1 - min(stress_score, 1.0),  # Normalize to [0,1]
            "stress_level_score": min(stress_score, 1.0),
            "desperation_index": max(0, -nlp_scores["sentiment_stability_score"]),
            "emotional_volatility_score": (volatility_audio + nlp_scores["emotional_volatility_score_nlp"]) / 2,
            "decision_consistency_score": consistency,
            "financial_literacy_score": nlp_scores["financial_literacy_score"],
            "clarity_of_purpose_score": consistency,
            "risk_awareness_score": nlp_scores["financial_literacy_score"],
            "planning_orientation_score": nlp_scores["planning_orientation_score"],
            "impulsivity_indicator": min(stress_score, 1.0),
            "honesty_consistency_score": consistency,
            "negotiation_behavior_score": 0.5,  # Placeholder - requires more sophisticated analysis
            "early_repayment_interest_flag": early_repayment_flag,
            "long_term_commitment_signal": nlp_scores["planning_orientation_score"],
            "evasiveness_score": 1 - consistency,
            "sentiment_stability_score": nlp_scores["sentiment_stability_score"],
            "intent_risk_score": float(
                0.4 * min(stress_score, 1.0) +
                0.3 * (1 - consistency) +
                0.3 * (1 - nlp_scores["financial_literacy_score"])
            )
        }

        print("\n" + "=" * 50)
        print("Analysis complete!")
        print("=" * 50)
        
        return result


# ==========================================================
# 🚀 Example Usage
# ==========================================================

def load_transcript_from_file(transcript_path):
    """
    Helper function to load transcript from a file
    Format expected: each line: [start_time end_time] text
    Example: [0.0 2.5] Hello, how can I help you?
    """
    segments = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Parse timestamp and text
                parts = line.split(']', 1)
                if len(parts) == 2:
                    timestamp = parts[0].strip('[')
                    text = parts[1].strip()
                    start, end = map(float, timestamp.split())
                    segments.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })
    return segments


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hybrid_voice_risk_engine.py <audio_file> [transcript_file]")
        print("\nIf transcript_file is not provided, it will be loaded from audio_file.txt")
        sys.exit(1)

    audio_file = sys.argv[1]
    
    # Get transcript file path
    if len(sys.argv) >= 3:
        transcript_file = sys.argv[2]
    else:
        # Default to audio file name with .txt extension
        transcript_file = os.path.splitext(audio_file)[0] + ".txt"
    
    # Check if files exist
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        sys.exit(1)
    
    if not os.path.exists(transcript_file):
        print(f"Error: Transcript file '{transcript_file}' not found.")
        print("Please provide a transcript file with the format: [start end] text")
        sys.exit(1)

    # Load transcript
    print(f"Loading transcript from: {transcript_file}")
    transcript_segments = load_transcript_from_file(transcript_file)
    print(f"Loaded {len(transcript_segments)} transcript segments")

    # Initialize engine (provide your HuggingFace token)
    # Get token from environment variable or replace with your token
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    if not hf_token:
        print("\n⚠️  Warning: No HuggingFace token found.")
        print("Please set HUGGINGFACE_TOKEN environment variable or pass it to the constructor")
        print("Get your token from: https://huggingface.co/settings/tokens")
    
    engine = HybridVoiceRiskEngine(hf_auth_token=hf_token)

    print("\n🔍 Analyzing conversation...\n")
    results = engine.analyze(audio_file, transcript_segments)

    print("\n" + "=" * 50)
    print("📊 FINAL BEHAVIORAL SCORES")
    print("=" * 50)
    for k, v in results.items():
        print(f"{k:35} : {v:.4f}")
    print("=" * 50)