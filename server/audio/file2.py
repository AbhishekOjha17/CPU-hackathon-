import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
from collections import defaultdict
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from scipy.stats import zscore
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


class HybridVoiceRiskEngine:
    def __init__(self, device=None):
        """
        Initialize the Hybrid Voice Risk Engine
        Completely offline, minimal dependencies
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\nRunning on: {self.device.upper()}\n")
        
        # -------------------------
        # Load Models (All offline)
        # -------------------------
        self._load_models()
        
    def _load_models(self):
        """Load all required models"""
        
        # Create cache directory
        os.makedirs("./models", exist_ok=True)
        
        print("Loading emotion model...")
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True,
            model_kwargs={"cache_dir": "./models"}
        )

        print("Loading sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            device=0 if self.device == "cuda" else -1,
            model_kwargs={"cache_dir": "./models"}
        )

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", 
            device=self.device,
            cache_folder="./models"
        )
        
        print("All models loaded successfully.\n")

    # ==========================================================
    # 1️⃣ Simple Voice Activity Detection
    # ==========================================================
    
    def simple_vad(self, audio_path, energy_threshold_percentile=20, min_segment_duration=1.0):
        """
        Simple Voice Activity Detection based on energy
        
        Args:
            audio_path: Path to audio file
            energy_threshold_percentile: Percentile for energy threshold (lower = more sensitive)
            min_segment_duration: Minimum segment duration in seconds
            
        Returns:
            List of speech segments with start and end times
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Calculate energy in frames
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate RMS energy
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Dynamic threshold based on percentile
        energy_threshold = np.percentile(energy, energy_threshold_percentile)
        
        # Voice activity detection
        voice_frames = energy > energy_threshold
        
        # Convert frames to time
        times = librosa.frames_to_time(np.arange(len(voice_frames)), sr=sr, hop_length=hop_length)
        
        # Merge consecutive voice frames into segments
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_speech:
                in_speech = True
                start_time = times[i]
            elif not is_voice and in_speech:
                in_speech = False
                end_time = times[i]
                
                if end_time - start_time >= min_segment_duration:
                    segments.append({
                        "start": start_time,
                        "end": end_time
                    })
        
        # Handle last segment
        if in_speech:
            end_time = times[-1]
            if end_time - start_time >= min_segment_duration:
                segments.append({
                    "start": start_time,
                    "end": end_time
                })
        
        return segments

    # ==========================================================
    # 2️⃣ Extract Voice Characteristics for Clustering
    # ==========================================================
    
    def extract_voice_features(self, audio_path, segment):
        """
        Extract voice characteristics from a segment for speaker clustering
        
        Args:
            audio_path: Path to audio file
            segment: Dictionary with start and end times
            
        Returns:
            Feature vector for the segment
        """
        y, sr = librosa.load(audio_path, sr=16000)
        
        start_sample = int(segment["start"] * sr)
        end_sample = int(segment["end"] * sr)
        segment_audio = y[start_sample:end_sample]
        
        if len(segment_audio) < sr * 0.5:
            return None
        
        features = []
        
        # MFCC features (voice characteristics)
        mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            features.append(np.mean(pitch_values))
            features.append(np.std(pitch_values))
        else:
            features.extend([0, 0])
        
        # Energy features
        rms = librosa.feature.rms(y=segment_audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Zero crossing rate (voice quality)
        zcr = librosa.feature.zero_crossing_rate(segment_audio)
        features.append(np.mean(zcr))
        
        return np.array(features)

    # ==========================================================
    # 3️⃣ Speaker Clustering
    # ==========================================================
    
    def cluster_speakers(self, features_list, num_speakers=2):
        """
        Cluster segments into speakers using hierarchical clustering
        
        Args:
            features_list: List of feature vectors
            num_speakers: Expected number of speakers
            
        Returns:
            List of cluster labels
        """
        if len(features_list) < num_speakers:
            return [f"SPEAKER_{i % num_speakers:02d}" for i in range(len(features_list))]
        
        # Remove None values
        valid_features = [f for f in features_list if f is not None]
        valid_indices = [i for i, f in enumerate(features_list) if f is not None]
        
        if len(valid_features) < 2:
            return [f"SPEAKER_{i % num_speakers:02d}" for i in range(len(features_list))]
        
        # Stack features
        X = np.stack(valid_features)
        
        # Hierarchical clustering
        distance_matrix = pdist(X, metric='cosine')
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Get cluster labels
        labels = fcluster(linkage_matrix, num_speakers, criterion='maxclust')
        
        # Map back to original indices
        result = ["UNKNOWN"] * len(features_list)
        for idx, label in zip(valid_indices, labels):
            result[idx] = f"SPEAKER_{label-1:02d}"
        
        return result

    # ==========================================================
    # 4️⃣ Complete Diarization Pipeline
    # ==========================================================
    
    def diarize_audio(self, audio_path, num_speakers=2):
        """
        Complete speaker diarization pipeline
        
        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers
            
        Returns:
            List of segments with start, end, and speaker label
        """
        print(f"   Performing VAD and speaker clustering...")
        
        # Step 1: Voice Activity Detection
        segments = self.simple_vad(audio_path)
        print(f"   Found {len(segments)} speech segments")
        
        if len(segments) == 0:
            print("   Warning: No speech segments found")
            return []
        
        # Step 2: Extract features for each segment
        features_list = []
        for seg in segments:
            features = self.extract_voice_features(audio_path, seg)
            features_list.append(features)
        
        # Step 3: Cluster into speakers
        speaker_labels = self.cluster_speakers(features_list, num_speakers)
        
        # Step 4: Attach labels to segments
        for i, seg in enumerate(segments):
            seg["speaker"] = speaker_labels[i]
        
        return segments

    # ==========================================================
    # 5️⃣ Identify Banker vs Customer
    # ==========================================================

    def identify_customer(self, transcript_segments):
        """
        Identify which speaker is the customer based on question patterns
        """
        speaker_stats = defaultdict(lambda: {"questions": 0, "words": 0})

        for seg in transcript_segments:
            if "speaker" not in seg or seg["speaker"] == "UNKNOWN":
                continue
                
            text = seg["text"]
            speaker = seg["speaker"]

            speaker_stats[speaker]["words"] += len(text.split())
            speaker_stats[speaker]["questions"] += text.count("?")

        if not speaker_stats:
            return "SPEAKER_00"
            
        # Banker typically asks more questions
        banker = max(speaker_stats.items(), key=lambda x: x[1]["questions"])[0]
        
        print(f"   Identified banker: {banker}")
        
        return banker

    # ==========================================================
    # 6️⃣ Align Transcripts with Diarization
    # ==========================================================
    
    def align_transcripts(self, transcript_segments, diarized_segments):
        """
        Align transcript segments with diarized segments based on timestamps
        """
        aligned_transcripts = []
        
        for trans_seg in transcript_segments:
            # Find best matching diarized segment
            best_match = None
            max_overlap = 0
            
            for dia_seg in diarized_segments:
                # Calculate overlap
                overlap_start = max(trans_seg["start"], dia_seg["start"])
                overlap_end = min(trans_seg["end"], dia_seg["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = dia_seg
            
            new_seg = trans_seg.copy()
            if best_match and max_overlap > 0:
                new_seg["speaker"] = best_match["speaker"]
            else:
                new_seg["speaker"] = "UNKNOWN"
            
            aligned_transcripts.append(new_seg)
        
        return aligned_transcripts

    # ==========================================================
    # 7️⃣ Extract Acoustic Features
    # ==========================================================

    def extract_acoustic_features(self, audio_path, segments, customer_label):
        """
        Extract acoustic features (stress, pitch volatility) for customer segments
        """
        y, sr = librosa.load(audio_path, sr=None)

        stress_scores = []
        pitch_vars = []

        for seg in segments:
            if seg.get("speaker") != customer_label:
                continue

            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) < sr * 0.5:
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
    # 8️⃣ NLP Analysis
    # ==========================================================

    def analyze_text(self, transcript_segments, customer_label):
        """
        Perform NLP analysis on customer's speech
        """
        # Extract only customer's speech
        customer_texts = [
            seg["text"] for seg in transcript_segments
            if seg.get("speaker") == customer_label
        ]

        if not customer_texts:
            print("   Warning: No customer text found")
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

        # Calculate emotional volatility
        emotion_variance = float(np.std(list(emotion_probs.values())))

        # Financial literacy heuristic
        financial_terms = ["interest", "emi", "apr", "collateral", "repayment", "principal", "rate", "term", "loan", "credit"]
        financial_score = sum(term in full_text.lower() for term in financial_terms) / len(financial_terms)

        # Planning orientation heuristic
        planning_terms = ["future", "plan", "next year", "long term", "budget", "save", "investment", "retirement", "goal"]
        planning_score = sum(term in full_text.lower() for term in planning_terms) / len(planning_terms)

        return {
            "sentiment_stability_score": sentiment_score,
            "emotional_volatility_score_nlp": emotion_variance,
            "financial_literacy_score": financial_score,
            "planning_orientation_score": planning_score
        }

    # ==========================================================
    # 9️⃣ Embedding-based Consistency
    # ==========================================================

    def consistency_score(self, transcript_segments, customer_label):
        """
        Calculate consistency of customer's responses using sentence embeddings
        """
        answers = [
            seg["text"] for seg in transcript_segments
            if seg.get("speaker") == customer_label
        ]

        if len(answers) < 2:
            return 1.0

        # Generate embeddings
        embeddings = self.embedding_model.encode(answers, convert_to_tensor=True)
        
        # Calculate cosine similarity matrix
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        
        # Remove self-similarities and average
        mask = torch.ones_like(sim_matrix) - torch.eye(len(answers), device=sim_matrix.device)
        avg_similarity = torch.sum(sim_matrix * mask) / (len(answers) * (len(answers) - 1))

        return float(avg_similarity.item())

    # ==========================================================
    # 🔟 Main Pipeline
    # ==========================================================

    def analyze(self, audio_path, transcript_segments):
        """
        Main analysis pipeline
        """
        print("=" * 50)
        print("Starting behavioral analysis...")
        print("=" * 50)
        
        # Step 1: Diarization
        print("\n1️⃣ Running speaker diarization...")
        diarized_segments = self.diarize_audio(audio_path)
        print(f"   Found {len(diarized_segments)} speech segments")

        if len(diarized_segments) == 0:
            print("   Error: No speech detected in audio")
            return {}

        # Step 2: Align transcripts with speakers
        print("\n2️⃣ Aligning transcripts with speakers...")
        aligned_transcripts = self.align_transcripts(transcript_segments, diarized_segments)
        
        # Count speakers
        speakers = set(seg["speaker"] for seg in aligned_transcripts if seg["speaker"] != "UNKNOWN")
        print(f"   Detected {len(speakers)} speakers: {', '.join(speakers)}")

        # Step 3: Identify roles
        print("\n3️⃣ Identifying roles...")
        banker_label = self.identify_customer(aligned_transcripts)
        
        # Customer is the other speaker
        all_speakers = [s for s in speakers if s != "UNKNOWN"]
        if len(all_speakers) > 1:
            customer_label = [s for s in all_speakers if s != banker_label][0]
        else:
            customer_label = banker_label
            
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
        nlp_scores = self.analyze_text(aligned_transcripts, customer_label)
        print(f"   Sentiment: {nlp_scores['sentiment_stability_score']:.4f}")
        print(f"   Financial literacy: {nlp_scores['financial_literacy_score']:.4f}")

        # Step 6: Consistency analysis
        print("\n6️⃣ Calculating consistency...")
        consistency = self.consistency_score(aligned_transcripts, customer_label)
        print(f"   Consistency score: {consistency:.4f}")

        # Step 7: Compile final results
        print("\n7️⃣ Compiling final scores...")
        
        # Check for early repayment mention
        all_text = " ".join([seg["text"] for seg in aligned_transcripts if seg.get("speaker") == customer_label])
        early_repayment_flag = 1 if "early repayment" in all_text.lower() else 0

        result = {
            "confidence_level_score": 1 - min(max(stress_score, 0), 1.0),
            "stress_level_score": min(max(stress_score, 0), 1.0),
            "desperation_index": max(0, -nlp_scores["sentiment_stability_score"]),
            "emotional_volatility_score": (volatility_audio + nlp_scores["emotional_volatility_score_nlp"]) / 2,
            "decision_consistency_score": consistency,
            "financial_literacy_score": nlp_scores["financial_literacy_score"],
            "clarity_of_purpose_score": consistency,
            "risk_awareness_score": nlp_scores["financial_literacy_score"],
            "planning_orientation_score": nlp_scores["planning_orientation_score"],
            "impulsivity_indicator": min(max(stress_score, 0), 1.0),
            "honesty_consistency_score": consistency,
            "negotiation_behavior_score": 0.5,
            "early_repayment_interest_flag": early_repayment_flag,
            "long_term_commitment_signal": nlp_scores["planning_orientation_score"],
            "evasiveness_score": 1 - consistency,
            "sentiment_stability_score": nlp_scores["sentiment_stability_score"],
            "intent_risk_score": float(
                0.4 * min(max(stress_score, 0), 1.0) +
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
    """
    segments = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Try to parse timestamp format [start end]
                if line.startswith('[') and ']' in line:
                    parts = line.split(']', 1)
                    timestamp = parts[0].strip('[')
                    text = parts[1].strip()
                    try:
                        start, end = map(float, timestamp.split())
                        segments.append({
                            "start": start,
                            "end": end,
                            "text": text
                        })
                    except:
                        # If timestamp parsing fails, treat as plain text
                        segments.append({
                            "start": 0,
                            "end": 0,
                            "text": line
                        })
                else:
                    # Plain text without timestamps
                    segments.append({
                        "start": 0,
                        "end": 0,
                        "text": line
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

    # Initialize engine (completely offline, no external dependencies)
    engine = HybridVoiceRiskEngine()

    print("\n🔍 Analyzing conversation...\n")
    results = engine.analyze(audio_file, transcript_segments)

    if results:
        print("\n" + "=" * 50)
        print("📊 FINAL BEHAVIORAL SCORES")
        print("=" * 50)
        for k, v in results.items():
            print(f"{k:35} : {v:.4f}")
        print("=" * 50)