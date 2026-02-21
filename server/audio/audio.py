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
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------
        # Load Models
        # -------------------------

        print("Loading diarization model...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=True
        )

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

    # ==========================================================
    # 1️⃣ Speaker Diarization
    # ==========================================================

    def diarize_audio(self, audio_path):
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
        speaker_stats = defaultdict(lambda: {"questions": 0, "length": 0})

        for seg in transcript_segments:
            text = seg["text"]
            speaker = seg["speaker"]

            speaker_stats[speaker]["length"] += len(text.split())
            speaker_stats[speaker]["questions"] += text.count("?")

        # Banker typically asks more questions
        banker = max(speaker_stats.items(), key=lambda x: x[1]["questions"])[0]

        return banker

    # ==========================================================
    # 3️⃣ Extract Acoustic Features
    # ==========================================================

    def extract_acoustic_features(self, audio_path, segments, customer_label):
        y, sr = librosa.load(audio_path, sr=None)

        stress_scores = []
        pitch_vars = []

        for seg in segments:
            if seg["speaker"] != customer_label:
                continue

            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) < sr * 0.5:
                continue

            # Pitch
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]

            if len(pitch_values) > 0:
                pitch_vars.append(np.var(pitch_values))

            # Energy
            rms = librosa.feature.rms(y=segment_audio)
            energy = np.mean(rms)

            stress_scores.append(energy)

        stress_score = float(np.mean(zscore(stress_scores))) if len(stress_scores) > 1 else 0.0
        volatility_score = float(np.std(pitch_vars)) if pitch_vars else 0.0

        return stress_score, volatility_score

    # ==========================================================
    # 4️⃣ NLP Analysis
    # ==========================================================

    def analyze_text(self, transcript_segments, customer_label):
        customer_texts = [
            seg["text"] for seg in transcript_segments
            if seg["speaker"] == customer_label
        ]

        full_text = " ".join(customer_texts)

        # Sentiment
        sentiment = self.sentiment_pipeline(full_text)[0]
        sentiment_score = sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"]

        # Emotion
        emotion_scores = self.emotion_pipeline(full_text)[0]
        emotion_probs = {e["label"]: e["score"] for e in emotion_scores}

        # Emotional volatility
        emotion_variance = float(np.std(list(emotion_probs.values())))

        # Financial literacy heuristic
        financial_terms = ["interest", "emi", "apr", "collateral", "repayment"]
        financial_score = sum(term in full_text.lower() for term in financial_terms) / len(financial_terms)

        # Planning orientation heuristic
        planning_terms = ["future", "plan", "next year", "long term", "budget"]
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
        answers = [
            seg["text"] for seg in transcript_segments
            if seg["speaker"] == customer_label
        ]

        if len(answers) < 2:
            return 1.0

        embeddings = self.embedding_model.encode(answers, convert_to_tensor=True)
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

        return float(torch.mean(sim_matrix).item())

    # ==========================================================
    # 6️⃣ Main Pipeline
    # ==========================================================

    def analyze(self, audio_path, transcript_segments):
        diarized_segments = self.diarize_audio(audio_path)

        # Attach speaker labels to transcript
        for i in range(len(transcript_segments)):
            transcript_segments[i]["speaker"] = diarized_segments[i]["speaker"]

        banker_label = self.identify_customer(transcript_segments)
        customer_label = [s["speaker"] for s in diarized_segments if s["speaker"] != banker_label][0]

        stress_score, volatility_audio = self.extract_acoustic_features(
            audio_path, diarized_segments, customer_label
        )

        nlp_scores = self.analyze_text(transcript_segments, customer_label)
        consistency = self.consistency_score(transcript_segments, customer_label)

        result = {
            "confidence_level_score": 1 - stress_score,
            "stress_level_score": stress_score,
            "desperation_index": max(0, -nlp_scores["sentiment_stability_score"]),
            "emotional_volatility_score": (volatility_audio + nlp_scores["emotional_volatility_score_nlp"]) / 2,
            "decision_consistency_score": consistency,
            "financial_literacy_score": nlp_scores["financial_literacy_score"],
            "clarity_of_purpose_score": consistency,
            "risk_awareness_score": nlp_scores["financial_literacy_score"],
            "planning_orientation_score": nlp_scores["planning_orientation_score"],
            "impulsivity_indicator": stress_score,
            "honesty_consistency_score": consistency,
            "negotiation_behavior_score": 0.5,
            "early_repayment_interest_flag": 1 if "early repayment" in str(transcript_segments).lower() else 0,
            "long_term_commitment_signal": nlp_scores["planning_orientation_score"],
            "evasiveness_score": 1 - consistency,
            "sentiment_stability_score": nlp_scores["sentiment_stability_score"],
            "intent_risk_score": float(
                0.4 * stress_score +
                0.3 * (1 - consistency) +
                0.3 * (1 - nlp_scores["financial_literacy_score"])
            )
        }

        return result
    
    
    
    
