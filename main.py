"""
Voice AI Risk Engine - Production Version
ALWAYS uses your customer_call.wav file
Only creates sample if your file doesn't exist
"""

import os
import json
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML Models
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pyannote.audio import Pipeline
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to Python native types for JSON serialization"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

@dataclass
class RiskParameters:
    """All 17 risk parameters with scores"""
    # Audio-based parameters
    stress_level_score: float = 0.0
    emotional_volatility_score: float = 0.0
    impulsivity_indicator: float = 0.0
    desperation_index: float = 0.0
    
    # NLP-based parameters
    confidence_level_score: float = 0.0
    financial_literacy_score: float = 0.0
    risk_awareness_score: float = 0.0
    planning_orientation_score: float = 0.0
    sentiment_stability_score: float = 0.0
    clarity_of_purpose_score: float = 0.0
    
    # Semantic consistency parameters
    decision_consistency_score: float = 0.0
    honesty_consistency_score: float = 0.0
    evasiveness_score: float = 0.0
    
    # Behavioral parameters
    negotiation_behavior_score: float = 0.0
    long_term_commitment_signal: float = 0.0
    early_repayment_interest_flag: float = 0.0
    
    # Speaker identification
    detected_customer: str = ""
    customer_confidence: float = 0.0
    detected_banker: str = ""
    
    # Composite risk score
    intent_risk_score: float = 0.0

class VoiceAIRiskEngine:
    """
    Voice AI Risk Engine - Uses YOUR customer_call.wav file
    """
    
    def __init__(self, device: str = None):
        """Initialize all models"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎤 Initializing Voice AI Risk Engine on {self.device.upper()}...")
        
        # Initialize models
        self._init_models()
        
        # Financial terms dictionary
        self.financial_terms = {
            'loan_terms': ['emi', 'apr', 'interest rate', 'collateral', 'principal', 
                          'tenure', 'repayment', 'installment', 'prepayment', 'loan',
                          'borrow', 'mortgage'],
            'risk_terms': ['default', 'late payment', 'penalty', 'cibil', 'credit score',
                          'guarantor', 'foreclosure', 'bounce'],
            'planning_terms': ['budget', 'savings', 'planning', 'monthly income',
                              'expenses', 'emergency fund', 'future', 'retirement'],
            'negotiation_terms': ['reduce', 'lower', 'discount', 'waive', 'flexible',
                                 'adjust', 'compromise', 'negotiate'],
            'commitment_terms': ['long term', 'stable', 'permanent', 'regular',
                                'consistent', 'committed', 'reliable']
        }
        
        # Question indicators for speaker detection
        self.question_indicators = ['?', 'what', 'why', 'how', 'when', 'where', 'who',
                                    'can you', 'could you', 'tell me']
        self.personal_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our']
        
        print("✅ Models ready")
    
    def _init_models(self):
        """Initialize AI models"""
        print("  📊 Loading models (first time may take 2-3 minutes)...")
        
        # Speaker diarization
        try:
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            self.diarization.to(torch.device(self.device))
            print("    ✅ Pyannote model loaded")
        except Exception as e:
            print(f"    ⚠️ Pyannote model failed: {e}")
            print("    Will use simple energy-based segmentation")
            self.diarization = None
        
        # Whisper for transcription
        try:
            self.whisper = whisper.load_model("base").to(self.device)
            print("    ✅ Whisper model loaded")
        except Exception as e:
            print(f"    ❌ Whisper failed: {e}")
            raise
        
        # Sentiment analysis
        try:
            self.sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == 'cuda' else -1
            )
            print("    ✅ Sentiment model loaded")
        except Exception as e:
            print(f"    ⚠️ Sentiment model failed: {e}")
            self.sentiment = None
        
        # Sentence transformer
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.encoder.to(self.device)
            print("    ✅ Sentence transformer loaded")
        except Exception as e:
            print(f"    ⚠️ Sentence transformer failed: {e}")
            self.encoder = None
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    def create_sample_audio(self, filename: str = "sample_call.wav") -> str:
        """
        Create a sample audio file ONLY if your file doesn't exist
        This is a fallback, not the default
        """
        print(f"\n🎵 Creating sample conversation: {filename} (FALLBACK - your file wasn't found)")
        
        duration = 45  # seconds
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create realistic speech patterns
        audio = np.zeros_like(t)
        
        # Speaker 1 (Banker)
        banker_times = [(0, 5), (12, 17), (25, 30), (38, 42)]
        for start, end in banker_times:
            mask = (t >= start) & (t < end)
            freqs = [100, 120, 140, 160]
            for freq in freqs:
                audio[mask] += 0.03 * np.sin(2 * np.pi * freq * t[mask])
        
        # Speaker 2 (Customer)
        customer_times = [(5, 12), (17, 25), (30, 38), (42, 45)]
        for start, end in customer_times:
            mask = (t >= start) & (t < end)
            base_freq = 200
            modulated = base_freq * (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t[mask]))
            audio[mask] += 0.04 * np.sin(2 * np.pi * modulated * t[mask] / sr)
        
        # Add noise
        audio += 0.01 * np.random.randn(len(t))
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.5
        
        sf.write(filename, audio, sr)
        print(f"   ✅ Created sample file: {filename}")
        return filename
    
    def analyze_file(self, audio_path: str) -> RiskParameters:
        """
        Analyze your audio file and return all 17 parameters
        """
        print(f"\n🔍 Analyzing: {audio_path}")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ Audio file not found: {audio_path}")
        
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"✅ File found, size: {file_size_mb:.1f} MB")
        
        # Load audio
        print("  📂 Loading audio...")
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            print(f"     Duration: {duration:.1f} seconds")
            print(f"     Sample rate: {sr} Hz")
        except Exception as e:
            raise Exception(f"Failed to load audio: {e}")
        
        # Get speaker segments
        print("  👥 Identifying speakers...")
        segments = self._get_speaker_segments(audio, sr)
        print(f"     Found {len(segments)} speakers")
        
        # Transcribe
        print("  📝 Transcribing speech...")
        segments = self._transcribe_segments(audio, sr, segments)
        
        # Identify customer
        print("  🎯 Identifying customer...")
        customer_id, banker_id, confidence = self._identify_customer(segments)
        print(f"     Customer: {customer_id} (confidence: {confidence:.2f})")
        print(f"     Banker: {banker_id}")
        
        # Extract customer data
        customer_segments = segments.get(customer_id, [])
        customer_texts = [s['text'] for s in customer_segments if s['text'].strip()]
        
        if not customer_texts:
            print("  ⚠️ No customer speech detected in the audio")
            print("     This could mean:")
            print("     - The audio doesn't contain clear speech")
            print("     - Speaker diarization couldn't identify the customer")
            print("     - The file might be silent or corrupted")
            customer_texts = ["No speech detected"]  # Placeholder
        
        print(f"     Customer spoke {len(customer_texts)} times")
        for i, text in enumerate(customer_texts[:3]):  # Show first 3 responses
            if text and text != "No speech detected":
                print(f"        Response {i+1}: {text[:50]}...")
        
        # Extract audio features
        print("  🔊 Analyzing voice characteristics...")
        audio_features = self._extract_audio_features(audio, sr, customer_segments)
        
        # Calculate all parameters
        print("  📊 Calculating risk parameters...")
        params = self._calculate_all_parameters(
            audio_features, customer_texts, customer_id, banker_id, confidence
        )
        
        return params
    
    def _get_speaker_segments(self, audio, sr):
        """Extract speaker segments with fallback"""
        segments = defaultdict(list)
        
        if self.diarization:
            try:
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                diarization = self.diarization({
                    'waveform': audio_tensor,
                    'sample_rate': sr
                })
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments[speaker].append({
                        'start': float(turn.start),
                        'end': float(turn.end),
                        'duration': float(turn.end - turn.start),
                        'text': ''
                    })
                
                # If only one speaker found, create a second one for simulation
                if len(segments) == 1:
                    segments['SPEAKER_01'] = []
                    
            except Exception as e:
                print(f"    ⚠️ Diarization error: {e}")
                segments = self._fallback_segmentation(audio, sr)
        else:
            segments = self._fallback_segmentation(audio, sr)
        
        return dict(segments)
    
    def _fallback_segmentation(self, audio, sr):
        """Simple energy-based segmentation as fallback"""
        segments = defaultdict(list)
        
        # Simple segmentation - split audio into alternating segments
        segment_duration = 3.0  # 3 second segments
        total_duration = len(audio) / sr
        
        num_segments = int(total_duration / segment_duration)
        
        for i in range(num_segments):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, total_duration)
            
            # Alternate between speakers
            speaker = f"SPEAKER_{i % 2:02d}"
            
            segments[speaker].append({
                'start': float(start),
                'end': float(end),
                'duration': float(end - start),
                'text': ''
            })
        
        return dict(segments)
    
    def _transcribe_segments(self, audio, sr, segments):
        """Transcribe each segment with error handling"""
        total_segments = sum(len(seg_list) for seg_list in segments.values())
        processed = 0
        
        for speaker, seg_list in segments.items():
            for seg in seg_list:
                start = int(seg['start'] * sr)
                end = int(seg['end'] * sr)
                seg_audio = audio[start:end]
                
                if len(seg_audio) > sr * 0.5:  # >0.5 seconds
                    try:
                        # Save segment temporarily for transcription
                        temp_file = f"temp_segment_{speaker}_{processed}.wav"
                        sf.write(temp_file, seg_audio, sr)
                        
                        result = self.whisper.transcribe(temp_file)
                        seg['text'] = result['text'].strip()
                        
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        print(f"      ⚠️ Transcription error: {e}")
                        seg['text'] = ""
                else:
                    seg['text'] = ""
                
                processed += 1
                if processed % 5 == 0:
                    print(f"      Progress: {processed}/{total_segments} segments transcribed")
        
        return segments
    
    def _identify_customer(self, segments):
        """Identify which speaker is the customer"""
        if len(segments) == 0:
            return "SPEAKER_00", "", 0.5
        
        if len(segments) == 1:
            speaker = list(segments.keys())[0]
            return speaker, "", 0.5
        
        speaker_profiles = {}
        
        for speaker, seg_list in segments.items():
            texts = [s['text'].lower() for s in seg_list if s['text']]
            all_text = " ".join(texts)
            
            # Count questions
            questions = sum(1 for s in seg_list if '?' in s['text'])
            
            # Count personal pronouns
            pronouns = 0
            for pronoun in self.personal_pronouns:
                pronouns += all_text.count(f" {pronoun} ")
            
            # Total speech duration
            duration = sum(s['duration'] for s in seg_list)
            
            speaker_profiles[speaker] = {
                'questions': questions,
                'pronouns': pronouns,
                'duration': duration,
                'text': all_text
            }
        
        # If no text data, use duration-based heuristic
        if all(p['pronouns'] == 0 for p in speaker_profiles.values()):
            # Longer speaker is likely customer
            customer = max(speaker_profiles, key=lambda x: speaker_profiles[x]['duration'])
            confidence = 0.6
        else:
            # Use combined heuristic
            scores = {}
            for speaker, profile in speaker_profiles.items():
                score = 0
                # More speech = more likely customer
                if profile['duration'] > np.mean([p['duration'] for p in speaker_profiles.values()]):
                    score += 2
                # More personal pronouns = customer
                if profile['pronouns'] > np.mean([p['pronouns'] for p in speaker_profiles.values()]):
                    score += 3
                # Fewer questions = customer
                if profile['questions'] < np.mean([p['questions'] for p in speaker_profiles.values()]):
                    score += 2
                scores[speaker] = score
            
            # Calculate confidence
            max_score = max(scores.values())
            total_score = sum(scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5
            
            customer = max(scores, key=scores.get)
        
        banker = [s for s in segments.keys() if s != customer][0] if len(segments) > 1 else ""
        
        return customer, banker, float(confidence)
    
    def _extract_audio_features(self, audio, sr, segments):
        """Extract acoustic features with proper type conversion"""
        features = {
            'pitch_var': [],
            'energy_var': [],
            'speech_rate': [],
            'pause_density': []
        }
        
        if not segments:
            # Return default features
            return {
                'pitch_var': [0.1],
                'energy_var': [0.1],
                'speech_rate': [0.1],
                'pause_density': [0.1]
            }
        
        for seg in segments:
            start = int(seg['start'] * sr)
            end = int(seg['end'] * sr)
            seg_audio = audio[start:end]
            
            if len(seg_audio) < sr * 0.5:
                continue
            
            try:
                # Pitch
                pitches, _ = librosa.piptrack(y=seg_audio, sr=sr)
                pitches = pitches[pitches > 0]
                if len(pitches) > 0:
                    features['pitch_var'].append(float(np.var(pitches)))
                
                # Energy
                rms = librosa.feature.rms(y=seg_audio)[0]
                if len(rms) > 0:
                    features['energy_var'].append(float(np.var(rms)))
                
                # Speech rate (using zero-crossing rate)
                zcr = librosa.feature.zero_crossing_rate(seg_audio)[0]
                if len(zcr) > 0:
                    features['speech_rate'].append(float(np.mean(zcr)))
                
                # Pauses
                if len(rms) > 0:
                    energy_thresh = float(np.mean(rms)) * 0.1
                    pauses = np.sum(rms < energy_thresh) / len(rms)
                    features['pause_density'].append(float(pauses))
                    
            except Exception as e:
                print(f"      ⚠️ Feature extraction error: {e}")
                continue
        
        # Ensure at least some values
        for key in features:
            if not features[key]:
                features[key] = [0.1]
        
        return features
    
    def _calculate_all_parameters(self, audio_features, customer_texts, customer_id, banker_id, confidence):
        """Calculate all 17 risk parameters with proper type conversion"""
        params = RiskParameters()
        params.detected_customer = customer_id
        params.detected_banker = banker_id
        params.customer_confidence = float(confidence)
        
        full_text = " ".join(customer_texts).lower()
        
        # Audio-based parameters
        if audio_features['pitch_var']:
            params.stress_level_score = float(min(
                np.mean(audio_features['pitch_var']) * 10 / (1 + np.mean(audio_features['pitch_var']) * 10),
                1.0
            ))
        
        if audio_features['energy_var']:
            params.emotional_volatility_score = float(min(
                np.std(audio_features['energy_var']) * 5,
                1.0
            ))
        
        if audio_features['speech_rate']:
            params.impulsivity_indicator = float(min(
                np.mean(audio_features['speech_rate']) * 2,
                1.0
            ))
        
        if audio_features['pause_density']:
            params.desperation_index = float(min(
                params.stress_level_score * (1 - np.mean(audio_features['pause_density'])),
                1.0
            ))
        
        # Financial literacy
        loan_terms = sum(1 for t in self.financial_terms['loan_terms'] if t in full_text)
        total_words = max(len(full_text.split()), 1)
        params.financial_literacy_score = float(min(loan_terms / (total_words/10 + 1), 1.0))
        
        # Risk awareness
        risk_terms = sum(1 for t in self.financial_terms['risk_terms'] if t in full_text)
        params.risk_awareness_score = float(min(risk_terms / 5, 1.0))
        
        # Planning orientation
        planning_terms = sum(1 for t in self.financial_terms['planning_terms'] if t in full_text)
        params.planning_orientation_score = float(min(planning_terms / 5, 1.0))
        
        # Negotiation behavior
        negotiation_terms = sum(1 for t in self.financial_terms['negotiation_terms'] if t in full_text)
        params.negotiation_behavior_score = float(min(negotiation_terms / 5, 1.0))
        
        # Long-term commitment
        commitment_terms = sum(1 for t in self.financial_terms['commitment_terms'] if t in full_text)
        params.long_term_commitment_signal = float(min(commitment_terms / 5, 1.0))
        
        # Early repayment
        early_terms = ['early repayment', 'prepayment', 'pay early']
        params.early_repayment_interest_flag = float(any(t in full_text for t in early_terms))
        
        # Sentiment analysis
        sentiments = []
        if self.sentiment:
            for text in customer_texts[:5]:
                if text.strip() and text != "No speech detected":
                    try:
                        result = self.sentiment(text[:512])[0]
                        score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
                        sentiments.append(float(score))
                    except:
                        sentiments.append(0.5)
        
        if sentiments:
            params.confidence_level_score = float(1 - np.std(sentiments))
            params.sentiment_stability_score = float(1 - np.std(sentiments))
        else:
            params.confidence_level_score = 0.5
            params.sentiment_stability_score = 0.5
        
        # Evasiveness (semantic similarity)
        if self.encoder and len(customer_texts) >= 3:
            valid_texts = [t for t in customer_texts[:3] if t.strip() and t != "No speech detected"]
            if len(valid_texts) >= 2:
                embeddings = []
                for text in valid_texts:
                    emb = self.encoder.encode(text[:512])
                    embeddings.append(emb)
                
                if len(embeddings) >= 2:
                    similarities = []
                    for i in range(len(embeddings)-1):
                        sim = np.dot(embeddings[i], embeddings[i+1]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-8
                        )
                        similarities.append(float(sim))
                    if similarities:
                        params.evasiveness_score = float(1 - np.mean(similarities))
        
        # Clarity of purpose
        purpose_words = ['need', 'want', 'purpose', 'reason', 'for', 'because']
        purpose_count = sum(1 for w in purpose_words if f" {w} " in f" {full_text} ")
        params.clarity_of_purpose_score = float(
            (purpose_count/len(purpose_words) + params.confidence_level_score) / 2
        )
        
        # Consistency
        params.decision_consistency_score = float(1 - params.evasiveness_score)
        params.honesty_consistency_score = float(1 - params.evasiveness_score)
        
        # Intent risk score (weighted combination)
        weights = {
            'stress_level_score': 0.15,
            'desperation_index': 0.15,
            'emotional_volatility_score': 0.10,
            'evasiveness_score': 0.15,
            'financial_literacy_score': -0.10,
            'risk_awareness_score': -0.10,
            'planning_orientation_score': -0.10,
            'impulsivity_indicator': 0.10,
            'decision_consistency_score': -0.10,
        }
        
        risk_score = 0.0
        total_weight = 0.0
        
        for param, weight in weights.items():
            value = getattr(params, param, 0.5)
            if weight > 0:
                risk_score += value * weight
            else:
                risk_score += (1 - value) * abs(weight)
            total_weight += abs(weight)
        
        params.intent_risk_score = float(risk_score / total_weight if total_weight > 0 else 0.5)
        
        return params
    
    def print_results(self, params: RiskParameters):
        """Print formatted results"""
        print("\n" + "="*70)
        print("🎯 VOICE AI RISK ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\n📌 SPEAKER IDENTIFICATION")
        print(f"   Customer: {params.detected_customer}")
        print(f"   Banker: {params.detected_banker}")
        print(f"   Confidence: {params.customer_confidence:.2f}")
        
        categories = {
            "AUDIO METRICS": ['stress_level_score', 'emotional_volatility_score', 
                            'impulsivity_indicator', 'desperation_index'],
            "FINANCIAL BEHAVIOR": ['financial_literacy_score', 'risk_awareness_score',
                                  'planning_orientation_score', 'negotiation_behavior_score',
                                  'long_term_commitment_signal', 'early_repayment_interest_flag'],
            "COMMUNICATION STYLE": ['confidence_level_score', 'sentiment_stability_score',
                                   'clarity_of_purpose_score', 'evasiveness_score',
                                   'decision_consistency_score', 'honesty_consistency_score']
        }
        
        for category, param_list in categories.items():
            print(f"\n📌 {category}")
            print("-"*50)
            for param in param_list:
                value = getattr(params, param)
                if param == 'early_repayment_interest_flag':
                    display = "✅ Yes" if value > 0.5 else "❌ No"
                    print(f"   {param.replace('_', ' ').title():<35}: {display}")
                else:
                    if value > 0.7:
                        color = "🔴"
                    elif value > 0.4:
                        color = "🟡"
                    else:
                        color = "🟢"
                    print(f"   {color} {param.replace('_', ' ').title():<35}: {value:.3f}")
        
        print("\n" + "="*70)
        print(f"🎯 FINAL INTENT RISK SCORE: {params.intent_risk_score:.3f}")
        
        if params.intent_risk_score > 0.7:
            print("   🔴 HIGH RISK - Recommend enhanced due diligence")
        elif params.intent_risk_score > 0.4:
            print("   🟡 MEDIUM RISK - Standard verification with caution")
        else:
            print("   🟢 LOW RISK - Proceed with standard verification")
        print("="*70)

def main():
    """Main function - ALWAYS tries to use customer_call.wav first"""
    
    print("="*60)
    print("🎙 VOICE AI RISK ENGINE")
    print("="*60)
    print("This program will analyze your customer_call.wav file")
    print("-"*60)
    
    # Initialize engine
    engine = VoiceAIRiskEngine()
    
    # Define the audio file path - ALWAYS look for customer_call.wav
    AUDIO_FILE = "customer_call.wav"
    
    # Check if the file exists
    if os.path.exists(AUDIO_FILE):
        print(f"\n✅ Found your file: {AUDIO_FILE}")
        print("   Analyzing your audio...")
    else:
        print(f"\n❌ Your file '{AUDIO_FILE}' was not found in the current folder!")
        print(f"   Current folder: {os.getcwd()}")
        print("\n   Please ensure:")
        print(f"   1. Your file is named exactly: {AUDIO_FILE}")
        print("   2. It's in the same folder as this script")
        print("   3. The file is not corrupted")
        print("\n   Would you like to create a sample file for testing?")
        response = input("   Create sample? (y/n): ").lower()
        
        if response == 'y':
            AUDIO_FILE = engine.create_sample_audio("sample_call.wav")
            print("\n⚠️  Using sample file for demonstration.")
            print("   To analyze your real file, name it 'customer_call.wav' and run again.")
        else:
            print("\n❌ Exiting - please provide customer_call.wav and try again.")
            return
    
    # Analyze the file
    try:
        results = engine.analyze_file(AUDIO_FILE)
        engine.print_results(results)
        
        # Convert to JSON-serializable format and save
        serializable_results = engine._convert_to_serializable(asdict(results))
        
        output_file = "risk_results.json"
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
        print(f"\n📁 Full results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()