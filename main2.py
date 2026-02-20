"""
Voice AI Risk Engine - Production Version with Enhanced Stress Detection
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
        
        # Extract enhanced audio features for stress analysis
        print("  🔊 Analyzing voice characteristics with enhanced stress detection...")
        audio_features = self._extract_enhanced_audio_features(audio, sr, customer_segments)
        
        # Calculate all parameters
        print("  📊 Calculating risk parameters...")
        params = self._calculate_all_parameters(
            audio_features, customer_texts, customer_id, banker_id, confidence,
            audio, sr, customer_segments  # Pass additional data for stress calculation
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
    
    def _extract_enhanced_audio_features(self, audio, sr, segments):
        """
        Extract comprehensive audio features for enhanced stress analysis
        Considers:
        - Pitch fluctuations
        - Speech gaps/irregularities
        - Submissive tone indicators
        - Fast pace
        - Voice tremor/jitter
        - Energy variability
        """
        features = {
            'pitch_var': [],
            'pitch_contour': [],  # Store pitch over time
            'pitch_jitter': [],    # Micro pitch variations
            'energy_mean': [],
            'energy_var': [],
            'speech_rate': [],
            'pause_density': [],
            'segment_durations': [],
            'pitch_at_ends': [],    # Pitch at end of segments
            'gaps': []               # Gaps between segments
        }
        
        if not segments:
            return features
        
        # First pass: collect basic features per segment
        for seg in segments:
            start = int(seg['start'] * sr)
            end = int(seg['end'] * sr)
            seg_audio = audio[start:end]
            
            if len(seg_audio) < sr * 0.5:  # Skip segments < 0.5s
                continue
            
            try:
                # Extract pitch contour
                pitches, magnitudes = librosa.piptrack(y=seg_audio, sr=sr)
                
                # Get pitch at each time frame
                pitch_contour = []
                for i in range(pitches.shape[1]):
                    index = magnitudes[:, i].argmax()
                    pitch = pitches[index, i]
                    if pitch > 0:
                        pitch_contour.append(pitch)
                
                if len(pitch_contour) > 0:
                    features['pitch_var'].append(float(np.var(pitch_contour)))
                    features['pitch_contour'].append(pitch_contour)
                    
                    # Calculate jitter (cycle-to-cycle pitch variation)
                    if len(pitch_contour) > 5:
                        diffs = np.abs(np.diff(pitch_contour))
                        jitter = np.mean(diffs) / (np.mean(pitch_contour) + 1e-8)
                        features['pitch_jitter'].append(float(jitter))
                    
                    # Pitch at end of segment (for rising intonation detection)
                    features['pitch_at_ends'].append(float(pitch_contour[-1]))
                
                # Energy features
                rms = librosa.feature.rms(y=seg_audio)[0]
                if len(rms) > 0:
                    features['energy_mean'].append(float(np.mean(rms)))
                    features['energy_var'].append(float(np.var(rms)))
                
                # Speech rate using zero-crossing rate
                zcr = librosa.feature.zero_crossing_rate(seg_audio)[0]
                if len(zcr) > 0:
                    features['speech_rate'].append(float(np.mean(zcr)))
                
                # Pause density (silence ratio within segment)
                if len(rms) > 0:
                    # Adaptive threshold - bottom 15% = silence
                    energy_thresh = float(np.percentile(rms, 15))
                    pauses = np.sum(rms < energy_thresh) / len(rms)
                    features['pause_density'].append(float(pauses))
                
                features['segment_durations'].append(seg['end'] - seg['start'])
                
            except Exception as e:
                print(f"      ⚠️ Feature extraction error: {e}")
                continue
        
        # Second pass: calculate gaps between segments
        if len(segments) > 1:
            sorted_segs = sorted(segments, key=lambda x: x['start'])
            for i in range(1, len(sorted_segs)):
                gap = sorted_segs[i]['start'] - sorted_segs[i-1]['end']
                if gap > 0.1:  # Only count significant gaps (>100ms)
                    features['gaps'].append(float(gap))
        
        # Ensure at least some values for all features
        for key in features:
            if not features[key] and key not in ['pitch_contour']:  # Skip pitch_contour which can be empty
                if key == 'gaps':
                    features[key] = [0.2]  # Default small gap
                else:
                    features[key] = [0.1]
        
        return features
    
    def _calculate_enhanced_stress_score(self, audio_features, segments, audio, sr):
        """
        Enhanced stress level calculation considering multiple factors:
        - Pitch fluctuations (35%)
        - Gap irregularity (20%)
        - Submissive tone (20%)
        - Fast pace (25%)
        """
        
        # 1. PITCH FLUCTUATION FACTOR (0-1) - 35% weight
        if audio_features.get('pitch_var') and len(audio_features['pitch_var']) > 0:
            pitch_var_mean = np.mean(audio_features['pitch_var'])
            # Normalize: typical pitch var 0-1000, map to 0-1
            pitch_factor = min(pitch_var_mean / 500, 1.0)
        else:
            pitch_factor = 0.3
        
        # 2. GAP/PAUSE IRREGULARITY FACTOR (0-1) - 20% weight
        if audio_features.get('gaps') and len(audio_features['gaps']) > 1:
            gaps = audio_features['gaps']
            gap_variance = np.var(gaps)
            gap_mean = np.mean(gaps)
            
            # Combine gap irregularity and length
            # High variance and long gaps = cognitive load = stress
            gap_factor = min((gap_variance / 0.5 + gap_mean / 2) / 2, 1.0)
        else:
            gap_factor = 0.2
        
        # 3. SUBMISSIVE TONE FACTOR (0-1) - 20% weight
        submissive_factor = 0.0
        
        # Check for rising intonation at segment ends (submissive/questioning tone)
        if (audio_features.get('pitch_contour') and 
            audio_features.get('pitch_at_ends') and 
            len(audio_features['pitch_contour']) > 0):
            
            rising_ends = 0
            for i, contour in enumerate(audio_features['pitch_contour'][:10]):  # Check first 10 segments
                if len(contour) > 5:
                    # Compare start and end pitch
                    start_pitch = np.mean(contour[:3])
                    end_pitch = np.mean(contour[-3:])
                    if end_pitch > start_pitch * 1.15:  # 15% rise = submissive/questioning
                        rising_ends += 1
            
            rising_ratio = rising_ends / max(len(audio_features['pitch_contour'][:10]), 1)
            
            # Lower energy = more submissive
            if audio_features.get('energy_mean') and len(audio_features['energy_mean']) > 0:
                energy_mean = np.mean(audio_features['energy_mean'])
                # Normalize: typical energy 0.01-0.1, lower = more submissive
                energy_factor = 1 - min(energy_mean * 10, 1.0)
            else:
                energy_factor = 0.3
            
            submissive_factor = (rising_ratio * 0.6 + energy_factor * 0.4)
        else:
            submissive_factor = 0.3
        
        # 4. SPEECH PACE FACTOR (0-1) - 25% weight
        if audio_features.get('speech_rate') and len(audio_features['speech_rate']) > 0:
            speech_rate_mean = np.mean(audio_features['speech_rate'])
            # Normalize: typical ZCR 0.02-0.08, map to 0-1
            pace_factor = min((speech_rate_mean - 0.02) / 0.06, 1.0) if speech_rate_mean > 0.02 else 0
        else:
            pace_factor = 0.3
        
        # Weighted combination
        weights = {
            'pitch_fluctuation': 0.35,
            'gap_irregularity': 0.20,
            'submissive_tone': 0.20,
            'fast_pace': 0.25
        }
        
        stress_score = (
            weights['pitch_fluctuation'] * pitch_factor +
            weights['gap_irregularity'] * gap_factor +
            weights['submissive_tone'] * submissive_factor +
            weights['fast_pace'] * pace_factor
        )
        
        return float(min(stress_score, 1.0))
    
    def _calculate_all_parameters(self, audio_features, customer_texts, customer_id, banker_id, confidence,
                                   audio=None, sr=None, customer_segments=None):
        """Calculate all 17 risk parameters with proper type conversion"""
        params = RiskParameters()
        params.detected_customer = customer_id
        params.detected_banker = banker_id
        params.customer_confidence = float(confidence)
        
        full_text = " ".join(customer_texts).lower()
        
        # AUDIO-BASED PARAMETERS
        
        # Enhanced stress level score (using new multi-factor calculation)
        if audio is not None and sr is not None and customer_segments is not None:
            params.stress_level_score = self._calculate_enhanced_stress_score(
                audio_features, customer_segments, audio, sr
            )
        else:
            # Fallback to old calculation if needed
            if audio_features['pitch_var']:
                params.stress_level_score = float(min(
                    np.mean(audio_features['pitch_var']) * 10 / (1 + np.mean(audio_features['pitch_var']) * 10),
                    1.0
                ))
        
        # Emotional volatility (still based on energy variance)
        if audio_features['energy_var']:
            params.emotional_volatility_score = float(min(
                np.std(audio_features['energy_var']) * 5,
                1.0
            ))
        
        # Impulsivity indicator (speech rate based)
        if audio_features['speech_rate']:
            params.impulsivity_indicator = float(min(
                np.mean(audio_features['speech_rate']) * 2,
                1.0
            ))
        
        # Desperation index (stress * inverse of pause density)
        if audio_features['pause_density']:
            params.desperation_index = float(min(
                params.stress_level_score * (1 - np.mean(audio_features['pause_density'])),
                1.0
            ))
        
        # FINANCIAL BEHAVIOR PARAMETERS
        
        # Financial literacy - improved with partial matching
        loan_terms = self._calculate_financial_score_partial(full_text, 'loan_terms')
        params.financial_literacy_score = float(loan_terms)
        
        # Risk awareness
        risk_terms = self._calculate_financial_score_partial(full_text, 'risk_terms')
        params.risk_awareness_score = float(risk_terms)
        
        # Planning orientation
        planning_terms = self._calculate_financial_score_partial(full_text, 'planning_terms')
        params.planning_orientation_score = float(planning_terms)
        
        # Negotiation behavior
        negotiation_terms = self._calculate_financial_score_partial(full_text, 'negotiation_terms')
        params.negotiation_behavior_score = float(negotiation_terms)
        
        # Long-term commitment
        commitment_terms = self._calculate_financial_score_partial(full_text, 'commitment_terms')
        params.long_term_commitment_signal = float(commitment_terms)
        
        # Early repayment
        early_terms = ['early repayment', 'prepayment', 'pay early']
        params.early_repayment_interest_flag = float(any(t in full_text for t in early_terms))
        
        # COMMUNICATION STYLE PARAMETERS
        
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
        
        # FINAL INTENT RISK SCORE
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
    
    def _calculate_financial_score_partial(self, text: str, category: str) -> float:
        """
        Improved financial score calculation with partial matching
        """
        terms = self.financial_terms.get(category, [])
        
        if not terms or not text:
            return 0.0
        
        score = 0
        words = text.split()
        
        for term in terms:
            term_lower = term.lower()
            
            # Check for exact match
            if term_lower in text:
                score += 1
            else:
                # Check for partial matches
                term_words = term_lower.split()
                if len(term_words) == 1:
                    # Single word term - check if similar word exists
                    for word in words:
                        if term_lower in word or word in term_lower:
                            score += 0.5
                            break
                else:
                    # Multi-word phrase - check if words appear near each other
                    phrase_found = False
                    for i in range(len(words) - len(term_words) + 1):
                        phrase = ' '.join(words[i:i+len(term_words)])
                        if term_lower in phrase:
                            score += 1
                            phrase_found = True
                            break
                    if not phrase_found:
                        # Check if individual words appear
                        matches = 0
                        for tw in term_words:
                            if any(tw in w for w in words):
                                matches += 1
                        if matches >= len(term_words) - 1:  # Almost all words match
                            score += 0.7
        
        # Normalize by number of terms
        normalized_score = min(score / len(terms), 1.0)
        
        return float(normalized_score)
    
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