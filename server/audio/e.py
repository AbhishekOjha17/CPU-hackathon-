from audio import HybridVoiceRiskEngine


engine = HybridVoiceRiskEngine()

transcript_segments = [
    {"text": "What is the purpose of this loan?"},
    {"text": "I need it for expanding my small business."},
    {"text": "How do you plan to repay?"},
    {"text": "Through increased revenue and structured EMI payments."}
]

result = engine.analyze("conversation.wav", transcript_segments)

print(result)