import joblib

# Inspect the model file
model_path = "./model/model-v1.joblib"
data = joblib.load(model_path)
print(f"Type: {type(data)}")
print(f"Keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")