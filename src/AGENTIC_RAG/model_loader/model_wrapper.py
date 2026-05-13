import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MedCPTModelWrapper:
    def __init__(self, model_path):
        self.cross_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.cross_model.eval() 