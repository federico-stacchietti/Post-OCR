from transformers import AutoTokenizer
import torch
from typing import List


class Translator:
    def __init__(self, model_name: str, tokenizer: str):
        self.model = torch.load(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def translate(self, texts: List[str], labels_list: List[List[int]]) -> List[str]:

        whitespace = ' '

        labels_masks = []
        input_texts = []

        texts = [list(filter(lambda item: item != '', text.split(whitespace))) for text in texts]

        for i, (text, labels) in enumerate(zip(texts, labels_list)):
            label_idx = 0
            while label_idx < len(labels) - 1:
                if labels[label_idx] == labels[label_idx + 1] == 2:
                    text[label_idx] += text[label_idx + 1]
                    labels[label_idx + 1] = -1
                label_idx += 1
            texts[i] = [text[j] for j in range(len(labels)) if labels[j] != -1]
            labels = [label for label in labels if label != -1]
            labels_masks.append([0 if label == 0 else 1 for label in labels])
            input_texts.append([texts[i][k] for k in range(len(texts[i])) if labels_masks[i][k] == 1])

        outputs = []
        for idx, input_text in enumerate(input_texts):
            if not input_text:
                input_text = texts[idx]
            inputs = self.tokenizer(input_text, return_tensors='pt', padding=True)
            outputs.append(self.model.generate(**inputs, max_length=5))

        decoded_outputs = [(self.tokenizer.decode(token, skip_special_tokens=True) for token in output) for output in
                           outputs]

        translations = []
        for i, label_mask in enumerate(labels_masks):
            for j, mask in enumerate(label_mask):
                if mask == 1:
                    texts[i][j] = next(decoded_outputs[i])
            translations.append(whitespace.join([token for token in texts[i]]))

        return translations
