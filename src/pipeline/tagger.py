from transformers import BertTokenizerFast
import torch
import re


class Tagger:
    def __init__(self, model_name, tokenizer):
        self.model = torch.load(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)

    def tag(self, texts):

        texts = [text.split(' ') for text in texts]

        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, is_split_into_words=True)

        len_inputs = []
        word_ids_list = []
        for i in range(len(inputs['input_ids'])):
            len_inputs.append(inputs.word_ids(batch_index=i).count(None) - 2)
            word_ids_list.append(inputs.word_ids(batch_index=i))

        logits = self.model(**inputs).logits
        predictions_list = logits.argmax(-1).tolist()

        for i in range(len(predictions_list)):
            predictions_list[i] = self.correct_prediction(inputs['input_ids'][i], predictions_list[i])[1:-1]

        decoded = []
        for input_idx in inputs['input_ids']:
            decoded.append([self.tokenizer.decode(tok) for tok in input_idx][1:-1])

        return predictions_list, len_inputs, word_ids_list

    def correct_prediction(self, input_ids, predicted_token_class):
        untokenized = [self.tokenizer.decode(ids) for ids in input_ids.tolist()]
        for i in range(len(untokenized)):
            if predicted_token_class[i] != 2 and re.search(r'(##)', untokenized[i]):
                if predicted_token_class[i - 1] == 2:
                    predicted_token_class[i] = 2
        return predicted_token_class

    def predict(self, texts):

        predictions_list, len_inputs, word_ids_list = self.tag(texts)

        labels_list = []
        for i, predictions in enumerate(predictions_list):
            word_ids = [word_id for word_id in word_ids_list[i] if word_id is not None]
            if len_inputs[i] > 0:
                predictions = predictions[:-len_inputs[i]]
            labels = []
            current_word_id = 0
            for j, word_id in enumerate(word_ids):
                if j == current_word_id == 0:
                    labels.append(predictions[j])
                if word_id > current_word_id:
                    current_word_id = word_id
                    labels.append(predictions[j])
            labels_list.append(labels)

        return labels_list
