from src.pipeline.tagger import Tagger
from src.pipeline.translator import Translator
import pandas as pd


def main():

    tagger_model_path = '../models/tagger-04-350k.pt'
    translator_model_path = '../models/translator-04-vatican-no-finetuning.pt'
    tagger_tokenizer = 'bert-base-multilingual-cased'
    translator_tokenizer = 'Helsinki-NLP/opus-mt-itc-itc'

    tagger = Tagger(tagger_model_path, tagger_tokenizer)
    translator = Translator(translator_model_path, translator_tokenizer)

    input_file_path = '../files/datasets/altered_04_test_vatican_texts.csv'
    output_file_path = '../files/outputs/corrected_test_set-04-04-vatican-no-finetuning.csv'

    test_set = pd.read_csv(input_file_path)
    test_set = test_set.head(200)

    correct_texts = test_set['texts'].tolist()
    sentences = test_set['altered'].tolist()
    labels = test_set['labels'].tolist()

    batch_size = 8
    predictions = []
    for i in range(0, len(sentences), batch_size):
        predictions.extend(tagger.predict(sentences[i: i + batch_size]))

    translated_sentences = []
    for i in range(0, len(sentences), batch_size):
        translated_sentences.extend(translator.translate(sentences[i: i + batch_size], predictions[i: i + batch_size]))

    results = pd.DataFrame({'corrections': translated_sentences, 'correct_texts': correct_texts,
                            'predictions': predictions, 'labels': labels})
    results.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    main()