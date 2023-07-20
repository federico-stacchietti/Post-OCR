import csv
from src.io.data_utils import create_dataset
import pandas as pd
from multiprocessing import Pool
from datasets import load_dataset
import nltk


def get_raw_texts(n_lines, output_path):
    dataset = load_dataset('mc4', languages=['it'], split='train', streaming=True)

    dataset_head = dataset.take(n_lines)

    counter = 0
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')

        writer.writerow(['texts'])

        for text in [row['text'] for row in dataset_head]:
            sentences = nltk.tokenize.sent_tokenize(text)
            for sentence in sentences:
                counter += 1
                writer.writerow([sentence])

    print(f'Wrote {counter} lines to file {output_path}')


def generate_dataset(raw_data_path, dataset_path, n_lines, probability, probabilities):
    data = pd.read_csv(raw_data_path)
    if n_lines == 0:
        data = data['texts'].tolist()
    else:
        data = data.head(n_lines)['texts'].tolist()

    sentences = []
    for text in data:
        if n_lines == 0:
            sentences += nltk.sent_tokenize(text)
        else:
            for sentence in nltk.sent_tokenize(text):
                if len(sentences) < n_lines:
                    sentences.append(sentence)
                else:
                    break

            if len(sentences) >= n_lines:
                break

    if len(sentences) % 2 != 0:
        del sentences[-1]

    n_processes = 10
    chunk_size = int(len(sentences) / n_processes)
    chunks = [sentences[i * chunk_size: (i + 1) * chunk_size] for i in range(n_processes)]
    with Pool(n_processes) as pool:
        results = pool.starmap(create_dataset, [(chunk, probability, probabilities) for chunk in chunks])

    results = [row for chunk in results for row in chunk]

    dataset = pd.DataFrame(data=results, columns=['texts', 'altered', 'labels'])
    if len(sentences) < 1000:
        dataset_path = dataset_path + f'{len(sentences)}.csv'
    else:
        dataset_path = dataset_path + f'{int(len(sentences) / 1000)}k.csv'
    dataset.to_csv(dataset_path, index=False, escapechar='\\')


def main(command):

    if command == 'get_raw_texts':
        # To download italian texts from the mc4 dataset. A specified number of lines will be downloaded
        get_raw_texts(n_lines=100_000, output_path='../files/downloads/raw_text.csv')
    else:
        dataset_name = 'test_vatican_texts'
        raw_data_path = '../files/downloads/' + dataset_name + '.csv'
        # A subset n_lines long of the dataset to be picked instead of the whole dataset
        n_lines = 0
        # The proportion of words that will be altered in a given piece of text
        probability = 0.4
        dataset_path = '../files/datasets/altered_' + f'0{int(probability * 10)}' + '_' + dataset_name + '_'

        # Probabilities that represent how much every form of errors will be present (misspelling, segmentation, etc).
        # Example: a list [0.5, 0.25, 0.25] means that half of the altered words will be misspelled, 25% wrongly
        # separated, 25% wrongly joined.
        # A dataset consisting of the original text, the altered text and the labels for every form of error will be
        # generated and saved to a .csv file
        probabilities = [0.5, 0.25, 0.25]
        generate_dataset(raw_data_path, dataset_path, n_lines, probability, probabilities)


if __name__ == '__main__':
    commands = ['get_raw_texts', 'generate_dataset']
    main(commands[0])
