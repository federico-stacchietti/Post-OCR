import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split


urls = set()
class_names = ['text parbase vaticanrichtext', 'text parbase container vaticanrichtext']
language = '/it/'
page_type = 'documents'
extension = '.pdf'


def get_documents_links(url):
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    for a_tag in soup.findAll('a'):
        href = a_tag.attrs.get('href')
        if href == '' or href is None:
            continue
        if 'index.html' in href:
            href = urljoin(url, href)
            parsed_href = urlparse(href)
            href = parsed_href.scheme + '://' + parsed_href.netloc + parsed_href.path
            urls.add(href)
    return urls


def download_html(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    return response.text


def extract_div_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    div = soup.find('div', class_=class_names[0])
    if not div:
        div = soup.find('div', class_=class_names[1])

    return div.text.strip() if div else None


def get_documents(n_documents):
    links, texts = set(), set()
    for url in urls:
        soup = BeautifulSoup(requests.get(url).content, 'html.parser')
        for a_tag in soup.findAll('a'):
            href = a_tag.attrs.get('href')
            if href == '' or href is None:
                continue
            if page_type in href and language in href and href[-4:] != extension:
                href = urljoin(url, href)
                parsed_href = urlparse(href)
                href = parsed_href.scheme + '://' + parsed_href.netloc + parsed_href.path
                html = download_html(href)
                links.add(href)
                texts.add(extract_div_content(html))
        if len(texts) > n_documents:
            break
    return list(links), list(texts)


def get_texts(start_url, n_documents):
    get_documents_links(start_url)
    links, documents = get_documents(n_documents)
    data = {'links': links, 'texts': documents}
    df = pd.DataFrame(data)
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv('downloads/train_vatican_texts.csv', index=False)
    test.to_csv('downloads/test_vatican_texts.csv', index=False)


def main():
    # Downloads and extracts the specified number of italian texts from the official vatican website
    # If the specified number of texts exceeds the number of texts present on the website, it just downloads the
    # available texts
    get_texts('https://www.vatican.va/content/vatican/it.html', 100)


if __name__ == '__main__':
    main()
