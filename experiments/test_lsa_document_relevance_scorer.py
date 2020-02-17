import csv
import os
import bs4
import lsa_document_relevance_scorer


def read_all(path: str):
    file_names = os.listdir(path)
    for file_name in file_names:
        with open(f'{path}/{file_name}', errors='ignore') as f:
            text = f.read()
            yield file_name, text


def strip_newlines(text: str) -> str:
    text = text.replace('\n', '')
    return text


def parse_html(html: str) -> str:
    soup = bs4.BeautifulSoup(html, 'html.parser')

    bad_parents = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        'style'

    ]

    texts = soup.find_all(text=True)
    texts = [text for text in texts if text.count(' ') > 0]
    texts = [text for text in texts if text.count(';') == 0]
    texts = [text for text in texts if len(text) > 5]
    texts = [text for text in texts
             if text.parent.name not in bad_parents]

    # remove extra whitespace
    texts = [' '.join(text.split()) for text in texts]

    combined_text = ' '.join(texts)
    return combined_text


if __name__ == '__main__':

    test_claims = ['I think Super Bowl does not symbolize progress at all.',
                   'Hand washing causes spread of viral infections ',
                   'When I was mayor of New York City, I encouraged adoptions. Adoptions went up 65 to 70 percent; '
                   'abortions went down 16 percent.',
                   'Hilary said that democrats are united for the change',
                   'Black people are more likely to receive death sentences in the US']

    all_data_names = []
    test_articles = []
    for data_names, html in read_all('lsa-based/datainput'):
        article = parse_html(html)
        test_articles.append(article)
        all_data_names.append(data_names)

    analyzer = lsa_document_relevance_scorer.LSADocumentRelevanceAnalyzer()

    test_sizes = [20, 60, 140]
    for test_size in test_sizes:
        articles_in_test = test_articles[:test_size]
        data_names = all_data_names[:test_size]

        relevance_dict = {}
        for claim in test_claims:
            relevances = analyzer.analyze(claim, articles_in_test)
            relevance_dict[claim] = relevances

        model_name = 'lsa'

        with open(f'{model_name}_{test_size}.csv', 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Claim"] + list(data_names)
            writer.writerow(header)
            for claim in test_claims:
                writer.writerow([claim] + list(relevance_dict[claim]))

        with open(f'{model_name}_{test_size}_ranks.csv', 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for claim in test_claims:
                relevances = list(relevance_dict[claim])

                ranked_doc_names = [(doc, relevance)
                                    for relevance, doc
                                    in sorted(zip(relevances, data_names))]
                ranked_doc_names.reverse()
                writer.writerow([claim] + ranked_doc_names)
        print("debug")
    # for test_claim in test_claims:
    #     results = analyzer.analyze(test_claim, test_articles)
    #     print(results)
    #     print('')
    print('')
