from newspaper import fulltext


class HTMLProcessResult:

    def __init__(self, text: str):
        self.text = text  # NLP-friendly article text, extracted from the body


class HTMLProcessor:

    def process(self, html) -> HTMLProcessResult:
        # TODO: include meta title and such from BS4
        article_text = self.__get_article_text_newspaper(html)
        article_text = ' '.join(article_text.split())
        return HTMLProcessResult(article_text)

    def __get_article_text_newspaper(self, html: str) -> str:
        try:
            return fulltext(html)
        except Exception:
            print("Could not process with newspaper")
        return ""
