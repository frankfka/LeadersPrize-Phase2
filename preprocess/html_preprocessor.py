from newspaper import fulltext
from bs4 import BeautifulSoup
import re


class HTMLProcessResult:
    class HTMLAttributes:

        def __init__(self, title, description):
            # Future - put in things like links, # of scripts, etc.
            self.title = title
            self.description = description

    def __init__(self, text: str, html_atts: HTMLAttributes):
        self.text = text  # NLP-friendly article text, extracted from the body
        self.html_atts = html_atts


class HTMLProcessor:

    def process(self, html) -> HTMLProcessResult:
        soup = BeautifulSoup(html, "html.parser")
        html_atts = self.__get_html_attributes(soup)
        article_text = self.__get_clean_article_text(html, soup=soup, html_atts=html_atts)
        return HTMLProcessResult(
            text=article_text,
            html_atts=html_atts
        )

    def __get_html_attributes(self, soup: BeautifulSoup) -> HTMLProcessResult.HTMLAttributes:
        title = self.__get_html_title(soup)
        desc = self.__get_html_description(soup)
        return HTMLProcessResult.HTMLAttributes(
            title=title,
            description=desc
        )

    def __get_html_title(self, soup: BeautifulSoup) -> str:
        title = ""
        if soup.title:
            title = soup.title.get_text()
        return title.strip()

    def __get_html_description(self, soup: BeautifulSoup) -> str:
        desc = ""
        # Get all meta attributes
        meta_tags = soup.find_all("meta")
        # Filter for property attribute containing description
        meta_tags = list(
            filter(lambda meta: meta.get("property") and "description" in meta["property"].lower(), meta_tags))
        for tag in meta_tags:
            # Find content attribute, if found, return immediately
            if tag.get("content"):
                desc = tag.get("content")
                break
        return desc.strip()

    def __get_clean_article_text(self, html: str, soup: BeautifulSoup = None,
                                 html_atts: HTMLProcessResult.HTMLAttributes = None) -> str:
        # Try to use external library to get article text. If that doesn't work, use naive BS4 method
        article_text = self.__get_article_text_newspaper(html)
        if not article_text and soup:
            article_text = self.__get_article_text_bs4(soup)
        # Add title & description from html attributes
        if html_atts.description:
            article_text = f"{html_atts.description} ." + article_text
        if html_atts.title:
            article_text = f"{html_atts.title} ." + article_text
        return self.__clean_article_text(article_text)

    def __get_article_text_bs4(self, soup: BeautifulSoup) -> str:
        if soup.body and soup.body.get_text().strip():
            return soup.body.get_text().strip()
        return ""

    def __get_article_text_newspaper(self, html: str) -> str:
        try:
            return fulltext(html)
        except Exception:
            print("Could not process with newspaper")
        return ""

    def __clean_article_text(self, text: str) -> str:
        # Replace newlines with a separator
        text = re.sub(r"\n+", ' . ', text)
        # Replace excess whitespaces
        text = re.sub(r"\s\s+", ' ', text)
        return text
