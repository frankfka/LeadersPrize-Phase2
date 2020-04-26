from core.models import PipelineClaim


def get_text_b_for_reasoner(claim: PipelineClaim) -> str:
    """
    Currently just concatenates all the sentences from all the articles
    - Consider creating articles_for_reasoner and sentences_for_reasoner in here
    - Consider having a relevance cutoff
    """
    text_b = ""
    for article in claim.articles_for_reasoner:
        text_b += f" %${article.relevance}$% "
        for sent in article.sentences_for_reasoner:
            text_b += f" ${sent.relevance}$ " + sent.sentence + " $.$ "
        text_b += " %$$% "
    return text_b
