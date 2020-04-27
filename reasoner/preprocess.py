from typing import List

from core.models import PipelineClaim, PipelineSentence


def get_text_b_for_reasoner(claim: PipelineClaim) -> str:
    """
    Currently just concatenates all the sentences from all the articles, sorted by sentence relevance
    - Consider creating articles_for_reasoner and sentences_for_reasoner in here
    - Consider having a relevance cutoff
    """
    text_b = ""
    all_sents: List[PipelineSentence] = []
    for article in claim.articles_for_reasoner:
        # text_b += f" %${article.relevance}$% "
        for sent in article.sentences_for_reasoner:
            all_sents.append(sent)
        # text_b += " %$$% "
    all_sents.sort(key=lambda j: j.relevance, reverse=True)
    for s in all_sents:
        text_b += f" ${s.relevance}$ " + s.sentence + " $.$ "
    return text_b
