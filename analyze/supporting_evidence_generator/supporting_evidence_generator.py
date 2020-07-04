from core.models import PipelineClaim


class SupportingEvidenceGenerator:

    def __init__(self):
        pass

    def get_evidence(self, predicted_pipeline_claim: PipelineClaim) -> (str, dict):
        """
        Constructs supporting evidence:
        - First item in returned tuple is the explanation
        - Second item is a dict of supporting article URL's: { '1': 'some_url', '2': 'some_url' }
        """
        prediction = predicted_pipeline_claim.submission_label
        used_sentences = predicted_pipeline_claim.sentences_for_transformer
        # TODO: Make this work!
        supporting_urls = {}
        for sent in used_sentences:
            if len(supporting_urls) == 2:
                break
            if sent.parent_article_url not in supporting_urls.keys():
                supporting_urls[str(len(supporting_urls) + 1)] = sent.parent_article_url

        return "some explanation", supporting_urls
