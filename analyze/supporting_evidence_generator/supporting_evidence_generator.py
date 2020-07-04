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
        prediction = predicted_pipeline_claim.submission_label  # TODO: Use the prediction somehow
        used_sentences = predicted_pipeline_claim.sentences_for_transformer

        supporting_info = {}
        used_urls = set()
        # Construct supporting information from the sentences used for the transformer
        for sent in used_sentences:
            if len(supporting_info) == 2:
                break
            if sent.parent_article_url not in used_urls:
                new_evidence = {
                    "url": sent.parent_article_url,
                    "evidence": sent.text  # Use the original text
                }
                support_key = str(len(supporting_info) + 1)

                supporting_info[support_key] = new_evidence
                used_urls.add(sent.parent_article_url)

        final_explanation = ""
        supporting_urls = {}

        for support_key, evidence in supporting_info.items():
            evidence_text = evidence["evidence"]
            final_explanation += f"Article {support_key} states that: {evidence_text}. "
            supporting_urls[support_key] = evidence["url"]

        return final_explanation, supporting_urls
