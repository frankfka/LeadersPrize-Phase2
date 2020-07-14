from core.models import PipelineClaim


class SupportingEvidenceGenerator:

    def __init__(self):
        self.max_sent_chars = 400
        self.ideal_min_sent_chars = 100
        self.url_key = "url"
        self.evidence_key = "evidence"

    def get_evidence(self, predicted_pipeline_claim: PipelineClaim) -> (str, dict):
        """
        Constructs supporting evidence:
        - First item in returned tuple is the explanation
        - Second item is a dict of supporting article URL's: { '1': 'some_url', '2': 'some_url' }
        """
        prediction = predicted_pipeline_claim.submission_label  # TODO: Use the prediction somehow
        used_sentences = predicted_pipeline_claim.sentences_for_transformer

        # Ideal - high sentence length
        ideal_supporting_info = []
        ideal_used_urls = set()
        # Backup - no limit on sentence length
        backup_supporting_info = []
        backup_used_urls = set()
        for sent in used_sentences:
            # Done - no need to keep looping
            if len(ideal_supporting_info) == 2:
                break
            sent_text = sent.text  # Use original text
            sent_text_chars = len(sent_text)
            # Truncate to max length
            if sent_text_chars > self.max_sent_chars:
                sent_text = f"{sent_text[:self.max_sent_chars]}..."
            sent_info_dict = {
                self.url_key: sent.parent_article_url,
                self.evidence_key: sent_text
            }
            # Ideal case
            if sent_text_chars > self.ideal_min_sent_chars:
                ideal_supporting_info.append(sent_info_dict)
                ideal_used_urls.add(sent.parent_article_url)
            # Backup case
            elif len(backup_supporting_info) < 2:
                backup_supporting_info.append(sent_info_dict)
                backup_used_urls.add(sent.parent_article_url)

        # Construct the actual evidence array
        if len(ideal_supporting_info) == 2:
            final_evidence_arr = ideal_supporting_info
        else:
            final_evidence_arr = backup_supporting_info
        # Hard limit on length
        if len(final_evidence_arr) > 2:
            final_evidence_arr = final_evidence_arr[:2]

        # Construct the final evidence for submission
        final_explanation = ""
        supporting_urls = {}
        for (idx, evidence_obj) in enumerate(final_evidence_arr):
            evidence_key = str(idx + 1)
            final_explanation += f"Article {evidence_key} states that: {evidence_obj[self.evidence_key]}. "
            supporting_urls[evidence_key] = evidence_obj[self.url_key]

        # Hard limit on character count:
        if len(final_explanation) > 1000:
            final_explanation = final_explanation[:1000]
        return final_explanation, supporting_urls
