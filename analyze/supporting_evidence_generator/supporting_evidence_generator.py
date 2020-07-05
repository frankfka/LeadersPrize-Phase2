from analyze.ne_reasoner.predict_ensemble import EnsembleClassifier
from analyze.ne_reasoner.reasoner_models import BeliefAnalysis
from core.models import PipelineClaim


class SupportingEvidenceGenerator:

    def __init__(self):
        self.ne_reasoner = EnsembleClassifier(
            ckpt_path="/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/ne-reasoner/ckpt",
            cvm_path="/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/ne-reasoner/bows_premise_space_cvm_utf8.csv",
            vocab_path="/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/ne-reasoner/ensemble_vocab.p",
            glove_emb_path="/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/ne-reasoner/glove.6B.50d.txt"
        )

    def __run_reasoner(self, claim: PipelineClaim):
        claim_str = claim.preprocessed_claim
        contexts = [sent.preprocessed_text for sent in claim.sentences_for_transformer]
        rel_sent_idxs, predictions = self.ne_reasoner.predict_statement_in_contexts(claim_str, contexts)

        print("\n\n")
        print(f"Claim: {claim_str} - Predicted {claim.submission_label}")

        for sent_idx, pred in zip(rel_sent_idxs, predictions):
            analyzed_result = BeliefAnalysis.from_dict(pred)
            analysis_str = f"T: {analyzed_result.true_magnitude} | N: {analyzed_result.neutral_magnitude} F: {analyzed_result.false_magnitude}"
            sentence = claim.sentences_for_transformer[sent_idx].text
            article_url = claim.sentences_for_transformer[sent_idx].parent_article_url
            print(article_url)
            print(f"({analysis_str}): {sentence}")
            print("----")


    def get_evidence(self, predicted_pipeline_claim: PipelineClaim) -> (str, dict):
        """
        Constructs supporting evidence:
        - First item in returned tuple is the explanation
        - Second item is a dict of supporting article URL's: { '1': 'some_url', '2': 'some_url' }
        """
        self.__run_reasoner(predicted_pipeline_claim)
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
