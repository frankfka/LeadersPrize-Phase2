# Paths
from core.pipeline import PipelineConfigKeys

ROOT = '/usr/local/'
# ROOT = '/Users/frankjia/Desktop/LeadersPrize/submission_test/'
METADATA_FILEPATH = ROOT + 'dataset/metadata.json'
PREDICTIONS_FILEPATH = ROOT + 'predictions.json'

PIPELINE_CONFIG = {
    PipelineConfigKeys.W2V_PATH: "assets/word2vec/GoogleNewsVectors.bin.gz",
    PipelineConfigKeys.API_KEY: "ff5fdad7-de1f-4a74-bfac-acd42538131f",
    PipelineConfigKeys.ENDPOINT: "http://lpsa.wrw.org",
    PipelineConfigKeys.DEBUG_MODE: False,
    PipelineConfigKeys.RETRIEVE_ARTICLES: True,
}
