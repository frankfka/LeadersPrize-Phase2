# Paths
from core.pipeline import LeadersPrizePipeline

ROOT = '/usr/local/'
ROOT = '/Users/frankjia/Desktop/LeadersPrize/submission_test/'
METADATA_FILEPATH = ROOT + 'dataset/metadata.json'
PREDICTIONS_FILEPATH = ROOT + 'predictions.json'

PIPELINE_CONFIG = {
    LeadersPrizePipeline.CONFIG_W2V_PATH: "assets/word2vec/GoogleNewsVectors.bin.gz",
    LeadersPrizePipeline.CONFIG_API_KEY: "ff5fdad7-de1f-4a74-bfac-acd42538131f",
    LeadersPrizePipeline.CONFIG_ENDPOINT: "http://lpsa.wrw.org",
    LeadersPrizePipeline.CONFIG_DEBUG: True
}
