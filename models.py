class LeadersPrizeClaim:
    class LeadersPrizeRelatedArticle:
        def __init__(self, dict_key, dict_val):
            self.filepath = dict_key
            self.url = dict_val

    def __init__(self, data):
        """
        Create claim from JSON data in metadata file
        """
        self.id = data.get("id", "")
        self.claim = data.get("claim", "")
        self.claimant = data.get("claimant", "")
        self.date = data.get("date", "")
        # Optional fields
        self.label = data.get("label", None)
        related_articles = []
        data_related_articles = data.get("related_articles", None)
        if data_related_articles:
            for k, v in data_related_articles.items():
                related_articles.append(LeadersPrizeClaim.LeadersPrizeRelatedArticle(k, v))
        self.related_articles = related_articles
