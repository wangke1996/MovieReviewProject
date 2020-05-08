from models.sentiment_rate import UnionModel, TripleRepresentationV2
from backend.analyzer import dataAnalyzer


class Recommender:
    def __init__(self):
        self.model_name = 'triplev2_ens'
        self.model = UnionModel('%s-0' % self.model_name, models=(TripleRepresentationV2,), overwrite=False)

    @staticmethod
    def detail2triple(details):
        triples = []
        for target, item in details.items():
            for sentiment, val in item.items():
                for description, sentences in val.items():
                    triples.extend([(target, description, sentiment) for _ in range(len(sentences))])
        return triples

    def recommend(self, user_id, _type='user', candidate_num=100, recommend_num=10):
        profile_details, _ = dataAnalyzer.analyze_profile(user_id, _type)
        triples = self.detail2triple(profile_details)
        return self.model.recommend_movies(None, triples, candidate_num=candidate_num, recommend_num=recommend_num)


recommender = Recommender()
