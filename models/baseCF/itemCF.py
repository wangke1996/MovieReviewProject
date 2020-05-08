
from . import similarity
from models.lib import utils


class ItemCF:
    """
    Item-based Collaborative filtering.
    Top-N recommendation.
    """

    def __init__(self, model_dir: str, k_sim_movie=20, n_rec_movie=10, use_iuf_similarity=False, save_model=True):
        """
        Init UserBasedCF with n_sim_user and n_rec_movie.
        :return: None
        """
        print("ItemBasedCF start...\n")
        self.k_sim_movie = k_sim_movie
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.save_model = save_model
        self.use_iuf_similarity = use_iuf_similarity
        self.model_manager = utils.ModelManager(model_dir)
        self.item_average_score = None

    def calculate_item_average_score(self):
        item_average_score = {}
        item_freq = {}
        for items_scores in self.trainset.values():
            for item, score in items_scores.items():
                total_score = item_average_score.get(item, 0) + score
                freq = item_freq.get(item, 0) + 1
                item_freq[item] = freq
                item_average_score[item] = total_score
        for item, freq in item_freq.items():
            item_average_score[item] = item_average_score[item] / freq
        self.item_average_score = item_average_score

    def fit(self, trainset, retrain=True):
        """
        Fit the trainset by calculate movie similarity matrix.
        :param trainset: train dataset
        :return: None
        """
        model_manager = self.model_manager
        if not retrain:
            self.movie_sim_mat = model_manager.load_pkl(
                'movie_sim_mat-iif' if self.use_iuf_similarity else 'movie_sim_mat')
            self.movie_popular = model_manager.load_pkl('movie_popular')
            self.movie_count = model_manager.load_pkl('movie_count')
            self.trainset = model_manager.load_pkl('trainset')
            self.item_average_score = model_manager.load_pkl('item_average_score')
            print('model loaded')
        else:
            print('Start training...')
            self.trainset = trainset
            self.calculate_item_average_score()
            self.movie_sim_mat, self.movie_popular, self.movie_count = \
                similarity.calculate_item_similarity(trainset=trainset,
                                                     use_iuf_similarity=self.use_iuf_similarity)
            print('Train model success.')
            if self.save_model:
                model_manager.save_pkl(self.movie_sim_mat,
                                       'movie_sim_mat-iif' if self.use_iuf_similarity else 'movie_sim_mat')
                model_manager.save_pkl(self.movie_popular, 'movie_popular')
                model_manager.save_pkl(self.movie_count, 'movie_count')
                model_manager.save_pkl(self.trainset, 'trainset')
                model_manager.save_pkl(self.item_average_score, 'item_average_score')
                print('The new model has saved success.\n')

    def predict_score(self, user, item):
        score = 0
        weight = 0.0
        if user not in self.trainset:
            print('user %s is not in trainset' % str(user))
            item_average_score = self.item_average_score.get(item, 2.5)
            return item_average_score
        if item not in self.item_average_score:
            print('item %s is not in trainset' % str(item))
            user_average_score = sum(self.trainset[user].values()) / len(self.trainset[user])
            return user_average_score
        item_similarity = self.movie_sim_mat[item]
        for oth_item, oth_rate in self.trainset[user].items():
            sim = item_similarity.get(oth_item, 0)
            score += oth_rate * sim
            weight += sim
        if weight == 0:
            user_average_score = sum(self.trainset[user].values()) / len(self.trainset[user])
            return user_average_score
        return score / weight

    def prediction(self, testset: list):
        """
        :param testset: (uid, item_id) list
        :return: predicted rates
        """
        predictions = list(map(lambda x: self.predict_score(x[0], x[1]), testset))
        return predictions

    # def recommend(self, user):
    #     """
    #     Find K similar movies and recommend N movies for the user.
    #     :param user: The user we recommend movies to.
    #     :return: the N best score movies
    #     """
    #     if not self.movie_sim_mat or not self.n_rec_movie or \
    #             not self.trainset or not self.movie_popular or not self.movie_count:
    #         raise NotImplementedError('ItemCF has not init or fit method has not called yet.')
    #     K = self.k_sim_movie
    #     N = self.n_rec_movie
    #     predict_score = collections.defaultdict(int)
    #     if user not in self.trainset:
    #         print('The user (%s) not in trainset.' % user)
    #         return
    #     # print('Recommend movies to user start...')
    #     watched_movies = self.trainset[user]
    #     for movie, rating in watched_movies.items():
    #         for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
    #                                                        key=itemgetter(1), reverse=True)[0:K]:
    #             if related_movie in watched_movies:
    #                 continue
    #             # predict the user's "interest" for each movie
    #             # the predict_score is sum(similarity_factor * rating)
    #             predict_score[related_movie] += similarity_factor * rating
    #             # log steps and times.
    #     # print('Recommend movies to user success.')
    #     # return the N best score movies
    #     return [movie for movie, _ in sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]]
    #
    # def test(self, testset):
    #     """
    #     Test the recommendation system by recommending scores to all users in testset.
    #     :param testset: test dataset
    #     :return:
    #     """
    #     if not self.n_rec_movie or not self.trainset or not self.movie_popular or not self.movie_count:
    #         raise ValueError('ItemCF has not init or fit method has not called yet.')
    #     self.testset = testset
    #     print('Test recommendation system start...')
    #     N = self.n_rec_movie
    #     #  varables for precision and recall
    #     hit = 0
    #     rec_count = 0
    #     test_count = 0
    #     # varables for coverage
    #     all_rec_movies = set()
    #     # varables for popularity
    #     popular_sum = 0
    #
    #     # record the calculate time has spent.
    #     test_time = LogTime(print_step=1000)
    #     for i, user in enumerate(self.trainset):
    #         test_movies = self.testset.get(user, {})
    #         rec_movies = self.recommend(user)  # type:list
    #         for movie in rec_movies:
    #             if movie in test_movies:
    #                 hit += 1
    #             all_rec_movies.add(movie)
    #             popular_sum += math.log(1 + self.movie_popular[movie])
    #             # log steps and times.
    #         rec_count += N
    #         test_count += len(test_movies)
    #         # print time per 500 times.
    #         test_time.count_time()
    #     precision = hit / (1.0 * rec_count)
    #     recall = hit / (1.0 * test_count)
    #     coverage = len(all_rec_movies) / (1.0 * self.movie_count)
    #     popularity = popular_sum / (1.0 * rec_count)
    #
    #     print('Test recommendation system success.')
    #     test_time.finish()
    #
    #     print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f\n' %
    #           (precision, recall, coverage, popularity))
    #
    # def predict(self, testset):
    #     """
    #     Recommend movies to all users in testset.
    #     :param testset: test dataset
    #     :return: `dict` : recommend list for each user.
    #     """
    #     movies_recommend = defaultdict(list)
    #     print('Predict scores start...')
    #     # record the calculate time has spent.
    #     predict_time = LogTime(print_step=500)
    #     for i, user in enumerate(testset):
    #         rec_movies = self.recommend(user)  # type:list
    #         movies_recommend[user].append(rec_movies)
    #         # log steps and times.
    #         predict_time.count_time()
    #     print('Predict scores success.')
    #     predict_time.finish()
    #     return movies_recommend
