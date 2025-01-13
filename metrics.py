from baseline_script import EvaluationMetric, EvaluationProtocol, Track
import numpy as np

class PrecisionAtK(EvaluationMetric):
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def evaluate(self, recommended_tracks, relevant_tracks):
        if not recommended_tracks:
            return 0
        recommended_set = set(recommended_tracks[:self.k])
        relevant_set = set(relevant_tracks)
        return len(recommended_set & relevant_set) / self.k


class RecallAtK(EvaluationMetric):
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def evaluate(self, recommended_tracks, relevant_tracks):
        if not relevant_tracks:
            return 0
        recommended_set = set(recommended_tracks[:self.k])
        relevant_set = set(relevant_tracks)
        return len(recommended_set & relevant_set) / len(relevant_set)


class NDCGAtK(EvaluationMetric):
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def evaluate(self, recommended_tracks, relevant_tracks):
        def dcg(recommended, relevant, k):
            return sum((1 / np.log2(idx + 2)) for idx, track in enumerate(recommended[:k]) if track in relevant)

        def idcg(relevant, k):
            return sum((1 / np.log2(idx + 2)) for idx in range(min(k, len(relevant))))

        dcg_value = dcg(recommended_tracks, relevant_tracks, self.k)
        idcg_value = idcg(relevant_tracks, self.k)
        return dcg_value / idcg_value if idcg_value > 0 else 0


class MRR(EvaluationMetric):
    def __init__(self):
        super().__init__()

    def evaluate(self, recommended_tracks, relevant_tracks):
        for idx, track in enumerate(recommended_tracks):
            if track in relevant_tracks:
                return 1 / (idx + 1)
        return 0


class Popularity(EvaluationMetric):
    def __init_(self):
        pass

    def evaluate(self, recommended_tracks, _):
        return sum([track.popularity for track in recommended_tracks]) / len(recommended_tracks)


class DiversityAtK(EvaluationMetric):
    def __init__(self, k=10, threshold=50, max_tags=-1):
        super().__init__()
        self.k = k
        self.threshold = threshold
        self.max_tags = int(max_tags)

    def evaluate(self, recommended_tracks: list[Track], _):
        # considering all provided tags (excluding genre) per song, average number of unique tag occurences
        # among retrieved documents
        occurring_tags = []
        for track in recommended_tracks[:self.k]:
            # append tag names
            if track.tags is None:
                continue

            filtered_tags = [tag for tag in track.tags if tag.tag not in track.genres and tag.weight > self.threshold]
            filtered_tags = filtered_tags[:self.max_tags] if self.max_tags > 0 else filtered_tags
            occurring_tags.extend(filtered_tags)
        tags_set = set(occurring_tags)
        return len(tags_set) / (self.k * self.max_tags if self.max_tags > 0 else 1)


class MetricsEvaluation(EvaluationProtocol):
    def __init__(self, tracks):
        super().__init__()
        self.tracks = tracks
        # calculate median tag weight
        tag_weights = []
        tag_sizes = []
        for track in tracks:
            if track.tags is not None:
                tag_weights.extend([tag.weight for tag in track.tags])
                tag_sizes.append(len(track.tags))

        median = np.median(tag_weights)
        self.tag_threshold = median if isinstance(median, np.float64) else median.item()

        tag_size_median = np.median(tag_sizes)
        self.tag_size_threshold = tag_size_median if isinstance(tag_size_median, np.float64) else tag_size_median.item()

    def evaluate(self, ir_system, k=10):
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        mrr_scores = []
        popularity_scores = []
        diversity_scores = []

        precision_metric = PrecisionAtK(k)
        recall_metric = RecallAtK(k)
        ndcg_metric = NDCGAtK(k)
        mrr_metric = MRR()
        popularity_metric = Popularity()
        diversity_metric = DiversityAtK(k, threshold=self.tag_threshold, max_tags=self.tag_size_threshold)

        for index, query_track in enumerate(self.tracks):
            # print progress every 10%
            if index % (len(self.tracks) // 10) == 0:
                print(f"[{ir_system.name}] Progress: {index / len(self.tracks) * 100:.2f}%")

            relevant_ids = [
                track.track_id
                for track in self.tracks
                if track.track_id != query_track.track_id and (
                    any(top_genre in track.top_genres for top_genre in query_track.top_genres)
                )
            ]

            recommended_tracks, probabilities = ir_system.query(query_track, n=k)
            recommended_ids = [track.track_id for track in recommended_tracks]

            precision = precision_metric.evaluate(recommended_ids, relevant_ids)
            recall = recall_metric.evaluate(recommended_ids, relevant_ids)
            ndcg = ndcg_metric.evaluate(recommended_ids, relevant_ids)
            mrr = mrr_metric.evaluate(recommended_ids, relevant_ids)
            popularity = popularity_metric.evaluate(recommended_tracks, relevant_ids)
            diversity = diversity_metric.evaluate(recommended_tracks, relevant_ids)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            popularity_scores.append(popularity)
            diversity_scores.append(diversity)

        print(f"[{ir_system.name}] Progress: 100.00%")
        results = {
            "Precision@10": np.mean(precision_scores),
            "Recall@10": np.mean(recall_scores),
            "NDCG@10": np.mean(ndcg_scores),
            "MRR": np.mean(mrr_scores),
            "Popularity": np.mean(popularity_scores),
            "Diversity": np.mean(diversity_scores)
        }
        return results

