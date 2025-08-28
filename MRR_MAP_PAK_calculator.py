import json
from collections import defaultdict

with open('paper_reviewer_matches_llamma_summary_jaccard.json') as f:
    recommendations = json.load(f)

with open('EC_dizionario_articoli.json') as f:
    ground_truth = json.load(f)

def get_ground_truth_reviewers(paper_id):
    return [reviewer.strip() for reviewer in ground_truth.get(paper_id, [""])[5].split(",")]


def calculate_mrr(recommendations, ground_truth, k=3):
    mrr_total = 0
    count = 0
    for paper_id, recs in recommendations.items():
        gt_reviewers = get_ground_truth_reviewers(paper_id)
        if not gt_reviewers or gt_reviewers == ["Reviewers articolo non trovati"]:
            continue
        
        found = False
        for rank, (reviewer, _) in enumerate(recs[:k], start=1):
            reviewer_name = eval(reviewer)[1]
            if reviewer_name in gt_reviewers:
                mrr_total += 1 / rank
                found = True
                break
        if found:
            count += 1
    return mrr_total / count if count else 0

def calculate_precision_at_k(recommendations, ground_truth, k=3):
    precision_total = 0
    count = 0
    for paper_id, recs in recommendations.items():
        gt_reviewers = get_ground_truth_reviewers(paper_id)
        if not gt_reviewers or gt_reviewers == ["Reviewers articolo non trovati"]:
            continue

        top_k_reviewers = [eval(reviewer)[1] for reviewer, _ in recs[:k]]
        relevant_count = sum(1 for reviewer in top_k_reviewers if reviewer in gt_reviewers)

        if relevant_count > 0:
            precision_total += relevant_count / k
            count += 1
    return precision_total / count if count else 0


def calculate_map(recommendations, ground_truth, k=3):
    ap_total = 0
    count = 0
    for paper_id, recs in recommendations.items():
        gt_reviewers = get_ground_truth_reviewers(paper_id)
        if not gt_reviewers or gt_reviewers == ["Reviewers articolo non trovati"]:
            continue

        relevant_ranks = 0
        avg_precision = 0
        for rank, (reviewer, _) in enumerate(recs[:k], start=1):
            reviewer_name = eval(reviewer)[1]
            if reviewer_name in gt_reviewers:
                relevant_ranks += 1
                avg_precision += relevant_ranks / rank
        if relevant_ranks > 0:
            ap_total += avg_precision / min(len(gt_reviewers), k)
            count += 1
    return ap_total / count if count else 0

mrr = calculate_mrr(recommendations, ground_truth, k=3)
precision_at_3 = calculate_precision_at_k(recommendations, ground_truth, k=3)
map_score = calculate_map(recommendations, ground_truth, k=3)


print("Mean Reciprocal Rank (MRR):", mrr)
print("Precision at 3 (P@3):", precision_at_3)
print("Mean Average Precision (MAP):", map_score)
