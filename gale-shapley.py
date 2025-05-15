import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def generate_preferences(n: int):
    """
    Given a number n, generate n random preference orderings of 1-n for use in Gale-Shapley
    """
    prefs = np.tile(np.arange(n), (n, 1))
    rng = np.random.default_rng()
    rng.permuted(prefs, axis=1, out=prefs)
    return prefs


def generate_popularity_preferences(n: int):
    rng = np.random.default_rng()
    s_range = np.arange(1, n + 1)
    scores = rng.permutation(s_range) ** 2
    scores = scores / np.sum(scores)
    prefs = np.ndarray((n, n), dtype=int)
    for i in range(n):
        prefs[i] = np.random.choice(np.arange(n), size=n, replace=False, p=scores)
    return prefs


def run_gale_shapley(n: int, pop=False):
    """
    Runs Gale-Shapley with random preferences for a given integer n being the number of doctors and hospitals. Returns an array giving the doctor for each hopsitla
    """

    # note we treat these asymmetrically: d_prefs[i,j] is taken to be doctor i's jth favorite hospital, while h_prefs[i,j] is how hospital i ranks doctor j. This makes running the algorithm much simpler
    #
    if pop:
        d_prefs = generate_popularity_preferences(n)
        h_prefs = generate_popularity_preferences(n)
    else:
        d_prefs = generate_preferences(n)
        h_prefs = generate_preferences(n)

    matches = np.ndarray((n), dtype=int)
    matches.fill(-1)
    # track the number of proposals for each doctor
    proposals = np.zeros((n), dtype=int)
    queue = deque(range(n))
    while len(queue) > 0:
        d = queue.popleft()
        i = proposals[d]
        while i < n:
            h = d_prefs[d, i]
            i += 1
            proposals[d] += 1
            if matches[h] == -1:
                matches[h] = d
                break
            # note a higher h_prefs means that that doctor is LESS preferred
            elif h_prefs[h, matches[h]] > h_prefs[h, d]:
                queue.appendleft(matches[h])
                matches[h] = d
                break

    # Anthropic's claude helped me figure out the indexing here
    h_ranks = h_prefs[range(n), matches] + 1
    return matches, proposals, h_ranks


def plot_average_proposals(pop=False):
    nvals = np.arange(100, 1001, 100)
    average_props = np.zeros_like(nvals, dtype=float)
    for i, n in enumerate(nvals):
        total_props = 0
        runs = 10
        for j in range(runs):
            _, props, _ = run_gale_shapley(n, pop)
            total_props += props.sum()
        average_props[i] = total_props / runs
    plt.close()
    plt.scatter(nvals, average_props, label="Average proposals")
    plt.plot(nvals, nvals * np.log(nvals), color="red", label="nlog(n)")
    plt.title("Average proposals as a function of n")
    plt.xlabel("n")
    plt.ylabel("Average proposals")
    plt.legend()
    if pop:
        plt.savefig("Average popularity Proposals.pdf", bbox_inches="tight")
    else:
        plt.savefig("Average Proposals.pdf", bbox_inches="tight")


def plot_total_proposals(n, pop=False):
    num_props = []
    for i in range(15):
        _, props, _ = run_gale_shapley(n, pop)
        num_props.append(props.sum())
    plt.close()
    plt.scatter(range(15), num_props)
    plt.title(f"Proposals with {n} doctors across several iterations")
    plt.xlabel("Iteration number")
    plt.ylabel("Proposals")
    if pop:
        plt.savefig("Total Proposals popularity.pdf", bbox_inches="tight")
    else:
        plt.savefig("Total Proposals.pdf", bbox_inches="tight")


def plot_average_ranking(pop=False):
    if pop:
        max_range = 1001
    else:
        max_range = 3001
    nvals = np.arange(100, max_range, 100)
    average_doc_ranks = np.zeros_like(nvals, dtype=float)
    average_hosp_ranks = np.zeros_like(nvals, dtype=float)
    for i, n in enumerate(nvals):
        total_doc_ranks = 0
        total_hosp_ranks = 0
        if pop:
            runs = 5
        else:
            runs = 10
        for j in range(runs):
            _, props, hosp_ranks = run_gale_shapley(n, pop)
            # note the number of proposals from each doctor also gives us their ranking for their resulting match
            total_doc_ranks += props.mean()
            total_hosp_ranks += hosp_ranks.mean()
        average_doc_ranks[i] = total_doc_ranks / runs
        average_hosp_ranks[i] = total_hosp_ranks / runs
    plt.close()
    plt.scatter(nvals, average_doc_ranks, label="Average doctor ranking")
    plt.ylabel("Average doctor rankings")
    plt.title("Average doctor rankings as a function of n")
    plt.xlabel("n")
    if not pop:
        plt.yticks(range(1, 11))
        x = np.linspace(100, max_range)
        y = np.log(x)
        plt.scatter(x, y, color="red", label="log(n)")
        plt.legend()
        plt.savefig("Average Doctor rankings.pdf", bbox_inches="tight")
    else:
        plt.savefig("Average Doctor popularity rankings.pdf", bbox_inches="tight")

    plt.close()
    plt.scatter(nvals, average_hosp_ranks, label="Average hospital ranking")
    plt.ylabel("Average hospital rankings")
    plt.title("Average hospital rankings as a function of n")
    plt.xlabel("n")
    if not pop:
        x = np.linspace(100, max_range)
        y = x / np.log(x)
        plt.scatter(x, y, color="red", label="n/log(n)")
        plt.legend()
        plt.savefig("Average Hospital rankings.pdf", bbox_inches="tight")
    else:
        plt.savefig("Average Hospital popularity rankings.pdf", bbox_inches="tight")


# plot_average_proposals()
# plot_average_proposals(pop=True)
# plot_total_proposals(2000)
# plot_total_proposals(2000, pop=True)
plot_average_ranking()
plot_average_ranking(pop=True)

run_gale_shapley(100)
