import random
import numpy as np
import torch
import glob
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from functools import cmp_to_key
from collections import defaultdict
from src.experiments.experiment_1 import Experiment1
from src.experiments.experiment_2 import Experiment2
from src.data import dataset_as_dataframe

# Lock random seeds
_RNG_SEED_VAL = 1337
random.seed(_RNG_SEED_VAL)
np.random.seed(_RNG_SEED_VAL)
torch.manual_seed(_RNG_SEED_VAL)
torch.cuda.manual_seed_all(_RNG_SEED_VAL)

# Get word freqencies
with open("wiki_freq_filtered.pkl", 'rb') as f:
    WIKI_WORD_FREQS = pickle.load(f)
TOP_2000_WORDS = set(WIKI_WORD_FREQS['index'][:2000])

WORD_DF = dataset_as_dataframe()

def load_checkpoint(checkpoint, model):
    print(f"=> Loading weights from {checkpoint}")
    model.load_state_dict(torch.load(checkpoint))


def custom_comparator(s1, s2):
    s = "params.pt."
    assert s1.startswith(s)
    assert s2.startswith(s)
    s1 = s1[len(s):]
    s2 = s2[len(s):]
    if s1 == s2:
        return 0
    elif s1 == "final":
        return 1
    elif s2 == "final":
        return -1
    else:
        return int(s1) - int(s2)


def get_sorted_checkpoints(ckpt_dir):
    L = glob.glob(os.path.join(ckpt_dir, "params.pt.*"))
    L = [os.path.basename(s) for s in L]
    return sorted(L, key=cmp_to_key(custom_comparator))


def run_eval(ckpt_dir):
    if "linear" in ckpt_dir:
        experiment = Experiment1("checkpoints/tmp", batch_size=32)
    elif "lstm" in ckpt_dir:
        experiment = Experiment2("checkpoints/tmp", batch_size=32)
    else:
        raise Exception("Bad ckpt dir")
    all_ckpts = get_sorted_checkpoints(ckpt_dir)
    for ckpt in all_ckpts:
        load_checkpoint(os.path.join(ckpt_dir, ckpt), experiment.model)
        if "linear" in ckpt_dir:
            get_accuracy = any(ckpt.endswith(s) for s in [".1", ".3", ".6", ".9", ".12", ".15"])
        elif "lstm" in ckpt_dir:
            get_accuracy = any(ckpt.endswith(s) for s in [".1", ".10", ".20", ".30", ".40", ".50", ".60", ".70", ".80", ".90", ".100"])
        else:
            raise Exception("Bad ckpt dir")

        stats = experiment.analyze(experiment.train_dataloader, check_accuracy=get_accuracy)
        with open(os.path.join(ckpt_dir, 'train-stats.' + ckpt), 'wb') as f:
            pickle.dump(stats, f)

        stats = experiment.analyze(experiment.val_dataloader, check_accuracy=get_accuracy)
        with open(os.path.join(ckpt_dir, "val-stats." + ckpt), 'wb') as f: pickle.dump(stats, f)
    return stats

def normalize_losses(losses, max_thresh=100):
    return [i if i < max_thresh else max_thresh for i in losses]

def do_analysis(ckpt_file, use_cached=False, train=True):
    global WIKI_WORD_FREQS, TOP_2000_WORDS, WORD_DF

    if train:
        title = "Training"
        prefix = "training"
    else:
        title = "Validation"
        prefix = "validation"

    if not use_cached:
        if 'linear' in ckpt_file:
            experiment = Experiment1("checkpoints/tmp", batch_size=32)
        elif 'lstm' in ckpt_file:
            experiment = Experiment2("checkpoints/tmp", batch_size=32)
        load_checkpoint(ckpt_file, experiment.model)
        if train:
            print("=> Analyzing training set")
            dataloader = experiment.train_dataloader
        else:
            print("=> Analyzing validation set")
            dataloader = experiment.val_dataloader
        perf, metadata = experiment.analyze(dataloader, check_accuracy=True, get_metadata=True)
        with open(ckpt_file + f".analysis.{prefix}", 'wb') as f:
            pickle.dump((perf, metadata), f)
    else:
        with open(ckpt_file + f".analysis.{prefix}", 'rb') as f:
            t = pickle.load(f)
        perf, metadata = t
    print(f"====== {title} set ======")
    print("Average batch time: ", metadata["avg_per_batch_time"])
    print("Top-1 Acc: ", perf["top_1"])
    print("Top-10 Acc: ", perf["top_10"])
    print("Top-100 Acc: ", perf["top_100"])
    print("Avg loss (all): ", sum(metadata["all_losses"]) / len(metadata["all_losses"]))
    print("Loss std dev: ", np.std(np.array(metadata["all_losses"])))
    print("Worst loss: ", max(metadata["all_losses"]))
    print("% of losses greater than 60: ", len([i for i in metadata["all_losses"] if i >= 60]) / len(metadata["all_losses"]))

    # Get the largest magnitude output vectors
    magnitudes = [(y_hat ** 2).sum() for y_hat in metadata["y_hat"]]
    print("Average magnitude of model output: ", sum(magnitudes) / len(magnitudes))
    print("Maximum magnitude of model outputs: ", max(magnitudes))

    # Generate loss histogram:
    print("Generating histogram of losses")
    normalized_losses = normalize_losses(metadata["all_losses"])
    plt.clf()
    plt.hist(normalized_losses, 50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel("Loss")
    plt.ylabel("% of samples")
    plt.title(f"{title} Loss Distribution by Samples")
    figure_path = os.path.join(os.path.dirname(ckpt_file), f"{prefix}_loss_dist.png")
    plt.savefig(figure_path)
    print(f"Saved histogram to {figure_path}")


    print("Doing PoS analysis...")
    # Part of speech dict
    pos_dist = {
        "top_100": defaultdict(int),
        "failed": defaultdict(int),
    }
    if not use_cached:
        for i in tqdm(range(len(metadata["labels"]))):
            ss_types = WORD_DF[WORD_DF["word"] == metadata["labels"][i]]["ss_type"]
            num_types = len(set(ss_types))
            if metadata["rank"][i] is None:
                pos_dist["failed"][num_types] += 1
            else:
                pos_dist["top_100"][num_types] += 1
        with open(ckpt_file + f".part_of_speech.{prefix}", 'wb') as f:
            pickle.dump(pos_dist, f)
    else:
        with open(ckpt_file + f".part_of_speech.{prefix}", 'rb') as f:
            pos_dist = pickle.load(f)
    print("Part of speech statistics:")
    print(pos_dist["failed"])


    # Top 10 worst performing words by loss:
    loss_sorted_indices = np.argsort(np.array(metadata["all_losses"]))
    worst_words = set()
    for idx in loss_sorted_indices[::-1]:
        if len(worst_words) >= 10:
            break
        worst_words.add(metadata["labels"][idx])
    print("Top 10 worst performing words: ", worst_words)

    # Get top 2000 words
    top_2000_indices = [idx for idx in range(len(metadata["labels"])) if metadata["labels"][idx] in TOP_2000_WORDS]
    print("Occurrences of top 2000 most frequent words: ", len(top_2000_indices))
    top_2000_losses = [metadata["all_losses"][idx] for idx in top_2000_indices]
    top_2000_ranks = [metadata["rank"][idx] for idx in top_2000_indices]

    print("Top-1 Acc: ", sum(1 for _ in filter(lambda i: i == 1, top_2000_ranks)) / len(top_2000_ranks))
    print("Top-10 Acc: ", sum(1 for _ in filter(lambda i: i is not None and i <= 10, top_2000_ranks)) / len(top_2000_ranks))
    print("Top-100 Acc: ", sum(1 for _ in filter(lambda i: i is not None and i <= 100, top_2000_ranks)) / len(top_2000_ranks))
    print("Top 2K Word Avg loss (all): ", sum(top_2000_losses) / len(top_2000_losses))
    print("Top 2K Worst loss: ", max(top_2000_losses))
    print("Top 2K % of losses greater than 60: ", len([i for i in top_2000_losses if i >= 60]) / len(top_2000_losses))

    # Plot loss distribution for top 2K words
    normalized_losses = normalize_losses(top_2000_losses)
    plt.clf()
    plt.hist(normalized_losses, 50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel("Loss")
    plt.ylabel("% of samples")
    plt.title(f"{title} Loss Distribution for Top 2K Words by Freq")
    figure_path = os.path.join(os.path.dirname(ckpt_file), f"{prefix}_loss_dist_top_2k.png")
    plt.savefig(figure_path)
    print(f"Saved histogram to {figure_path}")


if __name__ == "__main__":
    # checkpoint_dir = "checkpoints/linear.dropout.full-train"
    checkpoint_dir = "checkpoints/lstm.4-layer.100.full-train"
    run_eval(checkpoint_dir)

