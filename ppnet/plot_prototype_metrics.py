# plot_prototype_metrics.py
import argparse
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram_max_freq(csv_path, out_path, bins=50):
    df = pd.read_csv(csv_path)
    sns.histplot(df['max_freq'], bins=bins, kde=True)
    plt.xlabel('Prototype Max‐Frequency (dominant part fraction)')
    plt.ylabel('Number of Prototypes')
    plt.title('Distribution of Prototype Consistency Frequencies')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_scatter_class_accuracy_consistency(class_stats_csv, proto_max_freq_csv, out_path):
    """
    class_stats_csv: CSV with columns ['class', 'accuracy', 'avg_proto_max_freq']
    proto_max_freq_csv: CSV with columns ['proto_idx', 'max_freq']
    """
    df_class = pd.read_csv(class_stats_csv)
    sns.scatterplot(data=df_class, x='avg_proto_max_freq', y='accuracy')
    plt.xlabel('Avg Prototype Max‐Freq per Class')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('Class Accuracy vs Prototype Consistency')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_frac_same(csv_path, out_path, bins=50):
    df = pd.read_csv(csv_path)
    sns.histplot(df['frac_same'], bins=bins, kde=True)
    plt.xlabel('Prototype Stability Fraction (same part label under perturbation)')
    plt.ylabel('Number of Prototypes')
    plt.title('Distribution of Prototype Stability Fractions')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot metrics from prototype consistency / stability evaluation")
    parser.add_argument('--results_dir', type=str, required=True, help='directory where validation results (CSV/JSON) are saved')
    parser.add_argument('--class_stats_csv', type=str, default=None,
                        help='(Optional) CSV with class-level stats: class, accuracy, avg_proto_max_freq')
    parser.add_argument('--output_dir', type=str, required=True, help='directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Histogram of max_freq
    csv_max_freq = os.path.join(args.results_dir, "per_proto_max_freq.csv")
    if os.path.exists(csv_max_freq):
        hist_path = os.path.join(args.output_dir, "hist_max_freq.png")
        plot_histogram_max_freq(csv_max_freq, hist_path)
        print(f"Saved histogram of max_freq to {hist_path}")
    else:
        print(f"Warning: {csv_max_freq} not found.")

    # If stability was run
    csv_frac_same = os.path.join(args.results_dir, "per_proto_frac_same.csv")
    if os.path.exists(csv_frac_same):
        hist2_path = os.path.join(args.output_dir, "hist_frac_same.png")
        plot_frac_same(csv_frac_same, hist2_path)
        print(f"Saved histogram of frac_same to {hist2_path}")
    else:
        print(f"Stability results not found – skipping.")

    # Scatter plot of class accuracy vs avg consistency
    if args.class_stats_csv is not None and os.path.exists(args.class_stats_csv):
        scatter_path = os.path.join(args.output_dir, "scatter_acc_vs_consistency.png")
        plot_scatter_class_accuracy_consistency(args.class_stats_csv, csv_max_freq, scatter_path)
        print(f"Saved scatter plot to {scatter_path}")
    else:
        print("Class stats CSV not provided or not found – skipping scatter plot.")

    print("Plotting done.")

if __name__ == '__main__':
    main()
