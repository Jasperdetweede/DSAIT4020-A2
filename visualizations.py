import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_phq9_distribution(dataset, save_path='./visualizations/phq9_distribution.png', show=True):
    """Plot stacked bar chart of PHQ-9 depression symptom score distributions."""
    phq9_cols = ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090']
    phq9_labels = [
        'Have little \ninterest in doing things',
        'Feeling down,\ndepressed or hopeless',
        'Trouble sleeping or\nsleeping too much',
        'Feeling tired or\nhaving little energy',
        'Poor appetite\nor overeating',
        'Feeling bad\nabout yourself',
        'Trouble\nconcentrating on things',
        'Moving or speaking\nslowly or too fast',
        'Thought you would\nbe better off dead'
    ]

    score_counts = pd.DataFrame(index=phq9_cols, columns=[0, 1, 2, 3])
    for col in phq9_cols:
        counts = dataset[col].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
        score_counts.loc[col] = counts

    score_counts = score_counts.astype(float)

    _, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    score_labels = ['Not at all (0)', 'Several days (1)', 'More than half the days (2)', 'Nearly every day (3)']

    x = np.arange(len(phq9_cols))
    width = 0.7

    bottom = np.zeros(len(phq9_cols))
    for score in [0, 1, 2, 3]:
        values = score_counts[score].values
        ax.bar(x, values, width, label=score_labels[score], bottom=bottom, color=colors[score])
        bottom += values

    ax.set_ylabel('Percentage of Respondents (%)', fontsize=11)
    ax.set_xlabel('PHQ-9 Symptom', fontsize=11)
    ax.set_title('Distribution of PHQ-9 Depression Symptom Scores (NHANES 2021-2023)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phq9_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(title='Response', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 100)

    variable_text = '\n'.join([f'{col}: {i+1}' for i, col in enumerate(phq9_cols)])
    ax.text(1.02, 0.35, 'Variable Codes:\n' + '\n'.join([f'{i+1}. {col}' for i, col in enumerate(phq9_cols)]),
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    print("\n=== PHQ-9 Score Distribution Summary ===")
    print(f"Total respondents: {len(dataset)}")
    print(f"\nPercentage with score \u22652 (symptom present) per item:")
    for col, label in zip(phq9_cols, phq9_labels):
        pct_present = (dataset[col] >= 2).sum() / dataset[col].notna().sum() * 100
        print(f"  {label.replace(chr(10), ' ')}: {pct_present:.1f}%")
