import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HUMAN_V2    = "results_human_v2.csv"
SAL_CSV     = "results_three_axes_v2.csv"
SIGHTED_CSV = "behavioral_data/sighted_data_criticaltrials.csv"

def main():
    human_df = pd.read_csv(HUMAN_V2)

    sal_df = pd.read_csv(SAL_CSV)[['img_file', 'context', 'saliency']].copy()
    sal_df.loc[sal_df['saliency'] < 0.05, 'saliency'] = None

    sighted_df = pd.read_csv(SIGHTED_CSV)
    sighted_agg = sighted_df.groupby(
        ['img_id', 'context', 'description']
    ).agg(
        sighted_relevance_pre  = ('q_relevance.preimg',  'mean'),
        sighted_relevance_post = ('q_relevance.postimg', 'mean'),
        sighted_overall_post   = ('q_overall.postimg',   'mean'),
    ).reset_index()

    merged = human_df.merge(sal_df, on=['img_file', 'context'], how='left')
    merged = merged.merge(sighted_agg,
                          on=['img_id', 'context', 'description'],
                          how='left')

    plot_df = merged.dropna(subset=['saliency'])
    print(f'Total rows with saliency: {len(plot_df)}')

    outcomes = [
        ('sighted_reconstruct',    'Sighted Reconstructivity'),
        ('sighted_overall_post',   'Sighted Overall (post-image)'),
        ('sighted_relevance_pre',  'Sighted Relevance (pre-image)'),
        ('sighted_relevance_post', 'Sighted Relevance (post-image)'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        'Saliency Alignment vs. Human Ratings (Human-Written Descriptions)',
        fontsize=13)

    for ax, (outcome, label) in zip(axes, outcomes):
        df = plot_df.dropna(subset=[outcome])
        x  = df['saliency']
        y  = df[outcome]

        r,   p   = stats.pearsonr(x, y)
        rho, p_r = stats.spearmanr(x, y)

        ax.scatter(x, y, alpha=0.4, s=25, color='steelblue',
                   edgecolors='white', linewidth=0.3)
        m, b   = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, m*x_line+b, color='tomato', linewidth=2)

        ax.set_xlabel('Saliency Alignment', fontsize=11)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(
            f'r={r:.2f} (p={p:.3f})\nρ={rho:.2f} (p={p_r:.3f}), n={len(df)}',
            fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('saliency_vs_human_ratings.png', dpi=150, bbox_inches='tight')
    print('Saved saliency_vs_human_ratings.png')

if __name__ == '__main__':
    main()