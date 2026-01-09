"""
Visualizzazione dei risultati per il Record Linkage con Dedupe.

Confronta:
1. Pipeline ML con Auto-Blocking (P1, P2, P3) - Dedupe impara le regole di blocking
2. Pipeline ML con Manual Blocking (P1/P2/P3 × B1/B2/Union) - Blocking manuale
3. Blocking Solo (senza ML) - Solo regole euristiche

Output: Grafici salvati in appoggio/dedupe/plots/
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from collections import defaultdict

# Stile globale per presentazioni
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Palette colori coerente
COLORS = {
    'auto': '#2ecc71',      # Verde - Auto-blocking (best)
    'B1': '#3498db',        # Blu - Manual B1
    'B2': '#9b59b6',        # Viola - Manual B2
    'Union': '#e74c3c',     # Rosso - Manual Union
    'blocking_only': '#95a5a6',  # Grigio - Blocking solo
    'P1': '#e74c3c',
    'P2': '#f39c12', 
    'P3': '#2ecc71',
}


def load_all_results():
    """Carica tutti i risultati disponibili."""
    results = []
    
    # 1. Auto-blocking experiments
    auto_path = 'output/dedupe_results/experiments/summary_all.json'
    if os.path.exists(auto_path):
        with open(auto_path) as f:
            data = json.load(f)
            for r in data['runs']:
                results.append({
                    'name': r['name'],
                    'display_name': r['name'].replace('_', ' ').title(),
                    'category': 'Auto-Blocking',
                    'pipeline': r['name'].split('_')[0],
                    'blocking': 'auto',
                    'precision': r['metrics']['precision'],
                    'recall': r['metrics']['recall'],
                    'f1': r['metrics']['f1'],
                    'tp': r['metrics']['tp'],
                    'fp': r['metrics']['fp'],
                    'fn': r['metrics']['fn'],
                    'timings': r.get('timings', {}),
                    'total_time': sum(r.get('timings', {}).values())
                })
    
    # 2. Manual-blocking experiments
    manual_path = 'output/dedupe_results/manual_blocking_experiments/summary_all.json'
    if os.path.exists(manual_path):
        with open(manual_path) as f:
            data = json.load(f)
            for r in data['runs']:
                parts = r['name'].split('_manual_')
                pipeline_name = parts[0] if parts else r['name']
                blocking_strat = parts[1] if len(parts) > 1 else 'unknown'
                
                results.append({
                    'name': r['name'],
                    'display_name': f"{pipeline_name.split('_')[0]} + {blocking_strat}",
                    'category': 'Manual-Blocking',
                    'pipeline': pipeline_name.split('_')[0],
                    'blocking': blocking_strat,
                    'precision': r['metrics']['precision'],
                    'recall': r['metrics']['recall'],
                    'f1': r['metrics']['f1'],
                    'tp': r['metrics']['tp'],
                    'fp': r['metrics']['fp'],
                    'fn': r['metrics']['fn'],
                    'timings': r.get('timings', {}),
                    'total_time': sum(r.get('timings', {}).values())
                })
    
    # 3. Blocking-only results
    blocking_only = [
        {'name': 'B1_only', 'display_name': 'Blocking B1',
         'category': 'Blocking Only', 'pipeline': 'B1', 'blocking': 'B1',
         'precision': 0.272, 'recall': 1.0, 'f1': 0.428, 'tp': 306, 'fp': 818, 'fn': 0,
         'timings': {}, 'total_time': 0.1},
        {'name': 'B2_only', 'display_name': 'Blocking B2',
         'category': 'Blocking Only', 'pipeline': 'B2', 'blocking': 'B2',
         'precision': 0.231, 'recall': 0.974, 'f1': 0.373, 'tp': 298, 'fp': 992, 'fn': 8,
         'timings': {}, 'total_time': 0.1},
        {'name': 'Union_only', 'display_name': 'Blocking Union',
         'category': 'Blocking Only', 'pipeline': 'Union', 'blocking': 'Union',
         'precision': 0.162, 'recall': 1.0, 'f1': 0.278, 'tp': 306, 'fp': 1587, 'fn': 0,
         'timings': {}, 'total_time': 0.1},
    ]
    results.extend(blocking_only)
    
    # 4. P3 Extended experiment (P3 + price/mileage con manual blocking)
    # Esperimento per verificare se features aggiuntive migliorano P3 su manual blocking
    p3_extended = [
        {'name': 'P3_extended_B1', 'display_name': 'P3 Extended + B1',
         'category': 'Manual-Blocking', 'pipeline': 'P3ext', 'blocking': 'B1',
         'precision': 0.645, 'recall': 0.928, 'f1': 0.761, 'tp': 284, 'fp': 156, 'fn': 22,
         'timings': {}, 'total_time': 15.0},
        {'name': 'P3_extended_B2', 'display_name': 'P3 Extended + B2',
         'category': 'Manual-Blocking', 'pipeline': 'P3ext', 'blocking': 'B2',
         'precision': 0.541, 'recall': 0.585, 'f1': 0.562, 'tp': 179, 'fp': 152, 'fn': 127,
         'timings': {}, 'total_time': 12.0},
    ]
    results.extend(p3_extended)
    
    return results


def plot_main_comparison(results, output_dir):
    """
    Grafico principale: confronto F1 tra Auto-Blocking, Manual-Blocking e Blocking-Only.
    Ideale per slide di apertura.
    """
    print("  [1/6] Main F1 Comparison...")
    
    # Raggruppa per categoria
    auto = [r for r in results if r['category'] == 'Auto-Blocking']
    manual = [r for r in results if r['category'] == 'Manual-Blocking']
    blocking = [r for r in results if r['category'] == 'Blocking Only']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Ordina per F1
    auto = sorted(auto, key=lambda x: x['f1'], reverse=True)
    manual = sorted(manual, key=lambda x: x['f1'], reverse=True)
    blocking = sorted(blocking, key=lambda x: x['f1'], reverse=True)
    
    all_sorted = auto + manual + blocking
    
    positions = np.arange(len(all_sorted))
    colors = []
    for r in all_sorted:
        if r['category'] == 'Auto-Blocking':
            colors.append(COLORS['auto'])
        elif r['category'] == 'Blocking Only':
            colors.append(COLORS['blocking_only'])
        else:
            colors.append(COLORS.get(r['blocking'], '#3498db'))
    
    bars = ax.barh(positions, [r['f1'] for r in all_sorted], color=colors, 
                   edgecolor='white', linewidth=0.5, height=0.7)
    
    # Annotazioni
    for i, (bar, r) in enumerate(zip(bars, all_sorted)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(positions)
    ax.set_yticklabels([r['display_name'] for r in all_sorted])
    ax.set_xlabel('F1 Score')
    ax.set_title('Confronto Approcci: F1 Score', fontweight='bold')
    ax.set_xlim(0, 1.05)
    
    # Linea soglia
    ax.axvline(x=0.9, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(0.91, len(all_sorted)-0.5, 'Target\n(0.90)', fontsize=9, color='#2ecc71')
    
    # Separatori di categoria
    ax.axhline(y=len(auto)-0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axhline(y=len(auto)+len(manual)-0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # Legenda custom
    legend_elements = [
        mpatches.Patch(color=COLORS['auto'], label='Auto-Blocking ML'),
        mpatches.Patch(color=COLORS['B1'], label='Manual-Blocking (B1)'),
        mpatches.Patch(color=COLORS['B2'], label='Manual-Blocking (B2)'),
        mpatches.Patch(color=COLORS['Union'], label='Manual-Blocking (Union)'),
        mpatches.Patch(color=COLORS['blocking_only'], label='Blocking Solo'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_main_f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pipeline_heatmap(results, output_dir):
    """
    Heatmap: F1 Score per Pipeline × Blocking Strategy.
    Visualizzazione matriciale delle performance.
    """
    print("  [2/6] Pipeline × Blocking Heatmap...")
    
    pipelines = ['P1', 'P2', 'P3']
    blockings = ['auto', 'B1', 'B2', 'Union']
    
    # Costruisci matrice
    matrix = np.zeros((len(pipelines), len(blockings)))
    matrix[:] = np.nan  # NaN per celle vuote
    
    for r in results:
        if r['pipeline'] in pipelines and r['blocking'] in blockings:
            i = pipelines.index(r['pipeline'])
            j = blockings.index(r['blocking'])
            matrix[i, j] = r['f1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Cmap personalizzata
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='lightgray')
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Etichette
    ax.set_xticks(np.arange(len(blockings)))
    ax.set_yticks(np.arange(len(pipelines)))
    ax.set_xticklabels(['Auto-Blocking', 'Manual B1', 'Manual B2', 'Manual Union'])
    ax.set_yticklabels(['P1 (textual_core)', 'P2 (plus_location)', 'P3 (minimal_fast)'])
    
    # Annotazioni celle
    for i in range(len(pipelines)):
        for j in range(len(blockings)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color=color)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', 
                        fontsize=12, color='gray', style='italic')
    
    ax.set_title('F1 Score: Pipeline × Blocking Strategy', fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_pipeline_blocking_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_tradeoff(results, output_dir):
    """
    Scatter plot Precision vs Recall con curve iso-F1.
    Mostra il trade-off tra precision e recall.
    """
    print("  [3/6] Precision-Recall Trade-off...")
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Curve iso-F1
    for f1_val in [0.2, 0.4, 0.6, 0.8, 0.9]:
        recalls = np.linspace(0.01, 1, 200)
        precisions = (f1_val * recalls) / (2 * recalls - f1_val)
        valid = (precisions > 0) & (precisions <= 1.05)
        ax.plot(recalls[valid], precisions[valid], '--', alpha=0.4, color='gray', linewidth=1)
        # Label sulla curva
        idx = np.argmin(np.abs(recalls[valid] - 0.95))
        if idx < len(precisions[valid]):
            ax.text(0.97, precisions[valid][idx], f'F1={f1_val}', 
                    fontsize=8, color='gray', va='center')
    
    # Plot punti
    markers = {'Auto-Blocking': 'o', 'Manual-Blocking': 's', 'Blocking Only': '^'}
    
    for r in results:
        if r['f1'] < 0.01:  # Salta risultati falliti
            continue
            
        color = COLORS.get(r['blocking'], '#95a5a6')
        marker = markers.get(r['category'], 'o')
        size = 150 if r['category'] == 'Auto-Blocking' else 100
        
        ax.scatter(r['recall'], r['precision'], c=color, s=size, marker=marker,
                   edgecolors='black', linewidth=1, alpha=0.85, zorder=5)
        
        # Annotazione
        offset = (5, 5) if r['precision'] < 0.9 else (5, -10)
        ax.annotate(r['display_name'], (r['recall'], r['precision']),
                    fontsize=8, xytext=offset, textcoords='offset points',
                    alpha=0.9)
    
    # Evidenzia best performer
    best = max([r for r in results if r['f1'] > 0], key=lambda x: x['f1'])
    ax.scatter(best['recall'], best['precision'], s=300, facecolors='none',
               edgecolors='#2ecc71', linewidth=3, zorder=10, label=f"Best: {best['display_name']}")
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Trade-off Precision-Recall', fontweight='bold')
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['auto'], 
                   markersize=12, label='Auto-Blocking', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['B1'], 
                   markersize=10, label='Manual B1', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['B2'], 
                   markersize=10, label='Manual B2', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['Union'], 
                   markersize=10, label='Manual Union', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['blocking_only'], 
                   markersize=10, label='Blocking Only', markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_precision_recall_tradeoff.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_auto_blocking_focus(results, output_dir):
    """
    Focus sulle 3 pipeline Auto-Blocking: P1, P2, P3.
    Grafico a barre raggruppate per Precision, Recall, F1.
    """
    print("  [4/6] Auto-Blocking Focus...")
    
    auto_results = [r for r in results if r['category'] == 'Auto-Blocking']
    auto_results = sorted(auto_results, key=lambda x: x['pipeline'])
    
    if not auto_results:
        print("    No auto-blocking results, skipping...")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(auto_results))
    width = 0.25
    
    precision = [r['precision'] for r in auto_results]
    recall = [r['recall'] for r in auto_results]
    f1 = [r['f1'] for r in auto_results]
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#f39c12', alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#2ecc71', alpha=0.85)
    
    # Etichette valori
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([r['display_name'] for r in auto_results])
    ax.set_ylabel('Score')
    ax.set_title('Auto-Blocking ML: Confronto Pipeline', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.axhline(y=0.9, color='#27ae60', linestyle='--', alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_auto_blocking_focus.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_analysis(results, output_dir):
    """
    Analisi errori: TP, FP, FN per i top performer.
    """
    print("  [5/6] Error Analysis...")
    
    # Filtra e ordina per F1
    valid = [r for r in results if r['f1'] > 0.1]
    top_results = sorted(valid, key=lambda x: x['f1'], reverse=True)[:8]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(top_results))
    width = 0.25
    
    tp = [r['tp'] for r in top_results]
    fp = [r['fp'] for r in top_results]
    fn = [r['fn'] for r in top_results]
    
    bars1 = ax.bar(x - width, tp, width, label='True Positives', color='#2ecc71')
    bars2 = ax.bar(x, fp, width, label='False Positives', color='#e74c3c')
    bars3 = ax.bar(x + width, fn, width, label='False Negatives', color='#f39c12')
    
    # Etichette
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                        f'{int(h)}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([r['display_name'] for r in top_results], rotation=30, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Analisi Errori: Top 8 Approcci', fontweight='bold')
    ax.legend(loc='upper right')
    
    # Linea ground truth
    ax.axhline(y=306, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(len(top_results)-0.5, 310, 'Ground Truth: 306 match', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_time_vs_performance(results, output_dir):
    """
    Trade-off Tempo vs Performance (F1).
    """
    print("  [6/6] Time vs Performance...")
    
    # Filtra risultati con timing
    timed = [r for r in results if r.get('total_time', 0) > 0 and r['f1'] > 0.1]
    
    if not timed:
        print("    No timing data, skipping...")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for r in timed:
        color = COLORS.get(r['blocking'], COLORS['blocking_only'])
        marker = 'o' if r['category'] == 'Auto-Blocking' else 's'
        
        ax.scatter(r['total_time'], r['f1'], c=color, s=150, marker=marker,
                   edgecolors='black', linewidth=1, alpha=0.85, zorder=5)
        
        ax.annotate(r['display_name'], (r['total_time'], r['f1']),
                    fontsize=8, xytext=(8, 0), textcoords='offset points')
    
    ax.set_xlabel('Tempo Totale (secondi)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Trade-off: Tempo vs Performance', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0, 1.05)
    
    # Evidenzia zona ottimale (alto F1, basso tempo)
    ax.axhspan(0.85, 1.0, alpha=0.1, color='green', label='High Performance Zone')
    
    # Legenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['auto'], 
                   markersize=10, label='Auto-Blocking', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['B1'], 
                   markersize=10, label='Manual-Blocking', markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_time_vs_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_table(results, output_dir):
    """Genera tabella riassuntiva in Markdown."""
    print("  Generating summary table...")
    
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    lines = [
        "# Record Linkage - Riepilogo Risultati",
        "",
        "## Classifica per F1 Score",
        "",
        "| Rank | Approccio | Categoria | Precision | Recall | F1 | TP | FP | FN |",
        "|:----:|-----------|-----------|:---------:|:------:|:--:|:--:|:--:|:--:|"
    ]
    
    for i, r in enumerate(sorted_results, 1):
        f1_display = f"**{r['f1']:.3f}**" if r['f1'] > 0.85 else f"{r['f1']:.3f}"
        lines.append(
            f"| {i} | {r['display_name']} | {r['category']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {f1_display} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} |"
        )
    
    lines.extend([
        "",
        "## Conclusioni",
        "",
        "1. **Auto-Blocking ML** (Dedupe) raggiunge le migliori performance (F1 ≈ 0.92)",
        "2. **P3 (minimal_fast)** è la pipeline migliore: pochi campi = meno rumore",
        "3. **Manual-Blocking** ha recall più alta ma precision inferiore",
        "4. **Blocking Solo** garantisce recall ~1.0 ma troppi falsi positivi",
    ])
    
    with open(os.path.join(output_dir, 'results_summary.md'), 'w') as f:
        f.write('\n'.join(lines))


def main():
    output_dir = 'appoggio/dedupe/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Record Linkage - Visualizzazione Risultati")
    print("=" * 60)
    
    print("\nCaricamento risultati...")
    results = load_all_results()
    print(f"  Trovati {len(results)} esperimenti")
    
    if not results:
        print("ERRORE: Nessun risultato trovato!")
        return
    
    # Riepilogo
    categories = defaultdict(int)
    for r in results:
        categories[r['category']] += 1
    print("\nDistribuzione:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    
    print("\nGenerazione grafici...")
    
    plot_main_comparison(results, output_dir)
    plot_pipeline_heatmap(results, output_dir)
    plot_precision_recall_tradeoff(results, output_dir)
    plot_auto_blocking_focus(results, output_dir)
    plot_error_analysis(results, output_dir)
    plot_time_vs_performance(results, output_dir)
    generate_summary_table(results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Output salvato in: {output_dir}")
    print("\nFile generati:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
