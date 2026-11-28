"""Run the whole visual-analytics workflow end to end (about 4 minutes).

    python src/run_all.py

Order follows the unrolled workflow (V0):
  1. Parikshith  - statistics, convergence, data quality (V1-V3b), transformation (V9), synthesis (V14)
  2. Sindhoor    - PCA, clustering, anomalies (iteration 1); DTW trajectory clusters (iteration 2)
  3. Varshith    - supervised models iteration 1 vs 2, forecast + conformal intervals
  4. Varshith    - figures V7, V10-V12b, drift chart V13
"""
import time

import p1_parikshith_convergence_quality
import p1_parikshith_stats
import p2_sindhoor_unsupervised
import p2_sindhoor_figures
import p3_varshith_drift
import p3_varshith_plots
import p3_varshith_supervised

if __name__ == "__main__":
    t = time.time()
    for name, fn in [("Parikshith: stats/EDA/feedback/drift", p1_parikshith_stats.main),
                     ("Parikshith: convergence + data quality", p1_parikshith_convergence_quality.main),
                     ("Sindhoor: unsupervised iterations 1-2", p2_sindhoor_unsupervised.main),
                     ("Varshith: supervised + forecast", p3_varshith_supervised.main),
                     ("Sindhoor: figures", p2_sindhoor_figures.main),
                     ("Varshith: figures", p3_varshith_plots.main),
                     ("Varshith: drift chart", p3_varshith_drift.main),
                     ("Synthesis figure V14", p1_parikshith_stats.fig_v14_synthesis)]:
        print("\n" + "#" * 80 + f"\n# {name}\n" + "#" * 80)
        fn()
    print(f"\nAll done in {time.time() - t:.0f} s. Figures: images/, tables: outputs/")
