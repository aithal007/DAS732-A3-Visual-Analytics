"""Run the whole visual-analytics workflow end to end (about 5 minutes).

    python src/run_all.py

Three workflows answer one main question (Fig. V0); each is an unrolled Keim loop with its own
feedback loop (Figs. W1-W3).  Order of execution:
  W1  Parikshith - statistics on raw data (V1-V3), the shared transformation (V9),
                   iteration 1 map (V1d), iteration 2 in aligned time, headroom
                   and the leapfrog test (V1c, V1e, V1f)
  W2  Sindhoor   - snapshot mining (V4-V6), stability and anomaly evidence (V5b, V6b, V6c),
                   trajectories under DTW (V8, V8a, V8d)
  W3  Varshith   - baseline vs transformed models, forecast, conformal intervals, drift (V7, V10-V13)
  K*  synthesis V14, then the workflow diagrams (V0, W1-W3) are regenerated from the new figures
"""
import time

import p1_parikshith_convergence_quality
import p1_parikshith_evidence
import p1_parikshith_stats
import p2_sindhoor_evidence
import p2_sindhoor_figures
import p2_sindhoor_unsupervised
import p3_varshith_drift
import p3_varshith_plots
import p3_varshith_supervised

if __name__ == "__main__":
    t = time.time()
    for name, fn in [("W1 Parikshith: statistics, transformation T (V1-V3, V9)", p1_parikshith_stats.main),
                     ("W1 Parikshith: data quality, aligned time, headroom (V1c)",
                      p1_parikshith_convergence_quality.main),
                     ("W1 Parikshith: take-off map, headroom, leapfrog test (V1d, V1e, V1f)",
                      p1_parikshith_evidence.main),
                     ("W2 Sindhoor: unsupervised iterations 1-2", p2_sindhoor_unsupervised.main),
                     ("W3 Varshith: supervised + forecast", p3_varshith_supervised.main),
                     ("W2 Sindhoor: figures V4-V8", p2_sindhoor_figures.main),
                     ("W2 Sindhoor: evidence figures V5b, V6b, V6c, V8a, V8d", p2_sindhoor_evidence.main),
                     ("W3 Varshith: figures V7, V10-V12b", p3_varshith_plots.main),
                     ("W3 Varshith: drift chart V13", p3_varshith_drift.main),
                     ("K* synthesis figure V14", p1_parikshith_stats.fig_v14_synthesis)]:
        print("\n" + "#" * 80 + f"\n# {name}\n" + "#" * 80)
        fn()
    print(f"\nAll done in {time.time() - t:.0f} s. Figures: images/, tables: outputs/")
    print("Diagrams: python src/build_diagrams.py  (regenerates V0, W1-W3 with the new thumbnails)")
