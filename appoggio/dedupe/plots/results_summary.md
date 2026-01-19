# Record Linkage - Riepilogo Risultati

## Classifica per F1 Score

| Rank | Approccio | Categoria | Precision | Recall | F1 | TP | FP | FN |
|:----:|-----------|-----------|:---------:|:------:|:--:|:--:|:--:|:--:|
| 1 | P3 Extended | Auto-Blocking | 0.925 | 0.915 | **0.920** | 504 | 41 | 47 |
| 2 | P3 Minimal Fast | Auto-Blocking | 0.925 | 0.915 | **0.920** | 504 | 41 | 47 |
| 3 | P2 Plus Location | Auto-Blocking | 0.833 | 0.817 | 0.825 | 450 | 90 | 101 |
| 4 | P1 Textual Core | Auto-Blocking | 0.797 | 0.771 | 0.784 | 424 | 108 | 126 |
| 5 | P3 + B1 | Manual-Blocking | 0.663 | 0.932 | 0.775 | 551 | 280 | 40 |
| 6 | P3 + B1 | Manual-Blocking | 0.583 | 0.936 | 0.718 | 500 | 358 | 34 |
| 7 | P2 + B1 | Manual-Blocking | 0.618 | 0.808 | 0.700 | 445 | 275 | 106 |
| 8 | P1 + B1 | Manual-Blocking | 0.614 | 0.780 | 0.688 | 430 | 270 | 121 |
| 9 | P3 + Union | Manual-Blocking | 0.477 | 0.702 | 0.568 | 387 | 425 | 164 |
| 10 | P3 + B2 | Manual-Blocking | 0.540 | 0.584 | 0.562 | 322 | 425 | 229 |
| 11 | P2 + Union | Manual-Blocking | 0.468 | 0.604 | 0.528 | 333 | 378 | 217 |
| 12 | P1 + Union | Manual-Blocking | 0.460 | 0.588 | 0.516 | 324 | 380 | 227 |
| 13 | P2 + B2 | Manual-Blocking | 0.552 | 0.467 | 0.506 | 257 | 209 | 293 |
| 14 | P1 + B2 | Manual-Blocking | 0.652 | 0.324 | 0.433 | 178 | 95 | 372 |
| 15 | Blocking B1 | Blocking Only | 0.272 | 1.000 | 0.428 | 306 | 818 | 0 |
| 16 | Blocking B2 | Blocking Only | 0.231 | 0.974 | 0.373 | 298 | 992 | 8 |
| 17 | Blocking Union | Blocking Only | 0.162 | 1.000 | 0.278 | 306 | 1587 | 0 |
| 18 | P3 + B2 | Manual-Blocking | 0.000 | 0.000 | 0.000 | 0 | 0 | 551 |
| 19 | P3 + Union | Manual-Blocking | 0.000 | 0.000 | 0.000 | 0 | 0 | 551 |

## Conclusioni

1. **Auto-Blocking ML** (Dedupe) raggiunge le migliori performance (F1 circa 0.92)
2. **P3 (minimal_fast)** è la pipeline migliore: pochi campi = meno rumore
3. **Manual-Blocking** ha recall più alta ma precision inferiore
4. **Blocking Solo** garantisce recall circa 1.0 ma troppi falsi positivi