# Dedupe Risultati

## Classifica per F1 Score

| Rank | Approccio | Categoria | Precision | Recall | F1 | TP | FP | FN |
|:----:|-----------|-----------|:---------:|:------:|:--:|:--:|:--:|:--:|
| 1 | P3 Minimal Fast | Auto-Blocking | 0.924 | 0.915 | **0.920** | 280 | 23 | 26 |
| 2 | P2 Plus Location | Auto-Blocking | 0.833 | 0.817 | 0.825 | 250 | 50 | 56 |
| 3 | P1 Textual Core | Auto-Blocking | 0.797 | 0.771 | 0.784 | 236 | 60 | 70 |
| 4 | P3 Extended + B1 | Manual-Blocking | 0.645 | 0.928 | 0.761 | 284 | 156 | 22 |
| 5 | P3 + B1 | Manual-Blocking | 0.591 | 0.938 | 0.725 | 287 | 199 | 19 |
| 6 | P2 + B1 | Manual-Blocking | 0.618 | 0.807 | 0.700 | 247 | 153 | 59 |
| 7 | P1 + B1 | Manual-Blocking | 0.614 | 0.781 | 0.688 | 239 | 150 | 67 |
| 8 | P3 Extended + B2 | Manual-Blocking | 0.541 | 0.585 | 0.562 | 179 | 152 | 127 |
| 9 | P2 + Union | Manual-Blocking | 0.468 | 0.605 | 0.528 | 185 | 210 | 121 |
| 10 | P1 + Union | Manual-Blocking | 0.460 | 0.588 | 0.516 | 180 | 211 | 126 |
| 11 | P2 + B2 | Manual-Blocking | 0.552 | 0.467 | 0.506 | 143 | 116 | 163 |
| 12 | P1 + B2 | Manual-Blocking | 0.651 | 0.324 | 0.432 | 99 | 53 | 207 |
| 13 | Blocking B1 | Blocking Only | 0.272 | 1.000 | 0.428 | 306 | 818 | 0 |
| 14 | Blocking B2 | Blocking Only | 0.231 | 0.974 | 0.373 | 298 | 992 | 8 |
| 15 | Blocking Union | Blocking Only | 0.162 | 1.000 | 0.278 | 306 | 1587 | 0 |
| 16 | P3 + B2 | Manual-Blocking | 0.000 | 0.000 | 0.000 | 0 | 0 | 306 |
| 17 | P3 + Union | Manual-Blocking | 0.000 | 0.000 | 0.000 | 0 | 0 | 306 |
