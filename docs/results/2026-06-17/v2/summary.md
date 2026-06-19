# Experiment 0 results

arms/seeds: {'gpt': [1234, 1235, 1236], 'nextlat': [1234, 1235, 1236]}

## Headline (linear probe — primary)
label | gpt | nextlat  (best-layer AUROC mean±std @layer)
y_exp_1 | 1.000±0.000@2 | 1.000±0.000@2
y_exp_k | 0.999±0.000@0 | 1.000±0.000@1
y_dec | 1.000±0.000@0 | 1.000±0.000@0

## y_cell (exact position, 81-way) — best-layer ACCURACY mean±std @layer
arm | best linear acc | chance | per-layer linear acc
gpt | 1.000±0.000@0 | 0.017 | 1.00 0.99 0.99
nextlat | 1.000±0.000@0 | 0.017 | 1.00 1.00 1.00

## Effective rank (final layer) — lower = more compressed
gpt: 19.9±1.0
nextlat: 20.3±0.7

## Decision (auto, from linear AUROC — confirm by eye)
expressed y_exp_k: NextLat 1.000 vs GPT 0.999  (Δ=+0.001)
decoupled y_dec-in-W: NextLat 1.000 vs GPT 1.000  (Δ=+0.000)
=> NextLat NOT better on expressed: core premise looks FALSE (reclassify as capability).

## Decoupled y_dec accuracy vs (t - r) [final layer, linear]
gpt: 9:1.00, 10:1.00, 11:1.00, 12:1.00, 13:1.00, 14:1.00, 15:1.00, 16:1.00, 17:1.00, 18:1.00, 19:1.00, 20:1.00, 21:1.00, 22:1.00, 23:1.00, 24:1.00, 25:1.00, 26:1.00, 27:1.00, 28:1.00, 29:1.00, 30:1.00, 31:1.00, 32:1.00, 33:1.00, 34:1.00, 35:1.00, 36:1.00, 37:1.00, 38:1.00, 39:1.00, 40:1.00, 41:1.00, 42:1.00, 43:1.00, 44:1.00, 45:1.00, 46:1.00, 47:1.00, 48:1.00, 49:1.00, 50:1.00, 51:1.00, 52:1.00, 53:1.00, 54:1.00, 55:1.00, 56:1.00, 57:1.00, 58:1.00, 59:1.00, 60:1.00, 61:1.00, 62:1.00, 63:1.00
nextlat: 9:1.00, 10:1.00, 11:1.00, 12:1.00, 13:1.00, 14:1.00, 15:1.00, 16:1.00, 17:1.00, 18:1.00, 19:1.00, 20:1.00, 21:1.00, 22:1.00, 23:1.00, 24:1.00, 25:1.00, 26:1.00, 27:1.00, 28:1.00, 29:1.00, 30:1.00, 31:1.00, 32:1.00, 33:1.00, 34:1.00, 35:1.00, 36:1.00, 37:1.00, 38:1.00, 39:1.00, 40:1.00, 41:1.00, 42:1.00, 43:1.00, 44:1.00, 45:1.00, 46:1.00, 47:1.00, 48:1.00, 49:1.00, 50:1.00, 51:1.00, 52:1.00, 53:1.00, 54:1.00, 55:1.00, 56:1.00, 57:1.00, 58:1.00, 59:1.00, 60:1.00, 61:1.00, 62:1.00, 63:1.00

### y_exp_1 — linear probe (acc | AUROC), mean±std over seeds
layer | gpt acc | gpt auroc | nextlat acc | nextlat auroc | chance | shuffled_auroc
0 | 0.971±0.006 | 0.993±0.002 | 0.985±0.009 | 0.997±0.004 | 0.888 | 0.520
1 | 0.998±0.001 | 1.000±0.000 | 0.998±0.001 | 1.000±0.000 | 0.888 | 0.567
2 | 0.998±0.002 | 1.000±0.000 | 0.998±0.001 | 1.000±0.000 | 0.888 | 0.543

### y_exp_k — linear probe (acc | AUROC), mean±std over seeds
layer | gpt acc | gpt auroc | nextlat acc | nextlat auroc | chance | shuffled_auroc
0 | 0.981±0.001 | 0.999±0.000 | 0.991±0.009 | 1.000±0.001 | 0.519 | 0.533
1 | 0.975±0.003 | 0.998±0.000 | 0.991±0.003 | 1.000±0.000 | 0.519 | 0.507
2 | 0.974±0.002 | 0.997±0.000 | 0.991±0.003 | 1.000±0.000 | 0.519 | 0.505

### y_cell — linear probe (acc | AUROC), mean±std over seeds
layer | gpt acc | gpt auroc | nextlat acc | nextlat auroc | chance | shuffled_auroc
0 | 1.000±0.000 | nan±nan | 1.000±0.000 | nan±nan | 0.017 | nan
1 | 0.989±0.003 | nan±nan | 0.999±0.000 | nan±nan | 0.017 | nan
2 | 0.989±0.003 | nan±nan | 0.999±0.000 | nan±nan | 0.017 | nan

### y_dec — linear probe (acc | AUROC), mean±std over seeds
layer | gpt acc | gpt auroc | nextlat acc | nextlat auroc | chance | shuffled_auroc
0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.512 | 0.490
1 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.512 | 0.493
2 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.512 | 0.493

### y_exp_k — mlp probe (acc | AUROC), mean±std over seeds
layer | gpt acc | gpt auroc | nextlat acc | nextlat auroc | chance | shuffled_auroc
0 | 0.999±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.519 | nan
1 | 0.978±0.007 | 0.998±0.001 | 0.999±0.001 | 1.000±0.000 | 0.519 | nan
2 | 0.983±0.008 | 0.998±0.001 | 0.999±0.000 | 1.000±0.000 | 0.519 | nan

### y_dec — mlp probe (acc | AUROC), mean±std over seeds
layer | gpt acc | gpt auroc | nextlat acc | nextlat auroc | chance | shuffled_auroc
0 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.512 | nan
1 | 0.999±0.001 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.512 | nan
2 | 0.999±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.512 | nan
