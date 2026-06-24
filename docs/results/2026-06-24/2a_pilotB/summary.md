# Concealworld (2a) results

arms/seeds: {'gpt': [1234], 'nextlat': [1234]}

## Headline (linear probe — best-layer accuracy mean±std @layer)
label | gpt | nextlat | chance
S_run_wait | 0.435±0.000@3 | 0.459±0.000@6 | 0.253
S_run | 0.421±0.000@2 | 0.445±0.000@6 | 0.252

## Effective rank (final layer, wandering positions) — lower = more compressed
gpt: 133.0±0.0
nextlat: 26.8±0.0

## Decision (auto, from linear S_run_wait accuracy — confirm by eye)
GPT best-layer linear acc (running secret, wait positions) = 0.435
  nextlat: 0.459  (Δ vs GPT = +0.023)
=> no clear gap (mean Δ=+0.023) with GPT off-ceiling: predict-ahead doesn't visibly help; inspect retention + gather-vs-carry before 2b.

## Retention curve — S_run_wait linear acc vs position t [final layer, per-offset trained]
gpt: 1:1.00, 2:1.00, 3:1.00, 4:0.96, 5:0.89, 6:0.81, 7:0.73, 8:0.63, 9:0.54, 10:0.52, 11:0.46, 12:0.38, 13:0.35, 14:0.31, 15:0.33, 16:0.29, 17:0.28, 18:0.32, 19:0.26, 20:0.25, 21:0.23, 22:0.25, 23:0.24
nextlat: 1:1.00, 2:1.00, 3:1.00, 4:0.89, 5:0.82, 6:0.76, 7:0.70, 8:0.63, 9:0.54, 10:0.50, 11:0.45, 12:0.41, 13:0.40, 14:0.37, 15:0.41, 16:0.40, 17:0.39, 18:0.44, 19:0.39, 20:0.41, 21:0.31, 22:0.43, 23:0.46

## Retention curve — S_run linear acc vs position t [final layer, per-offset trained]
gpt: 0:1.00, 1:1.00, 2:1.00, 3:0.98, 4:0.92, 5:0.78, 6:0.69, 7:0.61, 8:0.55, 9:0.47, 10:0.45, 11:0.41, 12:0.35, 13:0.32, 14:0.29, 15:0.30, 16:0.30, 17:0.30, 18:0.28, 19:0.27, 20:0.25, 21:0.25, 22:0.24, 23:0.24
nextlat: 0:1.00, 1:1.00, 2:1.00, 3:0.90, 4:0.81, 5:0.75, 6:0.68, 7:0.63, 8:0.54, 9:0.50, 10:0.47, 11:0.40, 12:0.38, 13:0.36, 14:0.33, 15:0.38, 16:0.37, 17:0.41, 18:0.41, 19:0.38, 20:0.38, 21:0.30, 22:0.44, 23:0.45

## Gather-vs-carry (S_final decodability: last wandering token → fork token)
gpt: last=0.237 fork=0.498 jump=+0.261  (defers/gathers)
nextlat: last=0.450 fork=0.481 jump=+0.031  (carries)

## Availability vs linearization (best-layer linear vs MLP, S_run_wait)
gpt: linear 0.435 | MLP 0.586  (MLP−linear = +0.151)
nextlat: linear 0.459 | MLP 0.662  (MLP−linear = +0.203)

### S_run_wait — linear probe accuracy, mean±std over seeds
layer | gpt | nextlat | chance | shuffled
0 | 0.417±0.000 | 0.449±0.000 | 0.253 | 0.252
1 | 0.422±0.000 | 0.434±0.000 | 0.253 | 0.239
2 | 0.423±0.000 | 0.435±0.000 | 0.253 | 0.242
3 | 0.435±0.000 | 0.445±0.000 | 0.253 | 0.247
4 | 0.430±0.000 | 0.442±0.000 | 0.253 | 0.246
5 | 0.428±0.000 | 0.450±0.000 | 0.253 | 0.247
6 | 0.429±0.000 | 0.459±0.000 | 0.253 | 0.254
7 | 0.427±0.000 | 0.450±0.000 | 0.253 | 0.257
8 | 0.401±0.000 | 0.443±0.000 | 0.253 | 0.266

### S_run — linear probe accuracy, mean±std over seeds
layer | gpt | nextlat | chance | shuffled
0 | 0.416±0.000 | 0.424±0.000 | 0.252 | 0.264
1 | 0.418±0.000 | 0.415±0.000 | 0.252 | 0.274
2 | 0.421±0.000 | 0.418±0.000 | 0.252 | 0.277
3 | 0.420±0.000 | 0.426±0.000 | 0.252 | 0.297
4 | 0.420±0.000 | 0.426±0.000 | 0.252 | 0.292
5 | 0.417±0.000 | 0.435±0.000 | 0.252 | 0.286
6 | 0.417±0.000 | 0.445±0.000 | 0.252 | 0.284
7 | 0.415±0.000 | 0.430±0.000 | 0.252 | 0.277
8 | 0.399±0.000 | 0.421±0.000 | 0.252 | 0.293

### S_run_wait — mlp probe accuracy, mean±std over seeds
layer | gpt | nextlat | chance | shuffled
0 | 0.522±0.000 | 0.525±0.000 | 0.253 | —
1 | 0.568±0.000 | 0.660±0.000 | 0.253 | —
2 | 0.586±0.000 | 0.657±0.000 | 0.253 | —
3 | 0.574±0.000 | 0.654±0.000 | 0.253 | —
4 | 0.566±0.000 | 0.661±0.000 | 0.253 | —
5 | 0.562±0.000 | 0.662±0.000 | 0.253 | —
6 | 0.566±0.000 | 0.639±0.000 | 0.253 | —
7 | 0.553±0.000 | 0.635±0.000 | 0.253 | —
8 | 0.553±0.000 | 0.631±0.000 | 0.253 | —
