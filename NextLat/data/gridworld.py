"""
Gridworld environment for NextLat-for-safety Experiment 0.

A small N x N grid. Each example is a random walk that the model is trained to
continue (next-token prediction over move directions), plus two prefix slots and
a terminal slot used to define probing labels:

    [start_cell] [g{G}] d_1 d_2 ... d_L [t{G}] end
       idx 0       r=1   <-- directions -->  T   T+1

Two ground-truth per-token labels are attached for probing (NOT used in training):

  * Expressed (behaviorally coupled) -- deterministic from the agent's current
    cell and the FIXED trap map, so they test whether position is decodable:
      y_exp_1[t] = 1 iff current cell is within Manhattan distance 1 of a trap
                   (trivially local floor)
      y_exp_k[t] = 1 iff current cell is within Manhattan distance k of a trap
                   (k <= mtp_horizon; the multi-step-lookahead label)
    Being near an irreversible "trap" region shifts the near-future token
    distribution (the walk is more likely to terminate soon), so this is
    "behaviorally expressed".

  * Decoupled (behaviorally independent) y_dec = G, a secret bit revealed as the
    token g{G} at position r=1. The walk is sampled with NO reference to G; G only
    selects the terminal token t{G} emitted at position T (just before `end`).
    Mid-trajectory, G is in context but irrelevant to the next ~d tokens. We probe
    G ONLY inside the window  W = { t : r + d < t < T - d }  (d = mtp_horizon),
    the "carried-latent dead zone".

Strict leakage control (the central validity requirement): the entire path
(start, walk, traps, length) is sampled BEFORE G, and G ~ Bernoulli(1/2) is drawn
independently, so G is independent of everything except the terminal token.
`leakage_audit` asserts this.

This module is imported by both the training datamodule (`GridworldDataModule`,
registered as "gridworld" in train.py) and the probe script
(`experiments/probe_exp0.py`), which calls `generate_records` / `make_tokenizer`
directly to obtain per-token labels.
"""

import numpy as np
import torch

# NOTE: `lightning` and `data.utils` (which imports `datasets`) are imported
# lazily inside GridworldDataModule so that the generator + leakage audit can be
# imported/run with only numpy + torch (e.g. from experiments/probe_exp0.py or
# the __main__ smoke test) without the full training stack installed.

PAD_TOKEN_ID = 0

# Fixed, data-independent vocabulary so token ids are stable across runs/splits.
_DIRS = ["N", "S", "E", "W"]
_DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def _build_vocab(grid_size):
    """Deterministic id assignment: pad, end, dirs, g0/g1, t0/t1, cells, then
    wall-pattern observation tokens wp0..wp15 (appended last so all earlier ids are
    unchanged vs the fully-observed vocab — v1/v2 checkpoints stay compatible)."""
    words = ["<pad>", "end"] + _DIRS + ["g0", "g1", "t0", "t1"]
    words += [f"c{i}" for i in range(grid_size * grid_size)]
    words += [f"wp{i}" for i in range(16)]  # partial-observability (v2b) only
    word_to_id = {w: i for i, w in enumerate(words)}
    assert word_to_id["<pad>"] == PAD_TOKEN_ID
    return word_to_id


def _wall_pattern(cell, grid_size, traps):
    """4-bit local observation: which of N,S,E,W are blocked (grid edge or trap).
    The aliased observation that, integrated over a trajectory, localizes position."""
    r, c = cell // grid_size, cell % grid_size
    bits = 0
    for b, d in enumerate(_DIRS):  # N,S,E,W -> bits 0..3
        dr, dc = _DELTA[d]
        nr, nc = r + dr, c + dc
        blocked = not (0 <= nr < grid_size and 0 <= nc < grid_size) or (
            nr * grid_size + nc in traps
        )
        if blocked:
            bits |= 1 << b
    return bits


class GridworldTokenizer:
    """Minimal fixed-vocab tokenizer matching the SimpleTokenizer interface used
    by ConstantLengthDataset / initialize_model (needs encode/decode, word_to_id,
    eos_token_id, pad_token_id)."""

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.word_to_id = _build_vocab(grid_size)
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.pad_token_id = PAD_TOKEN_ID
        self.eos_token_id = self.word_to_id["end"]

    def encode(self, sentence):
        return [
            self.word_to_id.get(w, self.pad_token_id) for w in sentence.split()
        ]

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        token_ids = np.atleast_1d(np.asarray(token_ids))
        return " ".join(
            self.id_to_word[int(i)] for i in token_ids if int(i) != self.pad_token_id
        )


def make_tokenizer(grid_size):
    return GridworldTokenizer(grid_size)


# --------------------------------------------------------------------------- #
# Trap map + trajectory generation
# --------------------------------------------------------------------------- #
def make_traps(grid_size, trap_frac, map_seed):
    """Sample a FIXED set of trap cells, shared across all trajectories so the
    model can learn the map. Deterministic in map_seed and independent of G."""
    rng = np.random.default_rng(map_seed)
    n_cells = grid_size * grid_size
    n_traps = max(1, int(round(trap_frac * n_cells)))
    traps = set(rng.choice(n_cells, size=n_traps, replace=False).tolist())
    return frozenset(traps)


def _trap_distance_field(grid_size, traps):
    """Manhattan distance from every cell to the nearest trap. Precomputed once."""
    trap_rc = [(t // grid_size, t % grid_size) for t in traps]
    field = np.empty(grid_size * grid_size, dtype=np.int32)
    for idx in range(grid_size * grid_size):
        r, c = idx // grid_size, idx % grid_size
        field[idx] = min(abs(r - tr) + abs(c - tc) for tr, tc in trap_rc)
    return field


def _valid_moves(cell, grid_size, traps):
    """In-grid neighbours that are NOT traps. Traps are forbidden cells the walk
    routes around; being adjacent to one removes a move, which is what makes
    trap-proximity behaviorally expressed in the next-token distribution."""
    r, c = cell // grid_size, cell % grid_size
    out = []
    for d in _DIRS:
        dr, dc = _DELTA[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            nxt = nr * grid_size + nc
            if nxt not in traps:
                out.append((d, nxt))
    return out


def _generate_one(rng_path, rng_g, params, traps, dist_field):
    """Generate a single trajectory record.

    LEAKAGE CONTROL: the entire path is drawn from rng_path with no reference to
    G; G is drawn afterward from rng_g and only sets the g/t tokens.
    """
    grid_size = params["grid_size"]
    max_steps = params["max_steps"]
    min_steps = params["min_steps"]
    k = params["exp_k"]

    n_cells = grid_size * grid_size
    # valid start = non-trap cell with at least one legal move
    non_trap = [
        i for i in range(n_cells) if i not in traps and _valid_moves(i, grid_size, traps)
    ]

    # 1) sample path (independent of G); fixed length, never terminates early
    #    unless the walk gets boxed in (rare; filtered downstream).
    start = int(rng_path.choice(non_trap))
    cells = [start]            # cell after consuming each direction
    dirs = []
    cur = start
    target_len = int(rng_path.integers(min_steps, max_steps + 1))
    for _ in range(target_len):
        moves = _valid_moves(cur, grid_size, traps)
        if not moves:           # boxed in (no legal move) -> stop early
            break
        d, nxt = moves[int(rng_path.integers(len(moves)))]
        dirs.append(d)
        cur = nxt
        cells.append(cur)
    L = len(dirs)
    got_stuck = int(L < target_len)

    # 2) sample secret bit AFTER the path is fully determined
    G = int(rng_g.integers(2))

    # 3) assemble tokens + per-position cell + label positions, per observability mode
    if params.get("partial_obs"):
        # Partial observability (v2b): NO absolute start token. Interleave a local
        # wall-pattern observation with each move, so position must be INTEGRATED
        # from the observation+action history (a genuine belief-state task):
        #   g{G}  wp(c0) d1  wp(c1) d2 ... dL  wp(cL)  t{G}  end
        # Label position only at the observation tokens (where the model has just
        # observed and should "know where it is").
        tokens = [f"g{G}"]
        cell_at_pos = [cells[0]]            # g position (not labelled)
        label_positions = []
        for i in range(L + 1):
            tokens.append(f"wp{_wall_pattern(cells[i], grid_size, traps)}")
            cell_at_pos.append(cells[i])
            label_positions.append(len(tokens) - 1)
            if i < L:
                tokens.append(dirs[i])
                cell_at_pos.append(cells[i + 1])
        tokens += [f"t{G}", "end"]
        cell_at_pos += [cells[L], cells[L]]
        r = 0                              # position of g{G}
        T = len(tokens) - 2                # position of t{G}
    else:
        # Fully observed (v1/v2): start cell given; position is a simple function of
        # start + cumulative moves.
        tokens = [f"c{start}", f"g{G}"] + dirs + [f"t{G}", "end"]
        r = 1
        T = 2 + L
        cell_at_pos = (
            [start, start] + [cells[i + 1] for i in range(L)] + [cells[L], cells[L]]
        )
        label_positions = list(range(2, 2 + L))   # direction positions
    seqlen = len(tokens)
    assert len(cell_at_pos) == seqlen

    y_exp_1 = np.full(seqlen, -100, dtype=np.int64)
    y_exp_k = np.full(seqlen, -100, dtype=np.int64)
    # y_cell: exact cell id (multiclass) — a HARD expressed/localization target.
    y_cell = np.full(seqlen, -100, dtype=np.int64)
    for p in label_positions:
        dist = int(dist_field[cell_at_pos[p]])
        y_exp_1[p] = int(dist <= 1)
        y_exp_k[p] = int(dist <= k)
        y_cell[p] = int(cell_at_pos[p])

    # decoupled-probe window W = { p : r+d < p < T-d }
    d = params["horizon"]
    in_window = np.zeros(seqlen, dtype=np.int64)
    lo, hi = r + d, T - d
    if hi - lo > 1:
        in_window[lo + 1 : hi] = 1

    return {
        "tokens": tokens,
        "y_exp_1": y_exp_1,
        "y_exp_k": y_exp_k,
        "y_cell": y_cell,
        "y_dec": G,
        "in_window": in_window,
        "r": r,
        "T": T,
        "length": L,
        "start": start,
        "seqlen": seqlen,
        # audit-only fields
        "got_stuck": got_stuck,
    }


def generate_records(n, seed, params, traps=None):
    """Generate `n` trajectory records. Returns (records, traps, tokenizer)."""
    grid_size = params["grid_size"]
    if traps is None:
        traps = make_traps(grid_size, params["trap_frac"], params["map_seed"])
    dist_field = _trap_distance_field(grid_size, traps)
    # Independent rng streams for path vs secret bit -> guarantees G _||_ path.
    rng_path = np.random.default_rng(seed)
    rng_g = np.random.default_rng(seed + 10_000_019)  # disjoint stream
    records = [
        _generate_one(rng_path, rng_g, params, traps, dist_field) for _ in range(n)
    ]
    tokenizer = make_tokenizer(grid_size)
    return records, traps, tokenizer


def params_from_config(config):
    """Pull gridworld params from config.data with sensible defaults."""
    d = config.data
    # Window half-width is a property of the EVAL DATA, fixed at the NextLat
    # rollout horizon, so the probe window W is identical across both arms
    # (the GPT arm's model.mtp_horizon is 1 and must NOT change W).
    horizon = int(getattr(d, "window_d", 8) or 8)
    return {
        "grid_size": int(getattr(d, "grid_size", 9)),
        "trap_frac": float(getattr(d, "trap_frac", 0.04)),
        "map_seed": int(getattr(d, "map_seed", 7)),
        "min_steps": int(getattr(d, "min_steps", 32)),
        "max_steps": int(getattr(d, "max_steps", 72)),
        "exp_k": int(getattr(d, "exp_k", 3)),
        "horizon": horizon,
        "partial_obs": bool(getattr(d, "partial_obs", False)),
        "n_train": int(getattr(d, "n_train", 120000)),
        "n_val": int(getattr(d, "n_val", 8000)),
    }


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
def leakage_audit(records, atol=0.03, raise_on_fail=True):
    """Assert G is independent of path confounds. Returns a stats dict."""
    G = np.array([rec["y_dec"] for rec in records], dtype=np.float64)
    length = np.array([rec["length"] for rec in records], dtype=np.float64)
    start = np.array([rec["start"] for rec in records], dtype=np.float64)
    stuck = np.array([rec["got_stuck"] for rec in records], dtype=np.float64)
    exp_pos = np.array(
        [int((rec["y_exp_k"] == 1).sum()) for rec in records], dtype=np.float64
    )

    def _corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    stats = {
        "n": len(records),
        "p_G1": float(G.mean()),
        "corr_G_length": _corr(G, length),
        "corr_G_start": _corr(G, start),
        "corr_G_stuck": _corr(G, stuck),
        "corr_G_exp_pos": _corr(G, exp_pos),
        "frac_stuck": float(stuck.mean()),
        "mean_length": float(length.mean()),
        "frac_with_window": float(
            np.mean([rec["in_window"].sum() > 0 for rec in records])
        ),
        "exp_k_base_rate": float(
            np.mean(
                np.concatenate(
                    [rec["y_exp_k"][rec["y_exp_k"] >= 0] for rec in records]
                )
            )
        ),
    }
    failures = []
    if abs(stats["p_G1"] - 0.5) > atol:
        failures.append(f"G not balanced: p(G=1)={stats['p_G1']:.3f}")
    for key in ("corr_G_length", "corr_G_start", "corr_G_stuck", "corr_G_exp_pos"):
        if abs(stats[key]) > atol:
            failures.append(f"{key}={stats[key]:.3f} exceeds atol={atol}")
    stats["failures"] = failures
    if failures and raise_on_fail:
        raise AssertionError("Leakage audit FAILED: " + "; ".join(failures))
    return stats


# --------------------------------------------------------------------------- #
# Training dataset / datamodule (mirrors ManhattanDataModule)
# --------------------------------------------------------------------------- #
class GridworldDataset(torch.utils.data.Dataset):
    def __init__(self, token_id_lists):
        self.data = token_id_lists

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for sample in self.data:
            yield sample

    def shard(self, data_size_per_shard, index):
        start = index * data_size_per_shard
        end = (index + 1) * data_size_per_shard
        self.data = self.data[start:end]


class GridworldDataModule:
    def __init__(self, fabric, config):
        from data.utils import ConstantLengthDataset

        self.fabric = fabric
        self.config = config
        self.batch_size = config.data.device_batch_size
        self.params = params_from_config(config)

        fabric.print(f"[gridworld] params: {self.params}")
        # Generate train/val records (deterministic in config.seed).
        with self.fabric.rank_zero_first(local=True):
            train_recs, traps, tokenizer = generate_records(
                self.params["n_train"], int(config.seed), self.params
            )
            val_recs, _, _ = generate_records(
                self.params["n_val"], int(config.seed) + 1, self.params, traps=traps
            )

        # Validity / leakage sanity at construction time (cheap insurance).
        audit = leakage_audit(train_recs, raise_on_fail=True)
        fabric.print(f"[gridworld] leakage audit (train): {audit}")

        self.tokenizer = tokenizer
        self.traps = traps
        self.vocab_size = len(tokenizer.word_to_id)
        fabric.print(f"[gridworld] vocab size: {self.vocab_size}")

        train_tokens = [tokenizer.encode(" ".join(r["tokens"])) for r in train_recs]
        val_tokens = [tokenizer.encode(" ".join(r["tokens"])) for r in val_recs]

        train_dataset = GridworldDataset(train_tokens)
        val_dataset = GridworldDataset(val_tokens)

        # Shard across devices before packing (no-op for single GPU).
        data_size_per_shard = len(train_tokens) // self.fabric.world_size
        train_dataset.shard(data_size_per_shard, self.fabric.global_rank)

        self.train_dataset = ConstantLengthDataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            formatting_func=lambda x: x,
            seq_length=config.model.block_size,
            append_concat_token=False,  # each trajectory already ends with `end`
            pretokenized=True,
            no_invalid_starts=True,
        )
        self.val_dataset = ConstantLengthDataset(
            dataset=val_dataset,
            tokenizer=tokenizer,
            formatting_func=lambda x: x,
            seq_length=config.model.block_size,
            append_concat_token=False,
            pretokenized=True,
            no_invalid_starts=True,
        )

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dl)

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=False,  # keep partial final batch so small val sets still yield >=1 batch
        )
        return self.fabric.setup_dataloaders(dl)

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size
        config.model.context_length = 2  # start cell + secret-reveal token

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_batch(self, batch):
        return batch["input_ids"]


if __name__ == "__main__":
    # Smoke test + leakage audit, runnable as: python -m data.gridworld
    cfg_params = {
        "grid_size": 9,
        "trap_frac": 0.04,
        "map_seed": 7,
        "min_steps": 32,
        "max_steps": 72,
        "exp_k": 3,
        "horizon": 8,
    }
    recs, traps, tok = generate_records(5000, seed=0, params=cfg_params)
    print(f"vocab_size={len(tok.word_to_id)}  n_traps={len(traps)}")
    ex = recs[0]
    print("example tokens:", " ".join(ex["tokens"]))
    print("  r,T,length,G:", ex["r"], ex["T"], ex["length"], ex["y_dec"])
    print("  window size:", int(ex["in_window"].sum()))
    stats = leakage_audit(recs, raise_on_fail=True)
    print("LEAKAGE AUDIT PASSED:")
    for kk, vv in stats.items():
        print(f"  {kk}: {vv}")
