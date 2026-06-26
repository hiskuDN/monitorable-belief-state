"""Storyworld environment — belief tracking on a (templated) language surface, base rung of the
language track for the monitorable-belief-state program.

A short narrative moves objects around K shelves with RELATIVE moves; the model must track a
target object's CURRENT shelf (a running sum mod K of its deltas) and answer an end query. We
probe whether that belief is linearly decodable from the hidden state mid-narrative, predict-ahead
arms vs a vanilla GPT.

    the ball starts on shelf3 .            <- target reveal (sets the belief; uniform over K)
    the book starts on shelf1 .            <- distractor reveal
    alice moves the ball up two shelves .  <- target move: shelf = (shelf + 2) mod K
    bob moves the book down one shelves .  <- distractor move (target shelf unchanged)
    carol moves it up one shelves .        <- coreference: "it" = previous move's object
    ...
    where is the ball ? shelf2 end         <- query (forces tracking); not in the probe window

Why a RUNNING shelf (concealworld v1->v2 lesson): a single stated location is a surface token a
linear probe reads for free (saturates). A running sum of relative moves is a nonlinear function
of scattered updates, uniform at every position (with a uniform initial shelf), and cannot be
computed-once-carried. The NEW variable vs concealworld is the LANGUAGE SURFACE: light templated
sentences, function words, coreference ("it"), and distractor objects the belief must be carried
across / bound around. Fixed token width per sentence keeps per-position probing + uniformity clean.

Leakage control (`leakage_audit`): the target shelf is uniform over K at every probed position
(uniform initial + uniform deltas, mod K); #target-moves is independent of the final shelf; the
total length is fixed. Positive control: the answer token equals the final running shelf.
"""

import numpy as np
import torch

PAD_TOKEN_ID = 0

_NAMES = ["alice", "bob", "carol", "dave", "erin"]
_OBJS = ["ball", "book", "lamp", "cup", "key", "pen"]
_NUMWORDS = ["one", "two", "three", "four", "five"]


def _build_vocab(n_states, n_objects, n_agents, max_delta):
    """Deterministic id table. Shelf tokens appended LAST for id-stability across K."""
    words = ["<pad>", "end", "the", "starts", "on", "moves", "shelves", ".", "?", "where", "is", "it",
             "up", "down"]
    words += _NAMES[:n_agents]
    words += _OBJS[:n_objects]
    words += _NUMWORDS[:max_delta]
    words += [f"shelf{i}" for i in range(n_states)]
    word_to_id = {w: i for i, w in enumerate(words)}
    assert word_to_id["<pad>"] == PAD_TOKEN_ID
    return word_to_id


class StoryworldTokenizer:
    """Minimal fixed-vocab tokenizer matching the SimpleTokenizer interface."""

    def __init__(self, params):
        self.params = params
        self.word_to_id = _build_vocab(
            params["n_states"], 1 + params["n_distractors"], params["n_agents"], params["max_delta"]
        )
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.pad_token_id = PAD_TOKEN_ID
        self.eos_token_id = self.word_to_id["end"]

    def encode(self, sentence):
        return [self.word_to_id.get(w, self.pad_token_id) for w in sentence.split()]

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        token_ids = np.atleast_1d(np.asarray(token_ids))
        return " ".join(self.id_to_word[int(i)] for i in token_ids if int(i) != self.pad_token_id)


def make_tokenizer(params):
    return StoryworldTokenizer(params)


def _generate_one(rng, params):
    """One narrative. The target object is objs[0]; it is always the queried object (v1)."""
    K = params["n_states"]
    nd = params["n_distractors"]
    nm = params["n_move"]
    maxd = params["max_delta"]
    tdens = params["target_density"]
    coref = params["coref_prob"]
    names = _NAMES[:params["n_agents"]]
    objs = _OBJS[: 1 + nd]
    target = objs[0]

    shelves = {o: int(rng.integers(K)) for o in objs}  # uniform initial shelf per object

    tokens, S, inwin, isupd = [], [], [], []

    def emit(toks, in_window, is_target_update):
        for w in toks:
            tokens.append(w)
            S.append(shelves[target] if in_window else -100)
            inwin.append(1 if in_window else 0)
            isupd.append(1 if is_target_update else 0)

    # target reveal FIRST so the belief is defined + uniform from the start, then distractor reveals
    emit(["the", target, "starts", "on", f"shelf{shelves[target]}", "."], True, False)
    for o in objs[1:]:
        emit(["the", o, "starts", "on", f"shelf{shelves[o]}", "."], True, False)

    prev_obj, n_tupd = None, 0
    for _ in range(nm):
        is_t = (rng.random() < tdens) or nd == 0
        o = target if is_t else objs[1 + int(rng.integers(nd))]
        ag = names[int(rng.integers(len(names)))]
        up = rng.random() < 0.5
        num = int(rng.integers(1, maxd + 1))
        delta = num if up else -num
        slot = "it" if (o == prev_obj and rng.random() < coref) else o
        head = [ag, "moves", slot, "up" if up else "down", _NUMWORDS[num - 1], "shelves"]
        if o == target:
            emit(head, True, True)                 # belief is transitioning across this sentence
            shelves[target] = (shelves[target] + delta) % K
            emit(["."], True, True)                # shelf now updated; sentence ends
            n_tupd += 1
        else:
            emit(head + ["."], True, False)        # distractor: belief carried unchanged (purest read)
        prev_obj = o

    S_final = shelves[target]
    answer_pos = len(tokens) + 5  # index of the answer token within the query below
    emit(["where", "is", "the", target, "?"], False, False)
    emit([f"shelf{S_final}"], False, False)
    emit(["end"], False, False)

    L = len(tokens)
    return {
        "tokens": tokens,
        "S_running": np.array(S, dtype=np.int64),
        "y_dec": int(S_final),
        "in_window": np.array(inwin, dtype=np.int64),
        "act_window": np.array([1 if i == answer_pos else 0 for i in range(L)], dtype=np.int64),
        "is_update_pos": np.array(isupd, dtype=np.int64),
        "r": 0,
        "act_pos": int(answer_pos),
        "L": L,
        "n_updates": n_tupd,
        "seqlen": L,
    }


def generate_records(n, seed, params, traps=None):
    rng = np.random.default_rng(seed)
    records = [_generate_one(rng, params) for _ in range(n)]
    tokenizer = make_tokenizer(params)
    return records, None, tokenizer


def params_from_config(config):
    d = config.data
    return {
        "n_states": int(getattr(d, "n_states", 6)),          # K shelves (modulus)
        "n_distractors": int(getattr(d, "n_distractors", 2)),
        "n_move": int(getattr(d, "n_move", 16)),             # # move sentences (FIXED)
        "max_delta": int(getattr(d, "max_delta", 2)),
        "target_density": float(getattr(d, "target_density", 0.5)),
        "coref_prob": float(getattr(d, "coref_prob", 0.5)),
        "n_agents": int(getattr(d, "n_agents", 3)),
        "n_train": int(getattr(d, "n_train", 300000)),
        "n_val": int(getattr(d, "n_val", 10000)),
    }


def leakage_audit(records, n_states=None, atol=0.03, raise_on_fail=True):
    """Assert the target shelf is uniform over K at EVERY probed (in_window) position, the final
    shelf is balanced and independent of #target-moves. Structure is fixed so in_window columns
    align across records."""
    K = n_states or (int(np.max([r["y_dec"] for r in records])) + 1)
    L = records[0]["L"]
    inwin0 = records[0]["in_window"]
    S_mat = np.stack([r["S_running"][:L] for r in records])  # [n, L]
    S_final = np.array([r["y_dec"] for r in records])
    n_upd = np.array([r["n_updates"] for r in records], dtype=np.float64)

    per_offset_dev = []
    for t in range(L):
        if inwin0[t] != 1:
            continue
        col = S_mat[:, t]
        p = np.bincount(col[col >= 0], minlength=K) / len(records)
        per_offset_dev.append(float(np.max(np.abs(p - 1.0 / K))))
    max_offset_dev = float(np.max(per_offset_dev)) if per_offset_dev else 1.0

    p_final = np.bincount(S_final, minlength=K) / len(records)
    final_dev = float(np.max(np.abs(p_final - 1.0 / K)))

    def _eta_squared(obs, G):
        obs = np.asarray(obs, dtype=np.float64)
        if obs.std() < 1e-9:
            return 0.0
        grand = obs.mean()
        ss_total = float(((obs - grand) ** 2).sum())
        if ss_total < 1e-12:
            return 0.0
        ss_between = sum(len(obs[G == k]) * (obs[G == k].mean() - grand) ** 2 for k in np.unique(G))
        return float(ss_between / ss_total)

    eta_nupd = _eta_squared(n_upd, S_final)

    stats = {
        "n": len(records), "n_states": K, "L": L,
        "max_offset_dev": max_offset_dev, "final_dev": final_dev,
        "eta2_nupdates_vs_final": eta_nupd, "mean_n_updates": float(n_upd.mean()),
    }
    failures = []
    if max_offset_dev > atol:
        failures.append(f"target shelf not uniform per-offset: max_dev={max_offset_dev:.3f}")
    if final_dev > atol:
        failures.append(f"final shelf not balanced: dev={final_dev:.3f}")
    if eta_nupd > atol:
        failures.append(f"#target-moves leaks final shelf: eta2={eta_nupd:.3f}")
    stats["failures"] = failures
    if failures and raise_on_fail:
        raise AssertionError("Leakage audit FAILED: " + "; ".join(failures))
    return stats


class StoryworldDataset(torch.utils.data.Dataset):
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
        self.data = self.data[index * data_size_per_shard:(index + 1) * data_size_per_shard]


class StoryworldDataModule:
    def __init__(self, fabric, config):
        from data.utils import ConstantLengthDataset

        self.fabric = fabric
        self.config = config
        self.batch_size = config.data.device_batch_size
        self.params = params_from_config(config)

        fabric.print(f"[storyworld] params: {self.params}")
        with self.fabric.rank_zero_first(local=True):
            train_recs, _, tokenizer = generate_records(
                self.params["n_train"], int(config.seed), self.params
            )
            val_recs, _, _ = generate_records(self.params["n_val"], int(config.seed) + 1, self.params)

        audit = leakage_audit(train_recs, n_states=self.params["n_states"], raise_on_fail=True)
        fabric.print(f"[storyworld] leakage audit (train): {audit}")

        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_to_id)
        fabric.print(f"[storyworld] vocab size: {self.vocab_size}")

        train_tokens = [tokenizer.encode(" ".join(r["tokens"])) for r in train_recs]
        val_tokens = [tokenizer.encode(" ".join(r["tokens"])) for r in val_recs]

        train_dataset = StoryworldDataset(train_tokens)
        val_dataset = StoryworldDataset(val_tokens)

        data_size_per_shard = len(train_tokens) // self.fabric.world_size
        train_dataset.shard(data_size_per_shard, self.fabric.global_rank)

        self.train_dataset = ConstantLengthDataset(
            dataset=train_dataset, tokenizer=tokenizer, formatting_func=lambda x: x,
            seq_length=config.model.block_size, append_concat_token=False,
            pretokenized=True, no_invalid_starts=True,
        )
        self.val_dataset = ConstantLengthDataset(
            dataset=val_dataset, tokenizer=tokenizer, formatting_func=lambda x: x,
            seq_length=config.model.block_size, append_concat_token=False,
            pretokenized=True, no_invalid_starts=True,
        )

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=1,
            pin_memory=True, drop_last=True,
        )
        return self.fabric.setup_dataloaders(dl)

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=1,
            pin_memory=True, drop_last=False,
        )
        return self.fabric.setup_dataloaders(dl)

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size
        config.model.context_length = 1

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_batch(self, batch):
        return batch["input_ids"]


if __name__ == "__main__":
    p = params_from_config(type("C", (), {"data": type("D", (), {})()})())
    recs, _, tok = generate_records(8000, seed=0, params=p)
    print(f"vocab={len(tok.word_to_id)} K={p['n_states']} L={recs[0]['L']}")
    print("example:", " ".join(recs[0]["tokens"]))
    print(leakage_audit(recs, n_states=p["n_states"]))
