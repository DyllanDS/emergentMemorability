"""
SPoSE (Zheng, Pereira, Baker & Hebart, 2019; Hebart et al., 2020) implemented
in PyTorch, to refit the embedding from the THINGS odd-one-out triplets at a
range of dimensionalities.

Model
  W  : 1854 x D, non-negative
  For a triplet stored as (i, j, k) (the THINGS-data triplet files, osf.io/f5rn6,
  list the pair the participant kept together first and the odd one out last),
  the three pairwise similarities are dot products, and the choice probability is
  a softmax over them:

      s_ij = w_i . w_j     (the pair the participant kept together)
      s_ik = w_i . w_k
      s_jk = w_j . w_k
      p(chosen = ij) = softmax([s_ij, s_ik, s_jk])[0]

  loss = mean(-log p) + (lambda / n_items) * ||W||_1

  Non-negativity is enforced by clamping after each optimizer step, and the
  dimensionality of the fitted model is the number of columns with any weight
  above 0.1 (Hebart et al.'s own criterion).

lambda sets the sparsity and so the number of surviving dimensions (49 for the
1.46 million triplets of Hebart et al., 2020; 66 for the 4.6 million triplets of
Hebart et al., 2023).

Module A6b (Fig. S16). The 14 refits of Fig. S16 come from this program (PyTorch
on an Apple M2 Pro GPU, about 3 hours in all) and ship as saved outputs in
A6b_embedding_size_figS16/data/refits/, because training on a GPU does not give
exactly the same weights twice. The triplets are read from the two text files of
osf.io/f5rn6 (trainset.txt, 4,120,663 triplets; validationset.txt, 453,642
triplets; 0-based concept indices in THINGS order, the pair the participant kept
together first and the odd one out last) in the folder given by --triplets
(default: external/spose_triplets, filled by
python external/download_external.py --spose-triplets). Needs PyTorch. The
commands that made the 14 refits are REFIT_COMMANDS in
A6b_embedding_size_figS16/code/run_module.py; run_module.py --refit runs them.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

N_ITEMS = 1854


def surviving_dims(W, thresh=0.1):
    return int((W.max(0) > thresh).sum())


def prune_and_sort(W, thresh=0.1):
    keep = W.max(0) > thresh
    W = W[:, keep]
    return W[:, np.argsort(-W.sum(0))]


def triplet_loss(W, batch):
    a = W[batch[:, 0]]
    b = W[batch[:, 1]]
    c = W[batch[:, 2]]
    s_ab = (a * b).sum(1)
    s_ac = (a * c).sum(1)
    s_bc = (b * c).sum(1)
    logits = torch.stack([s_ab, s_ac, s_bc], 1)
    return torch.nn.functional.cross_entropy(
        logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))


@torch.no_grad()
def evaluate(W, triplets, device, chunk=200_000):
    ce, acc, n = 0.0, 0, 0
    for s in range(0, len(triplets), chunk):
        batch = triplets[s:s + chunk].to(device)
        a, b, c = W[batch[:, 0]], W[batch[:, 1]], W[batch[:, 2]]
        logits = torch.stack([(a * b).sum(1), (a * c).sum(1), (b * c).sum(1)], 1)
        tgt = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        ce += torch.nn.functional.cross_entropy(logits, tgt, reduction="sum").item()
        acc += (logits.argmax(1) == 0).sum().item()
        n += logits.shape[0]
    return ce / n, acc / n


def train_spose(train, val, lmbda, init_dim=100, epochs=400, batch_size=8192,
                lr=1e-3, seed=0, device="mps", log_every=25, log_fh=None,
                patience=60, use_final=True):
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)

    W = torch.nn.Parameter((torch.rand(N_ITEMS, init_dim, generator=g) * 0.1).to(dev))
    opt = torch.optim.Adam([W], lr=lr)

    # whole dataset lives on the device: the model is tiny, so per-step host
    # transfer dominates the runtime otherwise
    train_t = torch.from_numpy(train.astype(np.int64)).to(dev)
    val_t = torch.from_numpy(val.astype(np.int64))
    n = train_t.shape[0]
    steps = n // batch_size

    best = dict(val_ce=np.inf, epoch=-1, W=None, dim=None)
    last = dict(best)
    traj = []
    since_best = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g).to(dev)
        for s in range(steps):
            batch = train_t[perm[s * batch_size:(s + 1) * batch_size]]
            loss = triplet_loss(W, batch) + (lmbda / N_ITEMS) * W.abs().sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                W.clamp_(min=0.0)

        if ep % 5 == 0 or ep == epochs:
            val_ce, val_acc = evaluate(W.detach(), val_t, dev)
            Wn = W.detach().cpu().numpy()
            dim = surviving_dims(Wn)
            traj.append((ep, dim, val_ce, val_acc))
            last = dict(val_ce=val_ce, val_acc=val_acc, epoch=ep, W=Wn.copy(), dim=dim)
            if val_ce < best["val_ce"] - 1e-5:
                best = dict(val_ce=val_ce, val_acc=val_acc, epoch=ep,
                            W=Wn.copy(), dim=dim)
                since_best = 0
            else:
                since_best += 5
            if ep % log_every == 0 or ep == epochs:
                msg = (f"    lam={lmbda:<9.5g} ep {ep:4d}  dim {dim:3d}  "
                       f"val CE {val_ce:.4f}  val acc {val_acc:.4f}  "
                       f"[{time.time()-t0:.0f}s]")
                print(msg, flush=True)
                if log_fh:
                    log_fh.write(msg + "\n")
                    log_fh.flush()
            if use_final:
                # L1 keeps pruning after validation CE has flattened, so the
                # dimensionality (not the fit) is what we let converge
                if len(traj) > 12 and all(t[1] == traj[-1][1] for t in traj[-12:]):
                    break
            elif since_best >= patience:
                break
    out = last if use_final else best
    out["traj"] = traj
    return out


def default_triplet_folder():
    """external/spose_triplets of the release (this file is in <release>/A6b_.../code/refit/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "external", "spose_triplets"))


def load_triplets(folder):
    """The training and validation triplets of osf.io/f5rn6 (one odd-one-out trial per line: three
    0-based concept indices, the pair kept together first, the odd one out last)."""
    paths = [os.path.join(folder, "trainset.txt"), os.path.join(folder, "validationset.txt")]
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: put the SPoSE triplet files there "
                             "(python external/download_external.py --spose-triplets; "
                             "EXTERNAL_INPUTS.md M9)")
    train = np.loadtxt(paths[0], dtype=np.int64)
    val = np.loadtxt(paths[1], dtype=np.int64)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--init-dim", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--best-val", action="store_true",
                    help="keep the best-validation weights instead of the final ones")
    ap.add_argument("--triplets", default=default_triplet_folder(),
                    help="folder with trainset.txt and validationset.txt (osf.io/f5rn6)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train, val = load_triplets(args.triplets)
    print(f"train {train.shape}  val {val.shape}  device {args.device}", flush=True)

    summary = []
    log_path = os.path.join(args.outdir, f"{args.tag}_train.log")
    with open(log_path, "a") as fh:
        for lam in args.lambdas:
            best = train_spose(train, val, lam, init_dim=args.init_dim,
                               epochs=args.epochs, batch_size=args.batch_size,
                               lr=args.lr, seed=args.seed, device=args.device,
                               log_fh=fh, use_final=not args.best_val)
            Wp = prune_and_sort(best["W"])
            fn = os.path.join(args.outdir, f"spose_lam{lam:g}_seed{args.seed}_d{Wp.shape[1]}.txt")
            np.savetxt(fn, Wp, fmt="%.6f")
            rec = dict(lmbda=lam, dim=int(Wp.shape[1]), val_ce=best["val_ce"],
                       val_acc=best["val_acc"], epoch=best["epoch"],
                       seed=args.seed, file=os.path.basename(fn),
                       traj=best.get("traj", []))
            summary.append(rec)
            print(f"  DONE lam={lam:g} -> dim {Wp.shape[1]}  "
                  f"val CE {best['val_ce']:.4f}  acc {best['val_acc']:.4f} -> {fn}",
                  flush=True)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            with open(os.path.join(args.outdir, f"{args.tag}_summary.json"), "w") as sf:
                json.dump(summary, sf, indent=1)


if __name__ == "__main__":
    main()
