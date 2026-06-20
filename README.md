# sim

Recursive file similarity analysis using a hybrid metric that blends
**Normalized Compression Distance (NCD)** with **entropy-based distances**.

`sim` walks a directory, derives a compact fingerprint for every file, and
scores all pairs. It is a Go port of the original Python `filesim` tool, built
on the pure-Go [`klauspost/compress`](https://github.com/klauspost/compress)
zstd implementation — no cgo, no external libzstd required.

## Install

```sh
go install github.com/xen0bit/sim@latest
```

This installs a `sim` binary into `$(go env GOBIN)` (or `$(go env GOPATH)/bin`).

You can also import the core package directly:

```go
import "github.com/xen0bit/sim/filesim"
```

## Usage

```sh
sim /path/to/dir                                  # table, sorted by hybrid score
sim /path/to/dir --ext .bin,.exe --sort ncd_dict  # filter by extension, re-sort
sim /path/to/dir --format csv > results.csv       # streaming CSV
sim /path/to/dir --workers 8 --verbose            # tune concurrency, log progress
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--ext` | (all) | Comma-separated extensions to include, e.g. `.bin,.exe`. |
| `--max-size` | `10485760` | Skip files larger than N bytes. |
| `--alpha` | `0.5` | Weight for the dictionary NCD term. |
| `--beta` | `0.25` | Weight for the fingerprint NCD term. |
| `--workers` | `NumCPU-1` | Worker pool size. |
| `--sort` | `hybrid` | `hybrid` \| `ncd_dict` \| `ncd_fingerprint` \| `entropy_global` \| `entropy_profile`. |
| `--format` | `table` | `table` \| `csv`. |
| `--verbose` | off | Progress and warnings on stderr. |

`alpha + beta` must be `<= 1.0`.

## The method

For each unordered pair of files the tool computes four component distances and
blends them:

```
hybrid = alpha·ncd_dict
       + beta·ncd_fingerprint
       + gamma·entropy_global
       + gamma·entropy_profile          where gamma = (1 - alpha - beta) / 2
```

### 1. `ncd_dict` — content similarity via dictionary NCD

NCD asks: *if a compressor already "knows" file A, how much extra space does it
need for file B?* Similar files share structure and compress well together;
unrelated files do not.

Rather than the naive concatenation form, `sim` uses zstd **dictionary**
compression. During preprocessing it trains one zstd dictionary per file
(`zstd.BuildDict`) and bakes it into a reusable encoder. For a pair `(a, b)` it
measures the cheaper conditional direction:

```
ncd_dict = min( C(b | dict_a), C(a | dict_b) ) / max( C(a), C(b) )
```

This is closer to the information-theoretic ideal than concatenation, which
relies on the compressor noticing cross-boundary repetition. For files too small
to train a dictionary (fewer than 5 samples), `sim` falls back to the classic
concatenation NCD:

```
ncd_dict = ( C(ab) - min(C(a), C(b)) ) / max( C(a), C(b) )
```

### 2. `ncd_fingerprint` — class similarity via entropy NCD

NCD on raw bytes is blind to *structural class*: two encrypted blobs are
maximally incompressible together, so they look dissimilar even though they are
both "encrypted." To recover that, `sim` builds an **entropy fingerprint** — the
per-256-byte-block Shannon entropy packed into bytes — and runs NCD on the
fingerprints. Two encrypted (or compressed, or random) files have nearly
identical flat ~8 bits/byte profiles, so their fingerprint NCD is low.

### 3. `entropy_global` — overall information density

```
entropy_global = |H(a) - H(b)| / 8
```

where `H` is global Shannon entropy in bits/byte.

### 4. `entropy_profile` — shape of entropy across the file

Mean absolute difference of the per-block entropy profiles (length-normalized),
capturing whether two files have similar internal structure (flat vs. spiky
headers/sections).

## Score bounds

Every component is **normalized to `[0, 1]`**, where **lower means more
similar**:

| Component | Range | Notes |
|-----------|-------|-------|
| `ncd_dict` | `[0, 1]` | Clamped; ~0 identical, ~1 unrelated bytes. |
| `ncd_fingerprint` | `[0, 1]` | Clamped; low for files of the same entropy class. |
| `entropy_global` | `[0, 1]` | `|ΔH| / 8`. |
| `entropy_profile` | `[0, 1]` | Mean abs block-entropy diff / 8. |
| `hybrid` | `[0, 1]` | Convex combination of the above. |

The hybrid score is a convex combination of values in `[0, 1]`, so it is itself
in `[0, 1]`. NCD is an *approximate* metric — values can sit slightly outside
`[0, 1]` before clamping for very short or pathological inputs, which is why the
two NCD terms are explicitly clamped.

### Worked example

A directory with two similar sentences, an unrelated sentence, and two random
4 KB blobs produces roughly:

| Pair | Hybrid | Why |
|------|--------|-----|
| `doc_a` vs `doc_b` | ~0.25 | Nearly identical sentence structure. |
| `enc_a` vs `enc_b` | ~0.69 | Incompressible bytes (`ncd_dict` ~1.0) but matching flat entropy fingerprints rescue them. |
| `doc_a` vs `enc_a` | ~0.84 | Different content *and* very different entropy class. |

The encrypted pair scoring well below the doc-vs-encrypted pairs is exactly the
behaviour the fingerprint and entropy terms are designed to produce.

## Design notes

**Concurrency.** zstd compression is the hot path. `Encoder.EncodeAll` is
safe for concurrent use, so `sim` shares one dictionary-less encoder across all
goroutines and one dictionary-carrying encoder per file (built once during
preprocessing, not per pair). Files are preprocessed by a worker pool, and pairs
are streamed through a second worker pool; results flow out on a channel as they
complete, so fast pairs never wait on slow ones.

**Memory.** After preprocessing, only derived metadata is retained per file
(compressed size, global entropy, entropy profile, fingerprint, and the trained
encoder) — raw bytes are released. Pair computation re-reads file bytes from
disk on demand; the OS page cache makes this cheap for small corpora and trades
memory for I/O on large ones. The pair list itself is generated lazily and never
fully materialized — only `O(N)` metadata is held, not the `O(N²)` pairs.

## Library API

```go
files, _ := filesim.CollectFiles(dir, []string{".bin"}, 10<<20, nil)
metas := filesim.Preprocess(files, filesim.DefaultWorkers(), nil)
for r := range filesim.Pairs(metas, 0.5, 0.25, filesim.DefaultWorkers(), nil) {
    fmt.Println(r.FileA, r.FileB, r.Hybrid)
}
```

## License

See repository.
