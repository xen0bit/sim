// Package filesim provides recursive file similarity analysis using a hybrid
// metric that blends Normalized Compression Distance (NCD) with entropy-based
// distances.
//
// It is a Go port of the original Python filesim tool. The four components of
// the hybrid score are:
//
//   - ncd_dict        : NCD where a zstd dictionary trained on one file is used
//     to compress the other (falls back to concatenation NCD
//     when a file is too small to train a dictionary).
//   - ncd_fingerprint : NCD over the packed per-block entropy "fingerprint",
//     which captures class-level similarity (e.g. two
//     encrypted blobs both look flat at ~8 bits/byte).
//   - entropy_global  : |H(a) - H(b)| / 8.
//   - entropy_profile : mean absolute difference of per-block entropy.
//
// # Concurrency model
//
// klauspost/compress releases nothing to a GIL — Go has none — and its
// Encoder.EncodeAll is safe for concurrent use. We therefore build one zstd
// encoder per file (with its trained dictionary baked in) exactly once during
// preprocessing and share it across every pair goroutine. This avoids the
// O(N^2) dictionary reconstruction the reference Python implementation paid.
//
// # Memory model
//
// After preprocessing we keep only derived metadata per file, not raw bytes.
// Pair computation re-reads files from disk on demand; for small datasets the
// OS page cache makes this nearly free, and for large datasets it trades memory
// for I/O — the correct tradeoff when N is large.
package filesim

import (
	"encoding/binary"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/klauspost/compress/zstd"
)

// Tuning constants (mirrors the Python implementation).
const (
	ZstdLevel        = 10
	MinDictSamples   = 5
	MinSampleSize    = 8
	EntropyBlockSize = 256
)

// DefaultWorkers is cpu_count-1 (at least 1), matching the Python default.
func DefaultWorkers() int {
	if n := runtime.NumCPU() - 1; n > 0 {
		return n
	}
	return 1
}

// plainEnc is a shared, dictionary-less encoder. EncodeAll is concurrency-safe,
// so a single instance serves every goroutine.
var plainEnc *zstd.Encoder

func init() {
	enc, err := zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(ZstdLevel)))
	if err != nil {
		panic(fmt.Sprintf("filesim: cannot create zstd encoder: %v", err))
	}
	plainEnc = enc
}

// FileMeta holds the derived metadata for one file. Raw bytes are released after
// preprocessing; only this survives into the pair phase.
type FileMeta struct {
	Path    string
	Size    int
	cx      int           // compressed size with no dictionary
	h       float64       // global Shannon entropy [0, 8]
	profile []float64     // per-block entropy values
	fp      []byte        // entropy fingerprint (packed float32s)
	dictEnc *zstd.Encoder // encoder carrying this file's trained dict; nil if none
}

// Result is the hybrid similarity record for one unordered pair.
type Result struct {
	FileA          string
	FileB          string
	Hybrid         float64
	NCDDict        float64
	NCDFingerprint float64
	EntropyGlobal  float64
	EntropyProfile float64
	HA             float64
	HB             float64
}

// ---------------------------------------------------------------------------
// Entropy utilities
// ---------------------------------------------------------------------------

func shannonEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0.0
	}
	var counts [256]int
	for _, b := range data {
		counts[b]++
	}
	n := float64(len(data))
	h := 0.0
	for _, c := range counts {
		if c == 0 {
			continue
		}
		p := float64(c) / n
		h -= p * math.Log2(p)
	}
	return h
}

func entropyProfile(data []byte) []float64 {
	var result []float64
	for i := 0; i < len(data); i += EntropyBlockSize {
		end := min(i+EntropyBlockSize, len(data))
		blk := data[i:end]
		if len(blk) >= MinSampleSize {
			result = append(result, shannonEntropy(blk))
		}
	}
	if len(result) == 0 {
		return []float64{shannonEntropy(data)}
	}
	return result
}

func entropyFingerprint(profile []float64) []byte {
	buf := make([]byte, 4*len(profile))
	for i, v := range profile {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
	}
	return buf
}

func profileDistance(p1, p2 []float64) float64 {
	n := min(len(p1), len(p2))
	if n == 0 {
		return 1.0
	}
	sum := 0.0
	for i := range n {
		sum += math.Abs(p1[i] - p2[i])
	}
	return sum / (float64(n) * 8.0)
}

// ---------------------------------------------------------------------------
// Compression utilities
// ---------------------------------------------------------------------------

func compressSize(enc *zstd.Encoder, data []byte) int {
	return len(enc.EncodeAll(data, nil))
}

// buildDictEncoder trains a zstd dictionary on data and returns an encoder that
// uses it. Returns nil when data is too small to yield enough samples or when
// training fails — callers then fall back to concatenation NCD.
func buildDictEncoder(data []byte) *zstd.Encoder {
	chunk := max(len(data)/10, MinSampleSize)
	var samples [][]byte
	for i := 0; i < len(data); i += chunk {
		end := min(i+chunk, len(data))
		if end-i >= MinSampleSize {
			// copy: BuildDict retains references to the sample slices.
			s := make([]byte, end-i)
			copy(s, data[i:end])
			samples = append(samples, s)
		}
	}
	if len(samples) < MinDictSamples {
		return nil
	}
	dictBytes, err := zstd.BuildDict(zstd.BuildDictOptions{
		Contents: samples,
		Level:    zstd.EncoderLevelFromZstd(ZstdLevel),
	})
	if err != nil {
		return nil
	}
	enc, err := zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(ZstdLevel)),
		zstd.WithEncoderDict(dictBytes))
	if err != nil {
		return nil
	}
	return enc
}

// ---------------------------------------------------------------------------
// Preprocessing: file -> FileMeta (raw bytes released on return)
// ---------------------------------------------------------------------------

func preprocessFile(path string) (*FileMeta, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	profile := entropyProfile(data)
	return &FileMeta{
		Path:    path,
		Size:    len(data),
		cx:      compressSize(plainEnc, data),
		h:       shannonEntropy(data),
		profile: profile,
		fp:      entropyFingerprint(profile),
		dictEnc: buildDictEncoder(data),
	}, nil
}

// Preprocess reads and derives metadata for each path concurrently. Files that
// cannot be read are reported via warn (if non-nil) and skipped.
func Preprocess(paths []string, workers int, warn func(string)) []*FileMeta {
	if workers < 1 {
		workers = 1
	}
	jobs := make(chan string)
	var mu sync.Mutex
	var metas []*FileMeta
	var wg sync.WaitGroup

	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range jobs {
				meta, err := preprocessFile(p)
				if err != nil {
					if warn != nil {
						warn(fmt.Sprintf("cannot read %s: %v", p, err))
					}
					continue
				}
				mu.Lock()
				metas = append(metas, meta)
				mu.Unlock()
			}
		}()
	}
	for _, p := range paths {
		jobs <- p
	}
	close(jobs)
	wg.Wait()
	return metas
}

// ---------------------------------------------------------------------------
// Pair computation
// ---------------------------------------------------------------------------

func ncdFP(fpA, fpB []byte) float64 {
	ca := compressSize(plainEnc, fpA)
	cb := compressSize(plainEnc, fpB)
	cat := make([]byte, 0, len(fpA)+len(fpB))
	cat = append(append(cat, fpA...), fpB...)
	cab := compressSize(plainEnc, cat)
	denom := max(ca, cb)
	if denom == 0 {
		return 0.0
	}
	return float64(cab-min(ca, cb)) / float64(denom)
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func computePair(a, b *FileMeta, alpha, beta float64) (Result, error) {
	dataA, err := os.ReadFile(a.Path)
	if err != nil {
		return Result{}, err
	}
	dataB, err := os.ReadFile(b.Path)
	if err != nil {
		return Result{}, err
	}

	denom := max(a.cx, b.cx)

	var nd float64
	if a.dictEnc != nil || b.dictEnc != nil {
		// Conditional NCD: compress each file using a dictionary trained on the
		// other, then take the cheaper direction. Missing dict => plain encoder.
		var cBgivenA int
		if a.dictEnc != nil {
			cBgivenA = compressSize(a.dictEnc, dataB)
		} else {
			cBgivenA = compressSize(plainEnc, dataB)
		}
		var cAgivenB int
		if b.dictEnc != nil {
			cAgivenB = compressSize(b.dictEnc, dataA)
		} else {
			cAgivenB = compressSize(plainEnc, dataA)
		}
		if denom != 0 {
			nd = float64(min(cBgivenA, cAgivenB)) / float64(denom)
		}
	} else {
		cat := make([]byte, 0, len(dataA)+len(dataB))
		cat = append(append(cat, dataA...), dataB...)
		cxy := compressSize(plainEnc, cat)
		if denom != 0 {
			nd = float64(cxy-min(a.cx, b.cx)) / float64(denom)
		}
	}
	nd = clamp01(nd)

	nf := clamp01(ncdFP(a.fp, b.fp))
	eg := math.Abs(a.h-b.h) / 8.0
	ep := profileDistance(a.profile, b.profile)

	gamma := (1.0 - alpha - beta) / 2.0
	hybrid := alpha*nd + beta*nf + gamma*eg + gamma*ep

	return Result{
		FileA:          a.Path,
		FileB:          b.Path,
		Hybrid:         round6(hybrid),
		NCDDict:        round6(nd),
		NCDFingerprint: round6(nf),
		EntropyGlobal:  round6(eg),
		EntropyProfile: round6(ep),
		HA:             round3(a.h),
		HB:             round3(b.h),
	}, nil
}

func round6(v float64) float64 { return math.Round(v*1e6) / 1e6 }
func round3(v float64) float64 { return math.Round(v*1e3) / 1e3 }

// Pairs computes every unordered pair concurrently and streams results on the
// returned channel, which is closed when all pairs are done. Pairs are
// generated lazily, so the full O(N^2) list is never materialized. warn, if
// non-nil, receives per-pair failure messages.
func Pairs(metas []*FileMeta, alpha, beta float64, workers int, warn func(string)) <-chan Result {
	if workers < 1 {
		workers = 1
	}
	out := make(chan Result)

	type job struct{ a, b *FileMeta }
	jobs := make(chan job, workers*2)

	go func() {
		for i := range metas {
			for j := i + 1; j < len(metas); j++ {
				jobs <- job{metas[i], metas[j]}
			}
		}
		close(jobs)
	}()

	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for jb := range jobs {
				res, err := computePair(jb.a, jb.b, alpha, beta)
				if err != nil {
					if warn != nil {
						warn(fmt.Sprintf("pair (%s, %s) failed: %v",
							filepath.Base(jb.a.Path), filepath.Base(jb.b.Path), err))
					}
					continue
				}
				out <- res
			}
		}()
	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}

// NumPairs returns n*(n-1)/2.
func NumPairs(n int) int { return n * (n - 1) / 2 }

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

// CollectFiles walks dir recursively, returning files that pass the extension
// and size filters in sorted order. exts entries should include the leading dot
// and be lowercase; an empty exts means "any extension". skip, if non-nil,
// receives messages about skipped files.
func CollectFiles(dir string, exts []string, maxSize int64, skip func(string)) ([]string, error) {
	extSet := make(map[string]bool, len(exts))
	for _, e := range exts {
		extSet[strings.ToLower(e)] = true
	}

	var files []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // ignore unreadable entries, keep walking
		}
		if d.IsDir() {
			return nil
		}
		if len(extSet) > 0 && !extSet[strings.ToLower(filepath.Ext(path))] {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		size := info.Size()
		if size == 0 {
			return nil
		}
		if size > maxSize {
			if skip != nil {
				skip(fmt.Sprintf("skipping %s (%d B > max %d B)",
					filepath.Base(path), size, maxSize))
			}
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	return files, nil
}

// SortKey returns the comparable value of r for the given field name, used to
// sort table output. Unknown fields fall back to Hybrid.
func SortKey(r Result, by string) float64 {
	switch by {
	case "ncd_dict":
		return r.NCDDict
	case "ncd_fingerprint":
		return r.NCDFingerprint
	case "entropy_global":
		return r.EntropyGlobal
	case "entropy_profile":
		return r.EntropyProfile
	default:
		return r.Hybrid
	}
}
