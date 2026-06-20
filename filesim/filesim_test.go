package filesim

import (
	"crypto/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFixtures creates a small corpus: two similar docs, one unrelated doc,
// and two high-entropy ("encrypted") blobs.
func writeFixtures(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	write := func(name string, data []byte) {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("doc_a.txt", []byte(strings.Repeat("the cat sat on the mat. ", 200)))
	write("doc_b.txt", []byte(strings.Repeat("the cat lay on the rug. ", 200)))
	write("doc_c.txt", []byte(strings.Repeat("quantum entanglement abounds. ", 120)))
	encA := make([]byte, 4096)
	encB := make([]byte, 4096)
	rand.Read(encA)
	rand.Read(encB)
	write("enc_a.bin", encA)
	write("enc_b.bin", encB)
	return dir
}

func collectResults(t *testing.T, dir string, alpha, beta float64) map[string]Result {
	t.Helper()
	files, err := CollectFiles(dir, nil, 10*1024*1024, nil)
	if err != nil {
		t.Fatal(err)
	}
	metas := Preprocess(files, 4, nil)
	if len(metas) != 5 {
		t.Fatalf("expected 5 metas, got %d", len(metas))
	}
	out := map[string]Result{}
	for r := range Pairs(metas, alpha, beta, 4, nil) {
		a, b := filepath.Base(r.FileA), filepath.Base(r.FileB)
		if a > b {
			a, b = b, a
		}
		out[a+"|"+b] = r
	}
	return out
}

func TestComponentsBounded(t *testing.T) {
	dir := writeFixtures(t)
	results := collectResults(t, dir, 0.5, 0.25)
	if len(results) != 10 { // C(5,2)
		t.Fatalf("expected 10 pairs, got %d", len(results))
	}
	for k, r := range results {
		for name, v := range map[string]float64{
			"hybrid": r.Hybrid, "ncd_dict": r.NCDDict, "ncd_fp": r.NCDFingerprint,
			"eg": r.EntropyGlobal, "ep": r.EntropyProfile,
		} {
			if v < 0 || v > 1 {
				t.Errorf("pair %s: %s = %f out of [0,1]", k, name, v)
			}
		}
	}
}

// TestEncryptedClassSimilarity is the whole point of the hybrid metric: two
// high-entropy blobs are byte-wise incompressible together (ncd_dict ~1) yet
// should score as more similar overall than an encrypted blob vs. a text doc,
// because their entropy fingerprints match.
func TestEncryptedClassSimilarity(t *testing.T) {
	dir := writeFixtures(t)
	r := collectResults(t, dir, 0.5, 0.25)

	encPair := r["enc_a.bin|enc_b.bin"]
	mixed := r["doc_a.txt|enc_a.bin"]

	if encPair.NCDDict < 0.9 {
		t.Errorf("expected encrypted pair to be byte-wise dissimilar, ncd_dict=%f", encPair.NCDDict)
	}
	if encPair.Hybrid >= mixed.Hybrid {
		t.Errorf("encrypted pair (%f) should be more similar than enc/doc pair (%f)",
			encPair.Hybrid, mixed.Hybrid)
	}
	if encPair.EntropyGlobal > 0.05 {
		t.Errorf("two encrypted blobs should have near-identical global entropy, got %f", encPair.EntropyGlobal)
	}
}

func TestMostSimilarPair(t *testing.T) {
	dir := writeFixtures(t)
	r := collectResults(t, dir, 0.5, 0.25)
	best := ""
	bestVal := 2.0
	for k, res := range r {
		if res.Hybrid < bestVal {
			bestVal, best = res.Hybrid, k
		}
	}
	if best != "doc_a.txt|doc_b.txt" {
		t.Errorf("expected doc_a/doc_b most similar, got %s (%f)", best, bestVal)
	}
}

func TestShannonEntropyExtremes(t *testing.T) {
	if h := shannonEntropy(nil); h != 0 {
		t.Errorf("empty entropy = %f, want 0", h)
	}
	uniform := make([]byte, 256)
	for i := range uniform {
		uniform[i] = byte(i)
	}
	if h := shannonEntropy(uniform); h < 7.99 || h > 8.0001 {
		t.Errorf("uniform-byte entropy = %f, want ~8", h)
	}
	if h := shannonEntropy([]byte("aaaaaaaa")); h != 0 {
		t.Errorf("single-symbol entropy = %f, want 0", h)
	}
}
