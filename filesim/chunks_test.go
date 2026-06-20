package filesim

import (
	"bytes"
	"testing"
)

// pattern returns deterministic but uncorrelated pseudo-random bytes via an
// LCG, so different seeds produce streams that don't match each other by
// accident (a simple linear ramp would self-match across phase shifts).
func pattern(n, seed int) []byte {
	out := make([]byte, n)
	state := uint64(seed)*2862933555777941757 + 3037000493
	for i := range out {
		state = state*6364136223846793005 + 1442695040888963407
		out[i] = byte(state >> 33)
	}
	return out
}

// assertTiling verifies the chunks contiguously cover [0, total) with no gaps
// or overlaps — the core invariant of a decomposition.
func assertTiling(t *testing.T, chunks []Chunk, total int) {
	t.Helper()
	pos := 0
	for i, c := range chunks {
		if c.BStart != pos {
			t.Fatalf("chunk %d starts at %d, expected %d (gap/overlap)", i, c.BStart, pos)
		}
		if c.BEnd != c.BStart+c.Length {
			t.Fatalf("chunk %d: BEnd %d != BStart+Length %d", i, c.BEnd, c.BStart+c.Length)
		}
		if c.Matched && c.AOffset < 0 {
			t.Fatalf("chunk %d matched but AOffset=%d", i, c.AOffset)
		}
		if !c.Matched && c.AOffset != -1 {
			t.Fatalf("chunk %d literal but AOffset=%d", i, c.AOffset)
		}
		pos = c.BEnd
	}
	if pos != total {
		t.Fatalf("chunks cover %d bytes, expected %d", pos, total)
	}
}

func TestSharedBlockLocated(t *testing.T) {
	shared := pattern(1024, 0)
	a := append(append(pattern(300, 11), shared...), pattern(200, 22)...)
	b := append(append(pattern(100, 33), shared...), pattern(500, 44)...)

	chunks := SharedChunks(a, b, 16)
	assertTiling(t, chunks, len(b))

	var match *Chunk
	for i := range chunks {
		if chunks[i].Matched {
			if match != nil {
				t.Fatalf("expected exactly one match, found a second at %d", chunks[i].BStart)
			}
			match = &chunks[i]
		}
	}
	if match == nil {
		t.Fatal("shared block not detected")
	}
	if match.BStart != 100 || match.Length != 1024 || match.AOffset != 300 {
		t.Errorf("match = {BStart:%d Length:%d AOffset:%d}, want {100 1024 300}",
			match.BStart, match.Length, match.AOffset)
	}
	// Verify the reported span actually matches A at the reported offset.
	if !bytes.Equal(a[match.AOffset:match.AOffset+match.Length], b[match.BStart:match.BEnd]) {
		t.Error("reported match span does not equal the A bytes at AOffset")
	}

	m, total, spans := Coverage(chunks)
	if m != 1024 || total != len(b) || spans != 1 {
		t.Errorf("coverage = (%d,%d,%d), want (1024,%d,1)", m, total, spans, len(b))
	}
}

func TestIdenticalFilesFullCoverage(t *testing.T) {
	a := pattern(2000, 0)
	chunks := SharedChunks(a, a, 16)
	assertTiling(t, chunks, len(a))
	m, total, _ := Coverage(chunks)
	if m != total {
		t.Errorf("identical files should be fully covered, got %d/%d", m, total)
	}
}

func TestDisjointNoMatches(t *testing.T) {
	a := bytes.Repeat([]byte{0x00}, 2000)
	b := bytes.Repeat([]byte{0xFF}, 2000)
	chunks := SharedChunks(a, b, 16)
	assertTiling(t, chunks, len(b))
	m, _, spans := Coverage(chunks)
	if m != 0 || spans != 0 {
		t.Errorf("disjoint files should share nothing, got matched=%d spans=%d", m, spans)
	}
}

func TestMinMatchLargerThanInput(t *testing.T) {
	a := pattern(10, 0)
	b := pattern(10, 0)
	chunks := SharedChunks(a, b, 16) // min-match exceeds file length
	assertTiling(t, chunks, len(b))
	if len(chunks) != 1 || chunks[0].Matched {
		t.Errorf("expected single literal chunk, got %+v", chunks)
	}
}

func TestEmptyTarget(t *testing.T) {
	chunks := SharedChunks(pattern(100, 0), nil, 16)
	if len(chunks) != 0 {
		t.Errorf("empty target should yield no chunks, got %d", len(chunks))
	}
}
