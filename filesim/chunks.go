package filesim

// Shared-chunk decomposition.
//
// When file B is compressed against a dictionary trained on file A, zstd encodes
// B as a stream of literals and (offset, length) back-references; a reference
// whose offset reaches into A is "a chunk of B that was found in A." The
// high-level zstd API hides those references, so we reconstruct the same idea
// directly and exactly: a greedy LZ77 parse of B against A's raw bytes using a
// hash-chain match finder (the classic zlib/zstd match-finder structure).
//
// The result is the precise set of byte spans of B that also occur in A, each
// annotated with where in A it was found. This is more interpretable than
// zstd's internal, level-dependent parse and stays 100% pure Go.

const (
	hashBase uint64 = 1099511628211 // FNV-64 prime, used as a rolling-hash base
	maxChain int    = 64            // candidates examined per anchor (match quality vs speed)
)

// Chunk is one contiguous segment of file B in the decomposition. Matched
// segments occur verbatim in A starting at AOffset; literal segments
// (Matched=false, AOffset=-1) are runs of B with no qualifying match in A.
type Chunk struct {
	BStart  int // inclusive start offset in B
	BEnd    int // exclusive end offset in B
	Length  int // BEnd - BStart
	AOffset int // start offset in A for a match; -1 for literal runs
	Matched bool
}

func hashWindow(data []byte, p, k int) uint64 {
	var h uint64
	for i := range k {
		h = h*hashBase + uint64(data[p+i])
	}
	return h
}

// buildIndex hashes every k-byte window of a into a hash-chain: head[hash] is
// the most recent window position, and chain[p] links to the previous window
// with the same hash (-1 terminates the chain).
func buildIndex(a []byte, k int) (map[uint64]int32, []int32, uint64) {
	var pow uint64 = 1
	for i := 0; i < k-1; i++ {
		pow *= hashBase
	}
	n := len(a)
	head := make(map[uint64]int32)
	if n < k {
		return head, nil, pow
	}
	chain := make([]int32, n-k+1)
	h := hashWindow(a, 0, k)
	for p := 0; ; p++ {
		if prev, ok := head[h]; ok {
			chain[p] = prev
		} else {
			chain[p] = -1
		}
		head[h] = int32(p)
		if p == n-k {
			break
		}
		h = (h-uint64(a[p])*pow)*hashBase + uint64(a[p+k])
	}
	return head, chain, pow
}

func equalAt(a, b []byte, c, p, k int) bool {
	for i := range k {
		if a[c+i] != b[p+i] {
			return false
		}
	}
	return true
}

// longestMatch returns the length and A-offset of the longest match for the
// k-byte window of b at position p (whose hash is hB), or (0, -1) if none
// reaches the minimum length k. Ties prefer the smaller A-offset for
// determinism.
func longestMatch(a, b []byte, head map[uint64]int32, chain []int32, k, p int, hB uint64) (int, int) {
	cand, ok := head[hB]
	if !ok {
		return 0, -1
	}
	bestLen, bestOff := 0, -1
	for steps := 0; cand >= 0 && steps < maxChain; steps++ {
		c := int(cand)
		if equalAt(a, b, c, p, k) { // guard against hash collisions
			l := k
			for p+l < len(b) && c+l < len(a) && a[c+l] == b[p+l] {
				l++
			}
			if l > bestLen || (l == bestLen && c < bestOff) {
				bestLen, bestOff = l, c
			}
		}
		cand = chain[c]
	}
	if bestLen >= k {
		return bestLen, bestOff
	}
	return 0, -1
}

// SharedChunks decomposes b into an ordered cover of matched and literal
// segments, where a matched segment is the longest run starting at that
// position that also occurs verbatim in a (at least minMatch bytes long). The
// returned chunks tile [0, len(b)) contiguously with no gaps or overlaps.
func SharedChunks(a, b []byte, minMatch int) []Chunk {
	if minMatch < 1 {
		minMatch = 1
	}
	k := minMatch
	n := len(b)
	var chunks []Chunk

	if len(a) < k || n < k {
		if n > 0 {
			chunks = append(chunks, Chunk{BStart: 0, BEnd: n, Length: n, AOffset: -1})
		}
		return chunks
	}

	head, chain, pow := buildIndex(a, k)
	h := hashWindow(b, 0, k)
	p := 0
	litStart := 0

	flushLiteral := func(end int) {
		if end > litStart {
			chunks = append(chunks, Chunk{BStart: litStart, BEnd: end, Length: end - litStart, AOffset: -1})
		}
	}
	roll := func() {
		if p+k < n {
			h = (h-uint64(b[p])*pow)*hashBase + uint64(b[p+k])
		}
	}

	for p+k <= n {
		ml, off := longestMatch(a, b, head, chain, k, p, h)
		if ml >= k {
			flushLiteral(p)
			chunks = append(chunks, Chunk{BStart: p, BEnd: p + ml, Length: ml, AOffset: off, Matched: true})
			target := p + ml
			for p < target {
				roll()
				p++
			}
			litStart = p
		} else {
			roll()
			p++
		}
	}
	flushLiteral(n) // trailing literal tail (final partial window can't match)
	return chunks
}

// Coverage summarizes a decomposition: bytes covered by matches, total bytes,
// and the number of matched spans.
func Coverage(chunks []Chunk) (matched, total, spans int) {
	for _, c := range chunks {
		total += c.Length
		if c.Matched {
			matched += c.Length
			spans++
		}
	}
	return matched, total, spans
}
