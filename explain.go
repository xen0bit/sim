package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/xen0bit/sim/filesim"
)

// runExplain implements: sim explain <ref> <target> [options]
//
// It decomposes <target> into the byte spans that occur verbatim in <ref>
// (the "shared chunks"), modelling what zstd would back-reference when
// compressing <target> against a dictionary/prefix built from <ref>.
func runExplain(args []string) error {
	fs := flag.NewFlagSet("sim explain", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `sim explain - show the byte chunks of TARGET that occur in REF.

Usage:
  sim explain [options] <ref> <target>

Decomposes TARGET into matched spans (runs that also appear in REF, with the
offset where they occur in REF) and literal runs (no qualifying match). This is
a pure-Go greedy LZ parse of TARGET against REF's raw bytes.

Options:
`)
		fs.PrintDefaults()
	}
	minMatch := fs.Int("min-match", 16, "minimum match length in bytes")
	format := fs.String("format", "table", "output format: table|csv|json")
	both := fs.Bool("both", false, "also decompose REF against TARGET (reverse direction)")
	fs.Parse(args)

	if fs.NArg() != 2 {
		fs.Usage()
		return fmt.Errorf("explain needs exactly two files: <ref> <target>")
	}
	if *minMatch < 1 {
		return fmt.Errorf("--min-match must be >= 1")
	}
	switch *format {
	case "table", "csv", "json":
	default:
		return fmt.Errorf("invalid --format %q", *format)
	}

	refPath, tgtPath := fs.Arg(0), fs.Arg(1)
	ref, err := os.ReadFile(refPath)
	if err != nil {
		return err
	}
	tgt, err := os.ReadFile(tgtPath)
	if err != nil {
		return err
	}

	views := []decomposition{newDecomposition(refPath, tgtPath, ref, tgt, *minMatch)}
	if *both {
		views = append(views, newDecomposition(tgtPath, refPath, tgt, ref, *minMatch))
	}

	switch *format {
	case "json":
		return emitJSON(views)
	case "csv":
		return emitCSV(views)
	default:
		emitTable(views)
		return nil
	}
}

type decomposition struct {
	Ref      string
	Target   string
	MinMatch int
	Chunks   []filesim.Chunk
	Matched  int
	Total    int
	Spans    int
}

func newDecomposition(ref, target string, refData, tgtData []byte, minMatch int) decomposition {
	chunks := filesim.SharedChunks(refData, tgtData, minMatch)
	m, t, s := filesim.Coverage(chunks)
	return decomposition{
		Ref: ref, Target: target, MinMatch: minMatch,
		Chunks: chunks, Matched: m, Total: t, Spans: s,
	}
}

func (d decomposition) coverage() float64 {
	if d.Total == 0 {
		return 0
	}
	return float64(d.Matched) / float64(d.Total)
}

func emitTable(views []decomposition) {
	for _, d := range views {
		fmt.Printf("\n%s decomposed against %s  (min-match=%d)\n", d.Target, d.Ref, d.MinMatch)
		fmt.Printf("%5s  %-7s  %10s  %10s  %10s  %12s\n",
			"chunk", "kind", "b_start", "b_end", "len", "a_offset")
		fmt.Println("-----  -------  ----------  ----------  ----------  ------------")
		for i, c := range d.Chunks {
			off := "-"
			kind := "literal"
			if c.Matched {
				off = strconv.Itoa(c.AOffset)
				kind = "match"
			}
			fmt.Printf("%5d  %-7s  %10d  %10d  %10d  %12s\n",
				i, kind, c.BStart, c.BEnd, c.Length, off)
		}
		fmt.Printf("Coverage: %d/%d bytes (%.1f%%) matched across %d span(s)\n",
			d.Matched, d.Total, d.coverage()*100, d.Spans)
	}
}

func emitCSV(views []decomposition) error {
	w := csv.NewWriter(os.Stdout)
	if err := w.Write([]string{"ref", "target", "chunk", "kind", "b_start", "b_end", "len", "a_offset"}); err != nil {
		return err
	}
	for _, d := range views {
		for i, c := range d.Chunks {
			kind, off := "literal", ""
			if c.Matched {
				kind, off = "match", strconv.Itoa(c.AOffset)
			}
			rec := []string{
				d.Ref, d.Target, strconv.Itoa(i), kind,
				strconv.Itoa(c.BStart), strconv.Itoa(c.BEnd), strconv.Itoa(c.Length), off,
			}
			if err := w.Write(rec); err != nil {
				return err
			}
		}
	}
	w.Flush()
	return w.Error()
}

type chunkJSON struct {
	Kind    string `json:"kind"`
	BStart  int    `json:"b_start"`
	BEnd    int    `json:"b_end"`
	Length  int    `json:"len"`
	AOffset *int   `json:"a_offset"` // null for literal runs
}

type decompositionJSON struct {
	Ref          string      `json:"ref"`
	Target       string      `json:"target"`
	MinMatch     int         `json:"min_match"`
	MatchedBytes int         `json:"matched_bytes"`
	TotalBytes   int         `json:"total_bytes"`
	Coverage     float64     `json:"coverage"`
	Spans        int         `json:"spans"`
	Chunks       []chunkJSON `json:"chunks"`
}

func emitJSON(views []decomposition) error {
	out := make([]decompositionJSON, 0, len(views))
	for _, d := range views {
		cj := make([]chunkJSON, len(d.Chunks))
		for i, c := range d.Chunks {
			var off *int
			kind := "literal"
			if c.Matched {
				v := c.AOffset
				off = &v
				kind = "match"
			}
			cj[i] = chunkJSON{Kind: kind, BStart: c.BStart, BEnd: c.BEnd, Length: c.Length, AOffset: off}
		}
		out = append(out, decompositionJSON{
			Ref: d.Ref, Target: d.Target, MinMatch: d.MinMatch,
			MatchedBytes: d.Matched, TotalBytes: d.Total,
			Coverage: d.coverage(), Spans: d.Spans, Chunks: cj,
		})
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}
