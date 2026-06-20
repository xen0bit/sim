// Command sim recursively measures pairwise file similarity within a directory
// using the hybrid NCD + entropy metric implemented in the filesim package.
//
//	go install github.com/xen0bit/sim@latest
//	sim /path/to/dir
//	sim /path/to/dir --ext .bin,.exe --sort ncd_dict
//	sim /path/to/dir --format csv > results.csv
//	sim /path/to/dir --workers 8 --verbose
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/xen0bit/sim/filesim"
)

func main() {
	var err error
	if len(os.Args) > 1 && os.Args[1] == "explain" {
		err = runExplain(os.Args[2:])
	} else {
		err = run()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func run() error {
	fs := flag.NewFlagSet("sim", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, `sim - recursive file similarity: hybrid NCD + entropy distance.

Usage:
  sim [options] <directory>

Options:
`)
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, `
Score components (all in [0, 1], lower = more similar):
  ncd_dict        NCD using a zstd dictionary trained on one file to compress
                  the other (falls back to concatenation NCD for tiny files).
  ncd_fingerprint NCD over per-block entropy fingerprints (class similarity,
                  e.g. both files encrypted -> flat ~8 bits/byte profile).
  entropy_global  |H(a) - H(b)| / 8.
  entropy_profile mean absolute per-block entropy difference.

  hybrid = alpha*ncd_dict + beta*ncd_fingerprint + gamma*(entropy_global +
           entropy_profile), where gamma = (1 - alpha - beta) / 2.
`)
	}

	ext := fs.String("ext", "", "comma-separated extensions to include, e.g. .bin,.exe (default: all)")
	maxSize := fs.Int64("max-size", 10*1024*1024, "skip files larger than N bytes")
	alpha := fs.Float64("alpha", 0.5, "weight for dictionary NCD")
	beta := fs.Float64("beta", 0.25, "weight for fingerprint NCD")
	workers := fs.Int("workers", filesim.DefaultWorkers(), "worker pool size")
	sortBy := fs.String("sort", "hybrid", "sort field: hybrid|ncd_dict|ncd_fingerprint|entropy_global|entropy_profile")
	format := fs.String("format", "table", "output format: table|csv")
	verbose := fs.Bool("verbose", false, "verbose progress on stderr")
	fs.Parse(os.Args[1:])

	if fs.NArg() < 1 {
		fs.Usage()
		return fmt.Errorf("missing <directory> argument")
	}
	dir := fs.Arg(0)

	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("%s is not a directory", dir)
	}
	if *alpha+*beta > 1.0 {
		return fmt.Errorf("alpha + beta must be <= 1.0")
	}
	if *workers < 1 {
		return fmt.Errorf("--workers must be >= 1")
	}
	if !validSort(*sortBy) {
		return fmt.Errorf("invalid --sort %q", *sortBy)
	}
	if *format != "table" && *format != "csv" {
		return fmt.Errorf("invalid --format %q", *format)
	}

	var exts []string
	for e := range strings.SplitSeq(*ext, ",") {
		e = strings.TrimSpace(e)
		if e == "" {
			continue
		}
		if !strings.HasPrefix(e, ".") {
			e = "." + e
		}
		exts = append(exts, e)
	}

	warn := func(msg string) { fmt.Fprintln(os.Stderr, "  Warning:", msg) }
	var skip func(string)
	if *verbose {
		skip = func(msg string) { fmt.Fprintln(os.Stderr, "  "+msg) }
	}

	if *verbose {
		fmt.Fprintf(os.Stderr, "Scanning %s...\n", dir)
	}
	files, err := filesim.CollectFiles(dir, exts, *maxSize, skip)
	if err != nil {
		return err
	}
	if len(files) < 2 {
		return fmt.Errorf("need at least 2 files; found %d", len(files))
	}

	if *verbose {
		fmt.Fprintf(os.Stderr, "Found %d files. Preprocessing with %d workers...\n", len(files), *workers)
	}
	metas := filesim.Preprocess(files, *workers, warn)
	if len(metas) < 2 {
		return fmt.Errorf("too many files failed to load")
	}
	if *verbose {
		for _, m := range metas {
			fmt.Fprintf(os.Stderr, "  preprocessed %s\n", m.Path)
		}
		fmt.Fprintf(os.Stderr, "\nDispatching %d pairs across %d threads...\n",
			filesim.NumPairs(len(metas)), *workers)
	}

	results := filesim.Pairs(metas, *alpha, *beta, *workers, warn)

	if *format == "csv" {
		return streamCSV(results)
	}
	return printTable(results, *sortBy)
}

func validSort(s string) bool {
	switch s {
	case "hybrid", "ncd_dict", "ncd_fingerprint", "entropy_global", "entropy_profile":
		return true
	}
	return false
}

var header = []string{
	"file_a", "file_b", "hybrid", "ncd_dict", "ncd_fingerprint",
	"entropy_global", "entropy_profile", "H_a", "H_b",
}

func row(r filesim.Result) []string {
	return []string{
		r.FileA, r.FileB,
		f6(r.Hybrid), f6(r.NCDDict), f6(r.NCDFingerprint),
		f6(r.EntropyGlobal), f6(r.EntropyProfile),
		strconv.FormatFloat(r.HA, 'f', 3, 64),
		strconv.FormatFloat(r.HB, 'f', 3, 64),
	}
}

func f6(v float64) string { return strconv.FormatFloat(v, 'f', 6, 64) }

// streamCSV writes results straight to stdout as they arrive — never buffers.
func streamCSV(results <-chan filesim.Result) error {
	w := csv.NewWriter(os.Stdout)
	if err := w.Write(header); err != nil {
		return err
	}
	for r := range results {
		if err := w.Write(row(r)); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

// printTable buffers results (required for the final sort) then prints aligned.
func printTable(results <-chan filesim.Result, sortBy string) error {
	var rows []filesim.Result
	for r := range results {
		rows = append(rows, r)
	}
	if len(rows) == 0 {
		fmt.Println("No pairs computed.")
		return nil
	}
	sort.SliceStable(rows, func(i, j int) bool {
		return filesim.SortKey(rows[i], sortBy) < filesim.SortKey(rows[j], sortBy)
	})

	colA, colB := len("File A"), len("File B")
	for _, r := range rows {
		if len(r.FileA) > colA {
			colA = len(r.FileA)
		}
		if len(r.FileB) > colB {
			colB = len(r.FileB)
		}
	}

	head := fmt.Sprintf("%-*s  %-*s  %8s  %8s  %8s  %8s  %9s  %5s  %5s",
		colA, "File A", colB, "File B", "Hybrid", "NCD-Dict", "NCD-FP",
		"H-Global", "H-Profile", "H(a)", "H(b)")
	sep := strings.Repeat("-", len(head))
	fmt.Println(sep)
	fmt.Println(head)
	fmt.Println(sep)
	for _, r := range rows {
		fmt.Printf("%-*s  %-*s  %8.4f  %8.4f  %8.4f  %8.4f  %9.4f  %5.2f  %5.2f\n",
			colA, r.FileA, colB, r.FileB, r.Hybrid, r.NCDDict, r.NCDFingerprint,
			r.EntropyGlobal, r.EntropyProfile, r.HA, r.HB)
	}
	fmt.Println(sep)
	return nil
}
