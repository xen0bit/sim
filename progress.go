package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/xen0bit/sim/filesim"
)

const progressBarWidth = 30

// withProgress forwards every result from in to the returned channel. When
// enabled, it also renders a live progress bar with ETA to stderr (a few times
// per second), driven by how many of total pairs have been consumed. When
// disabled it returns in unchanged, adding zero overhead.
//
// Progress is written to stderr with carriage returns, so it never interferes
// with table/CSV output on stdout.
func withProgress(in <-chan filesim.Result, total int, enabled bool) <-chan filesim.Result {
	if !enabled {
		return in
	}
	out := make(chan filesim.Result)
	var done int64
	start := time.Now()
	stop := make(chan struct{})
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				renderProgress(atomic.LoadInt64(&done), total, start, true)
				return
			case <-ticker.C:
				renderProgress(atomic.LoadInt64(&done), total, start, false)
			}
		}
	}()

	go func() {
		for r := range in {
			out <- r // blocks until the consumer takes it, so done tracks real progress
			atomic.AddInt64(&done, 1)
		}
		close(stop)
		wg.Wait()
		close(out)
	}()
	return out
}

func renderProgress(done int64, total int, start time.Time, final bool) {
	elapsed := time.Since(start)

	frac := 0.0
	if total > 0 {
		frac = float64(done) / float64(total)
		if frac > 1 {
			frac = 1
		}
	}

	filled := int(frac * progressBarWidth)
	var bar strings.Builder
	bar.WriteString(strings.Repeat("=", filled))
	if filled < progressBarWidth {
		bar.WriteByte('>')
		bar.WriteString(strings.Repeat(" ", progressBarWidth-filled-1))
	}

	var rate float64
	if s := elapsed.Seconds(); s > 0 {
		rate = float64(done) / s
	}
	eta := "--:--"
	switch {
	case done >= int64(total):
		eta = "00:00"
	case rate > 0:
		remaining := float64(int64(total)-done) / rate
		eta = fmtDuration(time.Duration(remaining * float64(time.Second)))
	}

	fmt.Fprintf(os.Stderr, "\r[%s] %5.1f%%  %s/%s  %s/s  ETA %s  elapsed %s   ",
		bar.String(), frac*100,
		humanInt(done), humanInt(int64(total)),
		humanRate(rate), eta, fmtDuration(elapsed))
	if final {
		fmt.Fprintln(os.Stderr)
	}
}

func fmtDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	total := int(d.Seconds())
	h, m, s := total/3600, (total%3600)/60, total%60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%02d:%02d", m, s)
}

// humanInt formats n with thousands separators, e.g. 151006131 -> "151,006,131".
func humanInt(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var b strings.Builder
	for i := range len(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteByte(s[i])
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

func humanRate(r float64) string {
	switch {
	case r >= 1e6:
		return fmt.Sprintf("%.1fM", r/1e6)
	case r >= 1e3:
		return fmt.Sprintf("%.1fk", r/1e3)
	default:
		return fmt.Sprintf("%.0f", r)
	}
}
