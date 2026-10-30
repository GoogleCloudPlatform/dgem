// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// SVG can't be decoded by the serving image (#117): it is refused before the request, with a clear error.
func TestBuildMultimodalContentRejectsSVG(t *testing.T) {
	dir := t.TempDir()
	svg := filepath.Join(dir, "card.svg")
	disguised := filepath.Join(dir, "card.img") // SVG content, misleading extension
	png := filepath.Join(dir, "card.png")
	_ = os.WriteFile(svg, []byte(`<svg xmlns="http://www.w3.org/2000/svg"></svg>`), 0o644)
	_ = os.WriteFile(disguised, []byte(`<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>`), 0o644)
	_ = os.WriteFile(png, []byte("\x89PNG\r\n\x1a\n...."), 0o644)

	for _, ref := range []string{svg, disguised, "data:image/svg+xml;base64,PHN2Zy8+", "data:image/SVG+xml,<svg/>"} {
		_, err := BuildMultimodalContent("{}", []string{ref})
		if !errors.Is(err, ErrUnsupportedImageFormat) || !strings.Contains(err.Error(), "PNG") {
			t.Errorf("%s: err = %v, want ErrUnsupportedImageFormat with a PNG hint", ref, err)
		}
	}
	for _, ref := range []string{png, "data:image/png;base64,AAAA"} {
		if _, err := BuildMultimodalContent("{}", []string{ref}); err != nil {
			t.Errorf("%s: unexpected error %v", ref, err)
		}
	}
}

// The repo's PNG fixtures (used in README and docs) are accepted; their SVG sources are refused.
func TestFixtureImages(t *testing.T) {
	for _, name := range []string{"ui_component", "pcb_board", "id_card"} {
		base := filepath.Join("..", "..", "fixtures", name)
		if _, err := BuildMultimodalContent("{}", []string{base + ".png"}); err != nil {
			t.Errorf("%s.png: %v", name, err)
		}
		if _, err := BuildMultimodalContent("{}", []string{base + ".svg"}); !errors.Is(err, ErrUnsupportedImageFormat) {
			t.Errorf("%s.svg: err = %v, want refusal", name, err)
		}
	}
}

// Image URLs are fetched by the client and inlined; an SVG or non-image behind a URL is refused, a 404 errors.
func TestBuildMultimodalContentFetchesURLs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/a.png":
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))
		case "/a.svg":
			w.Header().Set("Content-Type", "image/svg+xml")
			_, _ = w.Write([]byte(`<svg xmlns="http://www.w3.org/2000/svg"></svg>`))
		case "/page":
			_, _ = w.Write([]byte("<html></html>"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	out, err := BuildMultimodalContent("{}", []string{srv.URL + "/a.png"})
	if err != nil || !strings.HasPrefix(out.([]ContentPart)[0].ImageURL.URL, "data:image/png;base64,") {
		t.Fatalf("png: %v %v", out, err)
	}
	for _, bad := range []string{"/a.svg", "/page", "/missing"} {
		if _, err := BuildMultimodalContent("{}", []string{srv.URL + bad}); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}
