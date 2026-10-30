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

package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

var pngBytes = []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde")

// imageServer serves a PNG at /img.png, HTML at /page and SVG at /x.svg, and lets the test treat it as public.
func imageServer(t *testing.T) *httptest.Server {
	t.Helper()
	serverMode(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/img.png":
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(pngBytes)
		case "/x.svg":
			w.Header().Set("Content-Type", "image/svg+xml")
			_, _ = w.Write([]byte(`<svg xmlns="http://www.w3.org/2000/svg"></svg>`))
		default:
			w.Header().Set("Content-Type", "text/html")
			_, _ = w.Write([]byte("<html>metadata</html>"))
		}
	}))
	t.Cleanup(srv.Close)
	addr := netip.MustParseAddrPort(srv.Listener.Addr().String())
	oldDial, oldHost := imageDialGuard, imageHostCheck
	imageDialGuard = func(ap netip.AddrPort) error {
		if ap == addr {
			return nil
		}
		return errNonPublicImageAddr
	}
	imageHostCheck = func(context.Context, string) error { return nil }
	t.Cleanup(func() { imageDialGuard, imageHostCheck = oldDial, oldHost })
	return srv
}

func TestPrepareRequestImagesInlinesURLs(t *testing.T) {
	srv := imageServer(t)
	out, err := prepareRequestImages(context.Background(), []string{srv.URL + "/img.png", "data:image/png;base64,AA=="})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out[0], "data:image/png;base64,") || out[1] != "data:image/png;base64,AA==" {
		t.Fatalf("got %q", out)
	}
	for _, bad := range []string{"/page", "/x.svg", "/missing"} {
		_, err := prepareRequestImages(context.Background(), []string{srv.URL + bad})
		if err == nil {
			t.Fatalf("%s accepted", bad)
		}
	}
	if _, err := prepareRequestImages(context.Background(), []string{srv.URL + "/page"}); imageErrorStatus(err) != http.StatusUnprocessableEntity {
		t.Fatalf("non-image: status %d (%v)", imageErrorStatus(err), err)
	}
	if !errors.Is(func() error {
		_, e := prepareRequestImages(context.Background(), []string{srv.URL + "/missing"})
		return e
	}(), ErrImageFetch) {
		t.Fatal("404 should be ErrImageFetch")
	}
}

// The CLI (allowLocalImagePaths) leaves URLs alone: it fetches from the caller's own network, or lets the model do it.
func TestPrepareRequestImagesCLIUnchanged(t *testing.T) {
	out, err := prepareRequestImages(context.Background(), []string{"https://example.com/a.png"})
	if err != nil || out[0] != "https://example.com/a.png" {
		t.Fatalf("%q %v", out, err)
	}
}

func TestInlineProxyBodyImages(t *testing.T) {
	srv := imageServer(t)
	so := `{"state": "x", "questions": {"q": {"type": "noul"}}, "images": ["` + srv.URL + `/img.png", "data:image/png;base64,AA=="], "extra": {"k": 1}}`
	nb, err := inlineProxyBodyImages(context.Background(), "/v1/systemone", []byte(so))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	_ = json.Unmarshal(nb, &got)
	imgs := got["images"].([]any)
	if !strings.HasPrefix(imgs[0].(string), "data:image/png;base64,") || imgs[1] != "data:image/png;base64,AA==" || got["extra"] == nil {
		t.Fatalf("systemone body: %s", nb)
	}
	chat := `{"messages": [{"role": "system", "content": "{}"}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "` + srv.URL + `/img.png", "detail": "high"}}, {"type": "text", "text": "{}"}]}]}`
	nb, err = inlineProxyBodyImages(context.Background(), "/v1/chat/completions", []byte(chat))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(nb), srv.URL) || !strings.Contains(string(nb), `"detail":"high"`) || !strings.Contains(string(nb), "data:image/png;base64,") {
		t.Fatalf("chat body: %s", nb)
	}
	// Bodies without URLs pass through byte for byte; a refused URL is an error.
	plain := []byte(`{"state": "x", "images": ["data:image/png;base64,AA=="]}`)
	if nb, _ := inlineProxyBodyImages(context.Background(), "/v1/systemone", plain); string(nb) != string(plain) {
		t.Fatalf("unchanged body rewritten: %s", nb)
	}
	if _, err := inlineProxyBodyImages(context.Background(), "/v1/systemone", []byte(`{"images": ["`+srv.URL+`/page"]}`)); err == nil {
		t.Fatal("non-image URL accepted")
	}
}

// `dgem systemone serve` and `dgem serve` install the guarded fetch client in pkg/client: a request's image URL that
// points at an internal address is refused at connect time, and the internal server is never hit.
func TestClientImageFetchGuardedInServerMode(t *testing.T) {
	serverMode(t)
	hit := false
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hit = true }))
	defer internal.Close()
	old := client.ImageHTTPClient
	client.ImageHTTPClient = imageFetchClient
	t.Cleanup(func() { client.ImageHTTPClient = old })
	_, err := client.BuildMultimodalContent("{}", []string{internal.URL + "/img.png"})
	if err == nil || hit || !errors.Is(err, client.ErrImageFetch) {
		t.Fatalf("internal fetch must be refused: err=%v hit=%v", err, hit)
	}
}
