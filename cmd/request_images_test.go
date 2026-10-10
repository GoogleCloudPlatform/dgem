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
	"errors"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"
)

func TestCheckRequestImages(t *testing.T) {
	ctx := context.Background()
	defer func() { allowLocalImagePaths = true }()

	allowLocalImagePaths = true
	if err := checkRequestImages(ctx, []string{"fixtures/x.png", "data:image/png;base64,AA=="}); err != nil {
		t.Fatalf("CLI must allow local paths and data URIs: %v", err)
	}

	allowLocalImagePaths = false
	for _, bad := range []string{"/etc/passwd", "../../secret.json", "http://127.0.0.1/x.png", "http://169.254.169.254/computeMetadata/v1/",
		"http://10.0.0.5/a.png", "http://[::1]/a.png", "http://localhost/a.png"} {
		if err := checkRequestImages(ctx, []string{bad}); err == nil {
			t.Fatalf("server must reject %q", bad)
		}
	}
	if err := checkRequestImages(ctx, []string{"data:image/png;base64,AA==", ""}); err != nil {
		t.Fatalf("server must allow data URIs: %v", err)
	}
}

func TestCheckSystemOneImages(t *testing.T) {
	ctx := context.Background()
	defer func() { allowLocalImagePaths = true }()
	allowLocalImagePaths = false
	if err := checkSystemOneImages(ctx, []string{"data:image/png;base64,AA==", ""}); err != nil {
		t.Fatalf("data URIs must pass: %v", err)
	}
	for _, bad := range []string{"/etc/passwd", "http://169.254.169.254/computeMetadata/v1/", "http://localhost/a.png"} {
		if err := checkSystemOneImages(ctx, []string{bad}); err == nil {
			t.Fatalf("gateway must reject %q", bad)
		}
	}
	// The adapter never reads local paths, even where the CLI would allow them.
	allowLocalImagePaths = true
	if err := checkSystemOneImages(ctx, []string{"fixtures/x.png"}); err == nil {
		t.Fatal("systemone images must not be local paths")
	}
}

// SVG references are refused at the gateway with a 400-able error before any model call (#117).
func TestCheckRequestImagesRejectsSVG(t *testing.T) {
	ctx := context.Background()
	for _, ref := range []string{"data:image/svg+xml;base64,PHN2Zy8+", "fixtures/ui_component.svg"} {
		if err := checkRequestImages(ctx, []string{ref}); err == nil || !strings.Contains(err.Error(), "SVG") {
			t.Errorf("%s: err = %v, want an SVG refusal", ref, err)
		}
	}
	if err := checkSystemOneImages(ctx, []string{"data:image/svg+xml,<svg/>"}); err == nil {
		t.Error("systemone: SVG data URI accepted")
	}
}

func TestIsPublicIP(t *testing.T) {
	for _, s := range []string{"127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.0.1", "169.254.169.254", "100.64.0.1",
		"0.0.0.0", "0.1.2.3", "255.255.255.255", "224.0.0.1", "::1", "::", "fe80::1", "fc00::1", "fd00:ec2::254",
		"::ffff:127.0.0.1", "::ffff:169.254.169.254", "64:ff9b::a9fe:a9fe", "64:ff9b::7f00:1", "2002:7f00:1::1", "2002:a9fe:a9fe::"} {
		if isPublicIP(netip.MustParseAddr(s)) {
			t.Errorf("%s must not be public", s)
		}
	}
	for _, s := range []string{"8.8.8.8", "142.250.80.46", "2607:f8b0:4004:800::200e", "::ffff:8.8.8.8", "64:ff9b::808:808"} {
		if !isPublicIP(netip.MustParseAddr(s)) {
			t.Errorf("%s must be public", s)
		}
	}
}

func serverMode(t *testing.T) {
	t.Helper()
	old := allowLocalImagePaths
	allowLocalImagePaths = false
	t.Cleanup(func() { allowLocalImagePaths = old })
}

// A host that passes checkRequestImages but resolves to an internal address when the fetch connects (DNS rebinding)
// is refused at connect time: the test server listens on loopback, as a rebound name would.
func TestImageFetchRefusesInternalAddressAtDial(t *testing.T) {
	serverMode(t)
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hit = true }))
	defer srv.Close()
	_, err := cascadeImageParts(context.Background(), []string{srv.URL + "/img.png"})
	if err == nil || !errors.Is(err, errNonPublicImageAddr) || hit {
		t.Fatalf("loopback fetch must be refused before connecting: err=%v hit=%v", err, hit)
	}
}

// A public host that redirects to an internal address is refused: every redirect hop goes through the dial check.
func TestImageFetchRefusesRedirectToInternal(t *testing.T) {
	serverMode(t)
	internalHit := false
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { internalHit = true }))
	defer internal.Close()
	public := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, internal.URL+"/computeMetadata/v1/", http.StatusFound)
	}))
	defer public.Close()
	publicAddr := netip.MustParseAddrPort(public.Listener.Addr().String())

	old := imageDialGuard
	imageDialGuard = func(ap netip.AddrPort) error { // treat only the "public" test server as public
		if ap == publicAddr {
			return nil
		}
		return errNonPublicImageAddr
	}
	t.Cleanup(func() { imageDialGuard = old })

	_, err := cascadeImageParts(context.Background(), []string{public.URL + "/img.png"})
	if err == nil || !errors.Is(err, errNonPublicImageAddr) || internalHit {
		t.Fatalf("redirect to internal address must be refused: err=%v internalHit=%v", err, internalHit)
	}
}

func TestImageFetchRedirectLimitAndStatus(t *testing.T) {
	serverMode(t)
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/loop":
			http.Redirect(w, r, srv.URL+"/loop", http.StatusFound)
		case "/missing":
			http.NotFound(w, r)
		default:
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte("\x89PNG\r\n\x1a\n"))
		}
	}))
	defer srv.Close()
	old := imageDialGuard
	imageDialGuard = func(netip.AddrPort) error { return nil }
	t.Cleanup(func() { imageDialGuard = old })

	if _, err := cascadeImageParts(context.Background(), []string{srv.URL + "/loop"}); err == nil || !strings.Contains(err.Error(), "redirects") {
		t.Fatalf("redirect loop: %v", err)
	}
	if _, err := cascadeImageParts(context.Background(), []string{srv.URL + "/missing"}); err == nil || !strings.Contains(err.Error(), "HTTP 404") {
		t.Fatalf("404 must fail: %v", err)
	}
	parts, err := cascadeImageParts(context.Background(), []string{srv.URL + "/ok.png", "data:image/png;base64,AA=="})
	if err != nil || len(parts) != 2 {
		t.Fatalf("allowed fetch: %v %d", err, len(parts))
	}
}

func TestCascadeImagePartsRefusesLocalPathsOnServer(t *testing.T) {
	serverMode(t)
	if _, err := cascadeImageParts(context.Background(), []string{"/etc/hostname"}); err == nil {
		t.Fatal("server must refuse local image paths")
	}
}
