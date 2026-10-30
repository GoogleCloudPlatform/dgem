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
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"syscall"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/decisionindex"
)

// allowLocalImagePaths lets image references name local files. It is true for the CLI and stdio MCP (the files are
// the caller's own) and set to false by `dgem serve`, where references come from network requests: there a local path
// would read the gateway's own filesystem and send the bytes to a model.
var allowLocalImagePaths = true

// checkRequestImages validates image references from a request: data: URIs always; http(s) URLs whose host does not
// resolve to a loopback, private, link-local or unspecified address (the gateway would otherwise fetch internal
// endpoints such as the metadata server for the Stage-2 cascade); local paths only when allowLocalImagePaths.
func checkRequestImages(ctx context.Context, images []string) error {
	for _, ref := range images {
		ref = strings.TrimSpace(ref)
		if err := client.CheckImageFormat(ref, nil); err != nil {
			return err
		}
		switch {
		case ref == "", strings.HasPrefix(ref, "data:"):
			continue
		case strings.HasPrefix(ref, "http://"), strings.HasPrefix(ref, "https://"):
			if err := imageHostCheck(ctx, ref); err != nil {
				return err
			}
		default:
			if !allowLocalImagePaths {
				return fmt.Errorf("image must be a data: URI or an http(s) URL on this server")
			}
		}
	}
	return nil
}

// checkSystemOneImages validates the "images" of a /v1/systemone request before the adapter forwards them upstream:
// the adapter's own rule (data: URIs or http(s) URLs only), then the gateway's public-host check.
func checkSystemOneImages(ctx context.Context, images []string) error {
	if err := decisionindex.ValidateImages(images); err != nil {
		return err
	}
	return checkRequestImages(ctx, images)
}

// imageHostCheck is the request-time host check for image URLs (a variable so tests can serve images from loopback).
var imageHostCheck = checkPublicImageURL

func checkPublicImageURL(ctx context.Context, ref string) error {
	u, err := url.Parse(ref)
	if err != nil || u.Hostname() == "" {
		return fmt.Errorf("image URL is not valid")
	}
	if allowLocalImagePaths {
		return nil // CLI: the caller's own network
	}
	ips, err := net.DefaultResolver.LookupIPAddr(ctx, u.Hostname())
	if err != nil {
		return fmt.Errorf("image URL host does not resolve")
	}
	for _, ip := range ips {
		a, ok := netip.AddrFromSlice(ip.IP)
		if !ok || !isPublicIP(a) {
			return fmt.Errorf("image URL must point to a public host")
		}
	}
	return nil
}

var nonPublicPrefixes = func() []netip.Prefix {
	var out []netip.Prefix
	for _, s := range []string{
		"0.0.0.0/8",       // "this network"
		"100.64.0.0/10",   // carrier-grade NAT (RFC 6598)
		"192.0.0.0/24",    // IETF protocol assignments
		"192.0.2.0/24",    // TEST-NET-1
		"198.18.0.0/15",   // benchmarking
		"198.51.100.0/24", // TEST-NET-2
		"203.0.113.0/24",  // TEST-NET-3
		"240.0.0.0/4",     // reserved, includes 255.255.255.255
		"fec0::/10",       // deprecated site-local
		"100::/64",        // discard-only
		"2001:db8::/32",   // documentation
	} {
		out = append(out, netip.MustParsePrefix(s))
	}
	return out
}()

var (
	nat64Prefix      = netip.MustParsePrefix("64:ff9b::/96")
	nat64LocalPrefix = netip.MustParsePrefix("64:ff9b:1::/48")
	sixToFourPrefix  = netip.MustParsePrefix("2002::/16")
)

// isPublicIP reports whether a is a globally routable unicast address: not loopback, private, link-local (including
// the 169.254.169.254 metadata server), CGNAT, multicast, unspecified or reserved. IPv4-mapped, NAT64 and 6to4
// addresses are judged by the IPv4 address they embed.
func isPublicIP(a netip.Addr) bool {
	a = a.Unmap()
	if !a.IsValid() || a.Zone() != "" {
		return false
	}
	if a.Is6() {
		b := a.As16()
		switch {
		case nat64Prefix.Contains(a):
			return isPublicIP(netip.AddrFrom4([4]byte{b[12], b[13], b[14], b[15]}))
		case nat64LocalPrefix.Contains(a):
			return false
		case sixToFourPrefix.Contains(a):
			return isPublicIP(netip.AddrFrom4([4]byte{b[2], b[3], b[4], b[5]}))
		}
	}
	if a.IsLoopback() || a.IsPrivate() || a.IsLinkLocalUnicast() || a.IsLinkLocalMulticast() ||
		a.IsInterfaceLocalMulticast() || a.IsMulticast() || a.IsUnspecified() {
		return false
	}
	for _, p := range nonPublicPrefixes {
		if p.Contains(a) {
			return false
		}
	}
	return true
}

var errNonPublicImageAddr = errors.New("image URL must point to a public host")

// imageDialGuard vets the address a server-side image fetch is about to connect to. It runs after DNS resolution,
// for every connection including redirects, so a host that re-resolves to an internal address between
// checkRequestImages and the fetch (DNS rebinding) is still refused. Tests replace it.
var imageDialGuard = func(ap netip.AddrPort) error {
	if !isPublicIP(ap.Addr()) {
		return errNonPublicImageAddr
	}
	return nil
}

const maxImageRedirects = 3

// imageFetchClient returns the HTTP client for fetching image URLs. The CLI and stdio MCP fetch from the caller's own
// network with a plain client. Under `dgem serve` (allowLocalImagePaths false) the URL comes from a network request, so
// the client ignores proxy settings, checks every dialed IP with imageDialGuard, and follows at most
// maxImageRedirects http(s) redirects, each going through the same dial check.
func imageFetchClient(timeout time.Duration) *http.Client {
	if allowLocalImagePaths {
		return &http.Client{Timeout: timeout}
	}
	dialer := &net.Dialer{
		Timeout:   10 * time.Second,
		KeepAlive: 30 * time.Second,
		Control: func(network, address string, _ syscall.RawConn) error {
			ap, err := netip.ParseAddrPort(address)
			if err != nil {
				return fmt.Errorf("image fetch: unexpected address %q", address)
			}
			return imageDialGuard(ap)
		},
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil // a proxy would dial on our behalf and bypass the address check
	tr.DialContext = dialer.DialContext
	return &http.Client{
		Timeout:   timeout,
		Transport: tr,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) > maxImageRedirects {
				return fmt.Errorf("image URL: more than %d redirects", maxImageRedirects)
			}
			if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
				return fmt.Errorf("image URL: redirect to unsupported scheme %q", req.URL.Scheme)
			}
			return nil
		},
	}
}
