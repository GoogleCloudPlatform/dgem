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
	"fmt"
	"net"
	"net/url"
	"strings"
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
		switch {
		case ref == "", strings.HasPrefix(ref, "data:"):
			continue
		case strings.HasPrefix(ref, "http://"), strings.HasPrefix(ref, "https://"):
			if err := checkPublicImageURL(ctx, ref); err != nil {
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
		a := ip.IP
		if a.IsLoopback() || a.IsPrivate() || a.IsLinkLocalUnicast() || a.IsLinkLocalMulticast() || a.IsUnspecified() {
			return fmt.Errorf("image URL must point to a public host")
		}
	}
	return nil
}
