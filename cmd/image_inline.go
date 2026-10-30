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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// Under `dgem serve` the gateway fetches image URLs itself (imageFetchClient: public addresses only, checked at connect
// time and on every redirect) and forwards them as data: URIs, so the serving image never fetches a caller-supplied
// URL. The serving image (vLLM) runs with VPC egress; a URL that passed the gateway's request-time host check could
// rebind or redirect to the metadata server or internal hosts when vLLM fetched it.

// maxImageBytes caps one fetched image (the same limit the Stage-2 cascade used).
const maxImageBytes = 20 << 20

// ErrImageFetch marks an image URL the gateway could not fetch or that is not an image: the caller's input, so 422.
// It is pkg/client's sentinel, so errors from either fetch path compare equal.
var ErrImageFetch = client.ErrImageFetch

// fetchImageURL downloads an http(s) image with imageFetchClient and returns its bytes and MIME type.
func fetchImageURL(ctx context.Context, ref string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ref, nil)
	if err != nil {
		return nil, "", fmt.Errorf("%w: %w", ErrImageFetch, err)
	}
	hr, err := imageFetchClient(30 * time.Second).Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("%w: %w", ErrImageFetch, err)
	}
	defer hr.Body.Close()
	if hr.StatusCode < 200 || hr.StatusCode > 299 {
		return nil, "", fmt.Errorf("%w: HTTP %d", ErrImageFetch, hr.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(hr.Body, maxImageBytes+1))
	if err != nil {
		return nil, "", fmt.Errorf("%w: %w", ErrImageFetch, err)
	}
	if len(data) > maxImageBytes {
		return nil, "", fmt.Errorf("%w: larger than %d MB", ErrImageFetch, maxImageBytes>>20)
	}
	mime := strings.TrimSpace(strings.SplitN(hr.Header.Get("Content-Type"), ";", 2)[0])
	if !strings.HasPrefix(mime, "image/") {
		mime = http.DetectContentType(data)
	}
	return data, mime, nil
}

// inlineImageURL fetches an http(s) image and returns it as a data: URI. Content that isn't a raster image (HTML, SVG,
// ...) is refused.
func inlineImageURL(ctx context.Context, ref string) (string, error) {
	data, mime, err := fetchImageURL(ctx, ref)
	if err != nil {
		return "", err
	}
	if sniff := http.DetectContentType(data); !strings.HasPrefix(mime, "image/") || !strings.HasPrefix(sniff, "image/") {
		return "", fmt.Errorf("%w: %s is not an image (%s)", ErrImageFetch, ref, sniff)
	}
	uri := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
	if err := client.CheckImageFormat(uri, data); err != nil {
		return "", err
	}
	return uri, nil
}

func isHTTPImageRef(ref string) bool {
	r := strings.ToLower(strings.TrimSpace(ref))
	return strings.HasPrefix(r, "http://") || strings.HasPrefix(r, "https://")
}

// prepareRequestImages validates image references (checkRequestImages) and, under `dgem serve`, replaces every
// http(s) URL with the fetched image as a data: URI. The CLI and stdio MCP return the references unchanged.
func prepareRequestImages(ctx context.Context, images []string) ([]string, error) {
	if err := checkRequestImages(ctx, images); err != nil {
		return nil, err
	}
	if allowLocalImagePaths {
		return images, nil
	}
	out := make([]string, len(images))
	for i, ref := range images {
		out[i] = ref
		if isHTTPImageRef(ref) {
			uri, err := inlineImageURL(ctx, strings.TrimSpace(ref))
			if err != nil {
				return nil, err
			}
			out[i] = uri
		}
	}
	return out, nil
}

// imageErrorStatus is the HTTP status for a prepareRequestImages error: 422 when the URL couldn't be fetched or isn't
// an image, 400 for a refused reference (local path, private host, SVG, ...).
func imageErrorStatus(err error) int {
	if errors.Is(err, ErrImageFetch) {
		return http.StatusUnprocessableEntity
	}
	return http.StatusBadRequest
}

// inlineProxyBodyImages rewrites a JSON /v1/systemone or /v1/chat/completions body so every http(s) image URL becomes
// a data: URI (prepareRequestImages): the top-level "images" of a /v1/systemone body, and image_url parts of chat
// messages. Other fields are passed through byte for byte. It returns the body unchanged when it has no image URLs or
// is not a JSON object (the upstream reports malformed bodies).
func inlineProxyBodyImages(ctx context.Context, path string, body []byte) ([]byte, error) {
	if !bytes.Contains(body, []byte("http")) {
		return body, nil
	}
	var top map[string]json.RawMessage
	if json.Unmarshal(body, &top) != nil {
		return body, nil
	}
	changed := false
	switch path {
	case "/v1/systemone":
		var imgs []json.RawMessage
		if raw, ok := top["images"]; ok && json.Unmarshal(raw, &imgs) == nil {
			for i, im := range imgs {
				var s string
				if json.Unmarshal(im, &s) != nil || !isHTTPImageRef(s) {
					continue
				}
				out, err := prepareRequestImages(ctx, []string{s})
				if err != nil {
					return nil, err
				}
				imgs[i], _ = json.Marshal(out[0])
				changed = true
			}
			if changed {
				top["images"], _ = json.Marshal(imgs)
			}
		}
	case "/v1/chat/completions":
		var msgs []map[string]json.RawMessage
		if raw, ok := top["messages"]; ok && json.Unmarshal(raw, &msgs) == nil {
			for mi, m := range msgs {
				var parts []map[string]json.RawMessage
				if json.Unmarshal(m["content"], &parts) != nil {
					continue
				}
				partChanged := false
				for pi, p := range parts {
					var iu struct {
						URL string `json:"url"`
					}
					if json.Unmarshal(p["image_url"], &iu) != nil || !isHTTPImageRef(iu.URL) {
						continue
					}
					out, err := prepareRequestImages(ctx, []string{iu.URL})
					if err != nil {
						return nil, err
					}
					var obj map[string]json.RawMessage
					_ = json.Unmarshal(p["image_url"], &obj)
					obj["url"], _ = json.Marshal(out[0])
					parts[pi]["image_url"], _ = json.Marshal(obj)
					partChanged = true
				}
				if partChanged {
					msgs[mi]["content"], _ = json.Marshal(parts)
					changed = true
				}
			}
			if changed {
				top["messages"], _ = json.Marshal(msgs)
			}
		}
	}
	if !changed {
		return body, nil
	}
	return json.Marshal(top)
}
