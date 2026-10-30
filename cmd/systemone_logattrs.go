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
	"encoding/json"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/pkg/decisionindex"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// failSpan marks a gateway span as failed with the HTTP status the caller gets and the reason. The span log line then
// carries severity WARNING/ERROR and the reason as "error" (see spanSeverity), so a refused request can be diagnosed
// from the logs alone.
func failSpan(span trace.Span, status int, reason string) {
	span.SetAttributes(attribute.Int("http.status_code", status))
	span.SetStatus(codes.Error, reason)
}

// systemOneBodyAttrs are attributes of a raw /v1/systemone body that survive decoding: its size and how many "media"
// entries it has (MergeMedia folds them into images). No content.
func systemOneBodyAttrs(body []byte) []attribute.KeyValue {
	attrs := []attribute.KeyValue{attribute.Int("dgem.systemone.body_bytes", len(body))}
	var top struct {
		Media []json.RawMessage `json:"media"`
	}
	if json.Unmarshal(body, &top) == nil {
		attrs = append(attrs, attribute.Int("dgem.systemone.media_entries", len(top.Media)))
	}
	return attrs
}

// systemOneShapeAttrs describe a decoded request without its content: question counts per type, the widest
// question, and the image count (images plus merged media).
func systemOneShapeAttrs(req decisionindex.SystemOneRequest) []attribute.KeyValue {
	var choice, noul, score, other, maxOpts int
	for _, q := range req.Questions {
		switch t := strings.ToLower(strings.TrimSpace(q.Type)); {
		case t == "noul" || t == "bool" || t == "boolean":
			noul++
		case t == "score" || (t == "" && q.Levels != nil):
			score++
		case t == "choice" || t == "":
			choice++
		default:
			other++
		}
		maxOpts = max(maxOpts, len(q.Criteria), len(q.Levels))
	}
	return []attribute.KeyValue{
		attribute.Int("dgem.systemone.questions", len(req.Questions)),
		attribute.Int("dgem.systemone.q_choice", choice),
		attribute.Int("dgem.systemone.q_noul", noul),
		attribute.Int("dgem.systemone.q_score", score),
		attribute.Int("dgem.systemone.q_other", other),
		attribute.Int("dgem.systemone.max_options", maxOpts),
		attribute.Int("dgem.image_count", len(req.Images)),
	}
}
