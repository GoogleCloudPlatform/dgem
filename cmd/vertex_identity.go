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
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

// vertexIdentity is the endpoint a Vertex URL actually reaches: the host the request (and the gateway's access token)
// is sent to, plus the project, location and endpoint ID the path names.
type vertexIdentity struct {
	Host       string // lower-case host name
	Project    string // project ID or number from the path
	Location   string // region from the path
	EndpointID string
	Dedicated  bool // <ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog
	ProjectNum string
}

const (
	dedicatedVertexSuffix = ".prediction.vertexai.goog"
	regionalVertexSuffix  = "-aiplatform.googleapis.com"
	globalVertexHost      = "aiplatform.googleapis.com"
)

var (
	vertexSegmentRe  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*$`)
	vertexEndpointRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*$`)
	vertexRegionRe   = regexp.MustCompile(`^[a-z]+(-[a-z0-9]+)+$`)
	vertexNumberRe   = regexp.MustCompile(`^[0-9]+$`)
	vertexMethods    = []string{":rawPredict", ":streamRawPredict", ":predict"}
)

// parseVertexEndpointIdentity parses a Vertex AI endpoint URL strictly. Accepted shapes:
//
//	https://<ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog/v1/projects/<P>/locations/<REGION>/endpoints/<ID>/invoke/...
//	https://<REGION>-aiplatform.googleapis.com/v1/projects/<P>/locations/<REGION>/endpoints/<ID>:rawPredict (or :predict, :streamRawPredict, /invoke/...)
//
// Query strings, fragments, user info, non-443 ports, escaped or dot path segments are rejected, and on a dedicated
// host the endpoint ID, region and (numeric) project in the path must agree with the host name. It never falls back
// to the configured endpoint.
func parseVertexEndpointIdentity(raw string) (vertexIdentity, error) {
	var id vertexIdentity
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return id, fmt.Errorf("not a URL")
	}
	if u.Scheme != "https" {
		return id, fmt.Errorf("scheme must be https")
	}
	if u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.Opaque != "" {
		return id, fmt.Errorf("user info, query strings and fragments are not allowed")
	}
	if p := u.Port(); p != "" && p != "443" {
		return id, fmt.Errorf("port %s is not allowed", p)
	}
	if u.RawPath != "" || strings.Contains(u.EscapedPath(), "%") {
		return id, fmt.Errorf("escaped path characters are not allowed")
	}
	host := strings.ToLower(u.Hostname())

	segs := strings.Split(strings.TrimPrefix(u.Path, "/"), "/")
	if len(segs) < 7 || !strings.HasPrefix(segs[0], "v1") || segs[1] != "projects" || segs[3] != "locations" || segs[5] != "endpoints" {
		return id, fmt.Errorf("path must be /v1/projects/<project>/locations/<region>/endpoints/<id>/...")
	}
	for _, s := range segs {
		if s == "" || s == "." || s == ".." {
			return id, fmt.Errorf("empty or dot path segments are not allowed")
		}
	}
	id.Project, id.Location = segs[2], segs[4]
	epSeg, tail := segs[6], segs[7:]
	method := ""
	for _, m := range vertexMethods {
		if strings.HasSuffix(epSeg, m) {
			method, epSeg = m, strings.TrimSuffix(epSeg, m)
			break
		}
	}
	switch {
	case method != "" && len(tail) == 0:
	case method == "" && len(tail) >= 1 && tail[0] == "invoke":
	default:
		return id, fmt.Errorf("path must end in :rawPredict / :predict or continue with /invoke/...")
	}
	for _, s := range tail {
		if !vertexSegmentRe.MatchString(s) {
			return id, fmt.Errorf("invalid path segment %q", s)
		}
	}
	if !vertexEndpointRe.MatchString(epSeg) || !vertexSegmentRe.MatchString(id.Project) || !vertexRegionRe.MatchString(id.Location) {
		return id, fmt.Errorf("invalid project, region or endpoint ID")
	}
	id.EndpointID = epSeg
	id.Host = host

	switch {
	case strings.HasSuffix(host, dedicatedVertexSuffix):
		// <ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog
		labels := strings.Split(strings.TrimSuffix(host, dedicatedVertexSuffix), ".")
		if len(labels) != 2 {
			return id, fmt.Errorf("dedicated endpoint host must be <id>.<region>-<project_number>%s", dedicatedVertexSuffix)
		}
		dash := strings.LastIndex(labels[1], "-")
		if dash <= 0 {
			return id, fmt.Errorf("dedicated endpoint host must be <id>.<region>-<project_number>%s", dedicatedVertexSuffix)
		}
		hostRegion, hostProjNum := labels[1][:dash], labels[1][dash+1:]
		if !vertexNumberRe.MatchString(hostProjNum) {
			return id, fmt.Errorf("dedicated endpoint host must carry the project number")
		}
		if labels[0] != strings.ToLower(id.EndpointID) || hostRegion != id.Location {
			return id, fmt.Errorf("endpoint ID or region in the path does not match the host")
		}
		if vertexNumberRe.MatchString(id.Project) && id.Project != hostProjNum {
			return id, fmt.Errorf("project number in the path does not match the host")
		}
		id.Dedicated, id.ProjectNum = true, hostProjNum
	case host == globalVertexHost:
	case strings.HasSuffix(host, regionalVertexSuffix):
		if strings.TrimSuffix(host, regionalVertexSuffix) != id.Location {
			return id, fmt.Errorf("region in the host does not match the path")
		}
	default:
		return id, fmt.Errorf("host must be <id>.<region>-<project_number>%s or <region>%s", dedicatedVertexSuffix, regionalVertexSuffix)
	}
	if id.ProjectNum == "" && vertexNumberRe.MatchString(id.Project) {
		id.ProjectNum = id.Project
	}
	return id, nil
}

// projectKey is the project as a number when known, so a URL naming the gateway's own project by ID or by number
// compares equal.
func (v vertexIdentity) projectKey() string {
	if v.ProjectNum != "" {
		return v.ProjectNum
	}
	if pid := detectGCPProjectID(); pid != "" && strings.EqualFold(pid, v.Project) {
		if num := detectGCPProjectNumber(); num != "" {
			return num
		}
	}
	return strings.ToLower(v.Project)
}

// sameVertexEndpoint reports whether two identities reach the same endpoint: same project, region and endpoint ID,
// and, for dedicated endpoints, the same host (the dedicated host receives the gateway's token directly).
func sameVertexEndpoint(a, b vertexIdentity) bool {
	if a.EndpointID != b.EndpointID || a.Location != b.Location || a.projectKey() != b.projectKey() {
		return false
	}
	if a.Dedicated && b.Dedicated {
		return a.Host == b.Host
	}
	return true
}
