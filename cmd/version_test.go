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
	"net/http"
	"net/http/httptest"
	"runtime/debug"
	"testing"
)

func withVersionState(t *testing.T, ver, commit, date string, bi *debug.BuildInfo) {
	t.Helper()
	oldVer, oldCommit, oldDate, oldRead := Version, Commit, Date, readBuildInfo
	Version = ver
	Commit = commit
	Date = date
	readBuildInfo = func() (*debug.BuildInfo, bool) {
		if bi == nil {
			return nil, false
		}
		return bi, true
	}
	t.Cleanup(func() {
		Version = oldVer
		Commit = oldCommit
		Date = oldDate
		readBuildInfo = oldRead
	})
}

func TestVersionStringAndInfo(t *testing.T) {
	tests := []struct {
		name     string
		ver      string
		commit   string
		date     string
		bi       *debug.BuildInfo
		wantStr  string
		wantVer  string
		wantRev  string
	}{
		{
			name:   "make build with ldflags",
			ver:    "v0.4.0",
			commit: "4368d3c",
			date:   "2026-11-06T00:00:00Z",
			bi: &debug.BuildInfo{
				Main: debug.Module{Version: "(devel)"},
				Settings: []debug.BuildSetting{
					{Key: "vcs.revision", Value: "9999999999999999"},
				},
			},
			wantStr: "v0.4.0 (revision 4368d3c, built 2026-11-06T00:00:00Z)",
			wantVer: "v0.4.0",
			wantRev: "4368d3c",
		},
		{
			name: "go install release tag without vcs.revision",
			ver:  "dev",
			bi: &debug.BuildInfo{
				Main: debug.Module{Path: "github.com/GoogleCloudPlatform/dgem", Version: "v0.4.0"},
			},
			wantStr: "v0.4.0",
			wantVer: "v0.4.0",
			wantRev: "",
		},
		{
			name: "go install pseudo-version without vcs.revision",
			ver:  "dev",
			bi: &debug.BuildInfo{
				Main: debug.Module{Path: "github.com/GoogleCloudPlatform/dgem", Version: "v0.4.1-0.20261106010203-4368d3c12345"},
			},
			wantStr: "v0.4.1-0.20261106010203-4368d3c12345",
			wantVer: "v0.4.1-0.20261106010203-4368d3c12345",
			wantRev: "",
		},
		{
			name: "local go build with devel module and vcs.revision",
			ver:  "dev",
			bi: &debug.BuildInfo{
				Main: debug.Module{Version: "(devel)"},
				Settings: []debug.BuildSetting{
					{Key: "vcs.revision", Value: "4368d3c89abcdef0"},
				},
			},
			wantStr: "dev (revision 4368d3c)",
			wantVer: "dev",
			wantRev: "4368d3c",
		},
		{
			name: "local go build outside vcs with devel module",
			ver:  "dev",
			bi: &debug.BuildInfo{
				Main: debug.Module{Version: "(devel)"},
			},
			wantStr: "dev",
			wantVer: "dev",
			wantRev: "",
		},
		{
			name:    "no build info available",
			ver:     "dev",
			bi:      nil,
			wantStr: "dev",
			wantVer: "dev",
			wantRev: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			withVersionState(t, tc.ver, tc.commit, tc.date, tc.bi)
			if got := versionString(); got != tc.wantStr {
				t.Errorf("versionString() = %q, want %q", got, tc.wantStr)
			}
			info := VersionInfo()
			if info["version"] != tc.wantVer {
				t.Errorf("VersionInfo()[version] = %q, want %q", info["version"], tc.wantVer)
			}
			if info["revision"] != tc.wantRev {
				t.Errorf("VersionInfo()[revision] = %q, want %q", info["revision"], tc.wantRev)
			}
		})
	}
}

func TestHealthEndpointsUseModuleBuildVersion(t *testing.T) {
	withVersionState(t, "dev", "", "", &debug.BuildInfo{
		Main: debug.Module{Path: "github.com/GoogleCloudPlatform/dgem", Version: "v0.4.0"},
	})

	for _, tc := range []struct {
		name    string
		handler http.HandlerFunc
	}{
		{name: "gateway /health", handler: gatewayHealth},
		{name: "systemone /health", handler: systemOneHealth},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.handler(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200", rec.Code)
			}
			var got map[string]any
			if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if got["version"] != "v0.4.0" {
				t.Errorf("version = %v, want v0.4.0", got["version"])
			}
			if got["revision"] != "" {
				t.Errorf("revision = %v, want empty string", got["revision"])
			}
		})
	}
}
