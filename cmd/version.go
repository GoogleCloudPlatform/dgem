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
	"regexp"
	"runtime/debug"
	"strings"
)

// Build metadata, set at build time with
//
//	-ldflags "-X github.com/GoogleCloudPlatform/dgem/cmd.Version=v0.1.0 -X github.com/GoogleCloudPlatform/dgem/cmd.Commit=<sha> -X github.com/GoogleCloudPlatform/dgem/cmd.Date=<rfc3339>"
//
// Version is `git describe --tags --match 'v*'`: a release tag such as v0.4.0 on a release build, or
// v0.4.0-3-gabc1234 for commits after it. Unset values fall back to Go's embedded module/VCS BuildInfo
// so `go install github.com/GoogleCloudPlatform/dgem@vX.Y.Z` reports vX.Y.Z rather than "dev".
var (
	Version = "dev"
	Commit  = ""
	Date    = ""

	readBuildInfo = debug.ReadBuildInfo
	semverTagRe   = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$`)
)

func isSemverTag(v string) bool {
	return semverTagRe.MatchString(strings.TrimSpace(v))
}

func buildVersion() string {
	if Version != "" && Version != "dev" {
		return Version
	}
	if bi, ok := readBuildInfo(); ok && bi != nil && isSemverTag(bi.Main.Version) {
		return strings.TrimSpace(bi.Main.Version)
	}
	if Version != "" {
		return Version
	}
	return "dev"
}

func buildRevision() string {
	if Commit != "" {
		return Commit
	}
	if bi, ok := readBuildInfo(); ok && bi != nil {
		for _, s := range bi.Settings {
			if s.Key == "vcs.revision" && s.Value != "" {
				if len(s.Value) >= 7 {
					return s.Value[:7]
				}
				return s.Value
			}
		}
	}
	return ""
}

// VersionInfo is reported by `dgem --version` and the gateway's /health.
func VersionInfo() map[string]string {
	return map[string]string{"version": buildVersion(), "revision": buildRevision(), "build_date": Date}
}

func versionString() string {
	ver := buildVersion()
	var meta []string
	if rev := buildRevision(); rev != "" {
		meta = append(meta, "revision "+rev)
	}
	if Date != "" {
		meta = append(meta, "built "+Date)
	}
	if len(meta) == 0 {
		return ver
	}
	return fmt.Sprintf("%s (%s)", ver, strings.Join(meta, ", "))
}

func init() {
	RootCmd.Version = versionString()
}
