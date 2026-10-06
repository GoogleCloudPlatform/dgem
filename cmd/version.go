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
	"runtime/debug"
)

// Build metadata, set at build time with
//
//	-ldflags "-X github.com/GoogleCloudPlatform/dgem/cmd.Version=v0.1.0 -X github.com/GoogleCloudPlatform/dgem/cmd.Commit=<sha> -X github.com/GoogleCloudPlatform/dgem/cmd.Date=<rfc3339>"
//
// Version is `git describe --tags --match 'v*'`: a release tag such as v0.1.0 on a release build, or
// v0.1.0-3-gabc1234 for commits after it. Unset values fall back to Go's embedded VCS info.
var (
	Version = "dev"
	Commit  = ""
	Date    = ""
)

func buildRevision() string {
	if Commit != "" {
		return Commit
	}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "vcs.revision" && len(s.Value) >= 7 {
				return s.Value[:7]
			}
		}
	}
	return "unknown"
}

// VersionInfo is reported by `dgem --version` and the gateway's /health.
func VersionInfo() map[string]string {
	return map[string]string{"version": Version, "revision": buildRevision(), "build_date": Date}
}

func versionString() string {
	s := fmt.Sprintf("%s (revision %s", Version, buildRevision())
	if Date != "" {
		s += ", built " + Date
	}
	return s + ")"
}

func init() {
	RootCmd.Version = versionString()
}
