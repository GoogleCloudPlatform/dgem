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

package client

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// X-DGem-Key is sent only when configured (DGEM_SERVER_KEY), and only the first entry of a rotation list.
func TestServerKeyHeader(t *testing.T) {
	for _, tc := range []struct{ env, want string }{{"", ""}, {"k1", "k1"}, {" new , old ", "new"}} {
		t.Setenv("DGEM_SERVER_KEY", tc.env)
		var got string
		var present bool
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			got = r.Header.Get(ServerKeyHeader)
			_, present = r.Header[http.CanonicalHeaderKey(ServerKeyHeader)]
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"{\"answers\":{\"q\":{\"type\":\"choice\",\"choice\":\"a\",\"probabilities\":{\"a\":1}}}}"}}]}`))
		}))
		cli := NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
		if _, _, err := cli.Decide(context.Background(), `{"questions":[]}`, `{}`); err != nil {
			t.Fatal(err)
		}
		srv.Close()
		if got != tc.want || present != (tc.want != "") {
			t.Errorf("env %q: header %q (present %v), want %q", tc.env, got, present, tc.want)
		}
	}
}
