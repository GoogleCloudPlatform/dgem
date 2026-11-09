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
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func jsonHTTPResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func saveAndRestoreVertexStatusGlobals(t *testing.T) {
	t.Helper()
	vertexStatusCacheMu.Lock()
	oldCached := vertexStatusCached
	oldExp := vertexStatusCacheExpires
	oldStale := vertexStatusStaleUntil
	vertexStatusCacheMu.Unlock()

	oldClient := vertexStatusHTTPClient

	t.Cleanup(func() {
		// Wait briefly for any in-flight refresh goroutines to drain before restoring globals.
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			vertexStatusRefreshMu.Lock()
			n := len(vertexStatusRefreshInFlight)
			vertexStatusRefreshMu.Unlock()
			if n == 0 {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		vertexStatusHTTPClient = oldClient
		vertexStatusCacheMu.Lock()
		vertexStatusCached = oldCached
		vertexStatusCacheExpires = oldExp
		vertexStatusStaleUntil = oldStale
		vertexStatusCacheMu.Unlock()
	})
}

func TestVertexStatusStaleWhileRevalidateNonBlocking(t *testing.T) {
	saveAndRestoreVertexStatusGlobals(t)
	t.Setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "test-access-token")

	const invokeURL = "https://777.us-central1-888.prediction.vertexai.goog/v1/projects/888/locations/us-central1/endpoints/777/invoke/v1/chat/completions"

	probeStarted := make(chan struct{}, 4)
	releaseProbe := make(chan struct{})
	var probeCalls atomic.Int32

	vertexStatusHTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if strings.HasSuffix(r.URL.Path, "/invoke/predict") && r.Method == http.MethodPost {
				probeCalls.Add(1)
				probeStarted <- struct{}{}
				<-releaseProbe
				return jsonHTTPResponse(http.StatusServiceUnavailable, `{"status":"down"}`), nil
			}
			if strings.Contains(r.URL.Host, "aiplatform.googleapis.com") && strings.Contains(r.URL.Path, "/endpoints/777") {
				return jsonHTTPResponse(http.StatusOK, `{"displayName":"ep-777","deployedModels":[]}`), nil
			}
			return jsonHTTPResponse(http.StatusOK, `{"operations":[]}`), nil
		}),
	}

	vertexStatusCacheMu.Lock()
	vertexStatusCached = vertexEndpointLiveStatus{
		EndpointID:   "777",
		State:        "deployed",
		ReplicaCount: 1,
	}
	vertexStatusCacheExpires = time.Now().Add(-2 * time.Second) // fresh TTL expired
	vertexStatusStaleUntil = time.Now().Add(60 * time.Second)   // within stale grace window
	vertexStatusCacheMu.Unlock()

	start := time.Now()
	var wg sync.WaitGroup
	results := make([]vertexEndpointLiveStatus, 6)
	for i := range results {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = inspectVertexEndpointState(context.Background(), invokeURL)
		}(i)
	}
	wg.Wait()
	elapsed := time.Since(start)

	if elapsed > 100*time.Millisecond {
		t.Fatalf("stale-while-revalidate blocked live callers for %v", elapsed)
	}
	for i, got := range results {
		if got.State != "deployed" || !got.CacheHit || got.ProbeSource != "stale_cache" {
			t.Errorf("caller %d got state=%q cacheHit=%v source=%q, want deployed/true/stale_cache", i, got.State, got.CacheHit, got.ProbeSource)
		}
	}

	select {
	case <-probeStarted:
	case <-time.After(time.Second):
		t.Fatal("background refresh did not start")
	}
	if probeCalls.Load() != 1 {
		t.Errorf("background refresh started %d probes for 6 concurrent callers, want 1", probeCalls.Load())
	}

	close(releaseProbe)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		vertexStatusRefreshMu.Lock()
		busy := len(vertexStatusRefreshInFlight) > 0
		vertexStatusRefreshMu.Unlock()
		if !busy {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	after := inspectVertexEndpointState(context.Background(), invokeURL)
	if after.State != "quiesced" {
		t.Errorf("after background refresh completed, state = %q, want quiesced", after.State)
	}
}

func TestVertexStatusColdCacheSingleflightDeduplication(t *testing.T) {
	saveAndRestoreVertexStatusGlobals(t)
	t.Setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "test-access-token")

	const invokeURL = "https://555.us-central1-888.prediction.vertexai.goog/v1/projects/888/locations/us-central1/endpoints/555/invoke/v1/chat/completions"

	invalidateVertexStatusCache()
	vertexStatusCacheMu.Lock()
	vertexStatusCached = vertexEndpointLiveStatus{}
	vertexStatusCacheMu.Unlock()

	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	var healthHits atomic.Int32

	vertexStatusHTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			healthHits.Add(1)
			select {
			case entered <- struct{}{}:
			default:
			}
			<-release
			return jsonHTTPResponse(http.StatusOK, `{"status":"ok","vllm_ready":true}`), nil
		}),
	}

	var wg sync.WaitGroup
	statuses := make([]vertexEndpointLiveStatus, 8)
	for i := range statuses {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			statuses[idx] = inspectVertexEndpointState(context.Background(), invokeURL)
		}(i)
	}

	<-entered
	time.Sleep(15 * time.Millisecond) // allow remaining goroutines to join singleflight
	close(release)
	wg.Wait()

	if healthHits.Load() != 1 {
		t.Errorf("cold cache stampede made %d health probes, want 1", healthHits.Load())
	}
	for i, st := range statuses {
		if st.State != "deployed" {
			t.Errorf("caller %d got state %q, want deployed", i, st.State)
		}
	}
}

func TestVertexStatusRawPredictFastHealthProbeAndWarmupFallback(t *testing.T) {
	saveAndRestoreVertexStatusGlobals(t)
	t.Setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "test-access-token")
	t.Setenv("DGEM_SERVER_KEY", "active-srv-key,previous-srv-key")

	const pscURL = "http://999-us-central1-888.prediction.p.vertexai.goog/v1/projects/888/locations/us-central1/endpoints/999:rawPredict"

	t.Run("fast_data_plane_ready", func(t *testing.T) {
		invalidateVertexStatusCache()
		vertexStatusCacheMu.Lock()
		vertexStatusCached = vertexEndpointLiveStatus{}
		vertexStatusCacheMu.Unlock()

		var controlPlaneHits atomic.Int32
		var seenServerKey, seenBody string
		vertexStatusHTTPClient = &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if strings.HasSuffix(r.URL.Path, ":rawPredict") {
					seenServerKey = r.Header.Get(client.ServerKeyHeader)
					b, _ := io.ReadAll(r.Body)
					seenBody = string(b)
					return jsonHTTPResponse(http.StatusOK, `{"status":"ok","vllm_ready":true}`), nil
				}
				controlPlaneHits.Add(1)
				return jsonHTTPResponse(http.StatusInternalServerError, `{}`), nil
			}),
		}

		st := inspectVertexEndpointState(context.Background(), pscURL)
		if st.State != "deployed" || st.ProbeSource != "rawpredict_health" {
			t.Fatalf("got state=%q source=%q, want deployed/rawpredict_health", st.State, st.ProbeSource)
		}
		if seenServerKey != "active-srv-key" {
			t.Errorf("X-DGem-Key = %q, want active-srv-key", seenServerKey)
		}
		if seenBody != `{"health":true}` {
			t.Errorf(":rawPredict probe body = %q, want {\"health\":true}", seenBody)
		}
		if controlPlaneHits.Load() != 0 {
			t.Errorf("control plane hit %d times when :rawPredict health succeeded, want 0", controlPlaneHits.Load())
		}
	})

	t.Run("warmup_vllm_not_ready_falls_back_to_control_plane", func(t *testing.T) {
		invalidateVertexStatusCache()
		vertexStatusCacheMu.Lock()
		vertexStatusCached = vertexEndpointLiveStatus{}
		vertexStatusCacheMu.Unlock()

		vertexStatusHTTPClient = &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if strings.HasSuffix(r.URL.Path, ":rawPredict") {
					return jsonHTTPResponse(http.StatusOK, `{"status":"ok","vllm_ready":false,"phase":"loading_vllm_siglip"}`), nil
				}
				if strings.Contains(r.URL.Host, "aiplatform.googleapis.com") && strings.Contains(r.URL.Path, "/endpoints/999") {
					return jsonHTTPResponse(http.StatusOK, `{
						"displayName":"ep-999",
						"deployedModels":[{"id":"dm1","dedicatedResources":{"machineSpec":{"machineType":"g4-standard-48","acceleratorType":"NVIDIA_RTX_PRO_6000"}},"status":{"availableReplicaCount":0}}]
					}`), nil
				}
				return jsonHTTPResponse(http.StatusOK, `{"operations":[]}`), nil
			}),
		}

		st := inspectVertexEndpointState(context.Background(), pscURL)
		if st.State != "deploying" || st.ProbeSource != "control_plane" {
			t.Fatalf("got state=%q source=%q, want deploying/control_plane", st.State, st.ProbeSource)
		}
	})
}
