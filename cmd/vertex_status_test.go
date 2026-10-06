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
	"testing"
)

// A blue/green deploy lists the new model first at 0% traffic. The endpoint state must follow the serving model.
func TestServingDeployedModelFollowsTraffic(t *testing.T) {
	raw := `{"displayName":"ep","trafficSplit":{"old":100,"new":0},"deployedModels":[
	  {"id":"new","displayName":"candidate","status":{}},
	  {"id":"old","displayName":"current","status":{"availableReplicaCount":1}}]}`
	var e vertexEndpointResource
	if err := json.Unmarshal([]byte(raw), &e); err != nil {
		t.Fatal(err)
	}
	if dm := e.servingDeployedModel(); dm.ID != "old" || dm.Status.AvailableReplicaCount != 1 {
		t.Fatalf("got %q (%d replicas), want the model with traffic", dm.ID, dm.Status.AvailableReplicaCount)
	}
}

func TestServingDeployedModelWithoutTrafficSplit(t *testing.T) {
	e := vertexEndpointResource{DeployedModels: []vertexDeployedModelRes{{ID: "a"}, {ID: "b"}}}
	if dm := e.servingDeployedModel(); dm.ID != "a" {
		t.Fatalf("got %q, want the first model", dm.ID)
	}
}
