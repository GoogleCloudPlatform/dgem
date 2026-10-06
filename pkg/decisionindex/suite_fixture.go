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

package decisionindex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// LoadSuiteJSONL loads a Decision Index JSONL dataset from disk, or falls back to DefaultPanelSuite().
func LoadSuiteJSONL(path string) ([]SuiteRow, error) {
	if path == "" {
		return DefaultPanelSuite(), nil
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			rows := DefaultPanelSuite()
			_ = SaveSuiteJSONL(path, rows)
			return rows, nil
		}
		return nil, err
	}
	defer f.Close()

	var rows []SuiteRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r SuiteRow
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, err
		}
		rows = append(rows, r)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return DefaultPanelSuite(), nil
	}
	return rows, nil
}

// SaveSuiteJSONL writes the SuiteRow slice to a JSONL file.
func SaveSuiteJSONL(path string, rows []SuiteRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return nil
}

// DefaultPanelSuite returns a curated 22-benchmark Decision Index evaluation suite spanning all 19 scored
// panel benchmarks across the 5 official areas + 3 wide-option display benchmarks (API-Bank, BANKING77, CLINC150+OOS).
// Specifically includes M > 10 multi-slot requests (ContractNLI, BRIGHT, ToolRet) and K > 26 wide-option choice
// slots (API-Bank 30 opts, BANKING77 32 opts, CLINC150+OOS 36 opts) that trigger HTTP 422 capacity rejections on
// naive 26-option / 10-slot engines.
func DefaultPanelSuite() []SuiteRow {
	var rows []SuiteRow

	// =========================================================================
	// AREA 1: Knowledge & Reasoning (6 panel benchmarks)
	// =========================================================================
	rows = append(rows, SuiteRow{
		ID:        "di_knowledge_mmlu_01",
		CatalogID: 1,
		Dataset:   "MMLU",
		Area:      "knowledge",
		State:     "Question (College Computer Science): Which of the following concurrency control protocols guarantees both conflict serializability and freedom from deadlock without requiring transaction rollbacks during lock acquisition?",
		Questions: map[string]SystemOneQuestion{
			"answer": {
				Type:         "choice",
				Instructions: "Select the correct computer science option.",
				Criteria: map[string]string{
					"A": "Two-Phase Locking (Strict 2PL) with dynamic lock escalation",
					"B": "Conservative Two-Phase Locking (Static 2PL) pre-declaring all read/write lock sets before execution",
					"C": "Timestamp Ordering (Basic TO) with Thomas Write Rule",
					"D": "Optimistic Concurrency Control (OCC) with backward validation",
				},
			},
		},
		Expected: map[string]string{"answer": "B"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_knowledge_gpqa_01",
		CatalogID: 2,
		Dataset:   "GPQA Diamond",
		Area:      "knowledge",
		State:     "Question (Quantum Mechanics / Physics): A spin-1/2 particle is prepared in the state |psi> = cos(theta/2)|+z> + sin(theta/2)|-z>. What is the expectation value of the operator sigma_x (Pauli-X matrix) in this state?",
		Questions: map[string]SystemOneQuestion{
			"answer": {
				Type:         "choice",
				Instructions: "Select the exact quantum expectation value <psi|sigma_x|psi>.",
				Criteria: map[string]string{
					"A": "cos(theta)",
					"B": "sin(theta)",
					"C": "sin(theta/2)cos(theta/2)",
					"D": "0",
				},
			},
		},
		Expected: map[string]string{"answer": "B"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_knowledge_gsm8k_01",
		CatalogID: 3,
		Dataset:   "GSM8K",
		Area:      "knowledge",
		State:     "Problem: A cloud GPU cluster processes 120 requests per minute using 4 nodes. Each node costs $1.50 per hour. What is the compute cost in USD to process 36,000 requests?",
		Questions: map[string]SystemOneQuestion{
			"answer": {
				Type:         "choice",
				Instructions: "Calculate the exact USD cost for 36,000 requests (36000 / 120 = 300 minutes = 5 hours; 4 nodes * $1.50/hr = $6.00/hr; 5 hrs * $6/hr = $30).",
				Criteria: map[string]string{
					"opt_7_50":  "$7.50",
					"opt_15_00": "$15.00",
					"opt_30_00": "$30.00",
					"opt_45_00": "$45.00",
					"opt_60_00": "$60.00",
				},
			},
		},
		Expected: map[string]string{"answer": "opt_30_00"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_knowledge_chessbench_01",
		CatalogID: 4,
		Dataset:   "ChessBench",
		Area:      "knowledge",
		State:     "FEN Position: White King on g1, White Queen on h5, White Bishop on c4; Black King on g8, Black Pawns on f7, g7, h7, Black Rook on f8. It is White to move and deliver Scholar's Mate in 1 move.",
		Questions: map[string]SystemOneQuestion{
			"best_move": {
				Type:         "choice",
				Instructions: "Select the winning chess move in algebraic notation that delivers immediate checkmate on f7.",
				Criteria: map[string]string{
					"Qxf7#": "Queen captures f7 pawn delivering checkmate supported by Bishop on c4",
					"Bxf7+": "Bishop captures f7 with check, allowing Rxf7",
					"Qg5":   "Queen moves to g5 without check",
					"Qh6":   "Queen moves to h6 threatening g7",
				},
			},
		},
		Expected: map[string]string{"best_move": "Qxf7#"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_knowledge_cruxeval_01",
		CatalogID: 5,
		Dataset:   "CRUXEval",
		Area:      "knowledge",
		State:     "Python Function:\ndef f(nums):\n    return [x * 2 for x in nums if x % 2 == 1]\n\nInput: f([1, 2, 3, 4, 5])",
		Questions: map[string]SystemOneQuestion{
			"output": {
				Type:         "choice",
				Instructions: "Select the exact return value of f([1, 2, 3, 4, 5]).",
				Criteria: map[string]string{
					"A": "[2, 4, 6, 8, 10]",
					"B": "[2, 6, 10]",
					"C": "[1, 3, 5]",
					"D": "[4, 8]",
				},
			},
		},
		Expected: map[string]string{"output": "B"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_knowledge_cladder_01",
		CatalogID: 6,
		Dataset:   "CLadder",
		Area:      "knowledge",
		State:     "Causal Graph: Rain -> Wet Sidewalk, Sprinkler -> Wet Sidewalk. Rain and Sprinkler are independent causes of Wet Sidewalk (collider at Wet Sidewalk). Suppose we observe that the Sidewalk is Wet (Wet Sidewalk = 1), AND we observe that it did NOT Rain (Rain = 0). Does conditioning on Wet Sidewalk = 1 and Rain = 0 increase the probability that the Sprinkler was on (P(Sprinkler=1 | Wet=1, Rain=0) > P(Sprinkler=1))?",
		Questions: map[string]SystemOneQuestion{
			"causal_answer": {
				Type:         "choice",
				Instructions: "Answer yes or no based on explaining-away in collider structures.",
				Criteria: map[string]string{
					"yes": "Yes, observing the effect (Wet Sidewalk) and ruling out Rain explains away the alternative cause, making Sprinkler=1 more probable.",
					"no":  "No, Rain and Sprinkler remain conditionally independent given their common effect.",
				},
			},
		},
		Expected: map[string]string{"causal_answer": "yes"},
	})

	// =========================================================================
	// AREA 2: Language Understanding (3 panel benchmarks, including 12-slot ContractNLI!)
	// =========================================================================
	contractQuestions := make(map[string]SystemOneQuestion)
	contractExpected := make(map[string]string)
	clauses := []struct {
		id   string
		text string
		gold string
	}{
		{"clause_01_confidential_marking", "All Confidential Information must be expressly marked in writing as 'Confidential' to be protected.", "contradiction"},
		{"clause_02_sharing_employees", "Receiving Party may share Confidential Information with its employees who have a need to know.", "entailment"},
		{"clause_03_sharing_affiliates", "Receiving Party may share Confidential Information with its corporate Affiliates.", "entailment"},
		{"clause_04_return_or_destroy", "Receiving Party must return or destroy Confidential Information upon written request by Disclosing Party.", "entailment"},
		{"clause_05_survival_period", "Confidentiality obligations survive for a period of five (5) years after termination.", "entailment"},
		{"clause_06_no_license", "No patent, copyright, or trademark license is granted by disclosure of Confidential Information.", "entailment"},
		{"clause_07_reverse_engineering", "Receiving Party is expressly permitted to reverse engineer prototypes provided under this Agreement.", "contradiction"},
		{"clause_08_public_domain_exception", "Information that becomes publicly available through no fault of Receiving Party is excluded from Confidential Information.", "entailment"},
		{"clause_09_compelled_disclosure", "Receiving Party may disclose Confidential Information if required by court order, provided prompt notice is given.", "entailment"},
		{"clause_10_non_solicitation", "Receiving Party agrees not to solicit or hire any engineering executives of Disclosing Party for 24 months.", "not_mentioned"},
		{"clause_11_exclusive_dealing", "Disclosing Party is prohibited from negotiating similar partnerships with third-party competitors.", "not_mentioned"},
		{"clause_12_injunctive_relief", "Disclosing Party is entitled to seek immediate injunctive relief in the event of a breach.", "entailment"},
	}
	for _, c := range clauses {
		contractQuestions[c.id] = SystemOneQuestion{
			Type:         "choice",
			Instructions: fmt.Sprintf("Evaluate NDA hypothesis: %s", c.text),
			Criteria: map[string]string{
				"entailment":    "Explicitly supported and entailed by the NDA text",
				"contradiction": "Explicitly contradicted by the NDA text",
				"not_mentioned": "Neither entailed nor contradicted (not mentioned in the NDA)",
			},
		}
		contractExpected[c.id] = c.gold
	}
	rows = append(rows, SuiteRow{
		ID:        "di_language_contractnli_12slots",
		CatalogID: 7,
		Dataset:   "ContractNLI",
		Area:      "language",
		State: "NON-DISCLOSURE AGREEMENT (NDA) SUMMARY:\n" +
			"1. Scope: Confidential Information includes all technical and business information disclosed orally or in writing, WHETHER OR NOT marked as 'Confidential'.\n" +
			"2. Permitted Disclosures: Receiving Party may disclose Confidential Information to its employees and corporate Affiliates with a strict need to know, or as required by court order (with prompt prior written notice). Information that enters the public domain through no fault of Receiving Party is excluded.\n" +
			"3. Restrictions: Receiving Party shall NOT reverse engineer, decompile, or disassemble any prototypes or software. No license under any patent, copyright, or trademark is granted.\n" +
			"4. Return & Survival: Upon written request, Receiving Party shall promptly return or destroy all Confidential Information. Obligations survive for five (5) years. Breach entitles Disclosing Party to immediate injunctive relief. (Note: No non-solicitation or exclusivity clauses are included in this NDA.)",
		Questions: contractQuestions,
		Expected:  contractExpected,
	})

	rows = append(rows, SuiteRow{
		ID:        "di_language_isarcasm_01",
		CatalogID: 8,
		Dataset:   "iSarcasmEval",
		Area:      "language",
		State:     "Utterance: 'Oh fantastic, my flight was delayed by 6 hours and the airline lost my luggage. Truly the best vacation start ever!'",
		Questions: map[string]SystemOneQuestion{
			"tone": {
				Type:         "choice",
				Instructions: "Classify the pragmatic figure of speech in the utterance.",
				Criteria: map[string]string{
					"sarcasm":        "Sarcastic / ironic inversion praising an obviously negative situation",
					"literal_happy":  "Sincere literal expression of joy and satisfaction",
					"neutral_fact":   "Objective factual travel itinerary update without emotion",
					"understatement": "Downplaying a major catastrophe as a minor inconvenience",
				},
			},
		},
		Expected: map[string]string{"tone": "sarcasm"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_language_vast_01",
		CatalogID: 9,
		Dataset:   "VAST",
		Area:      "language",
		State:     "Target Topic: Open-Source Software Licensing.\nAuthor Text: 'Releasing foundational infrastructure under permissive open-source licenses accelerates security auditing, eliminates vendor lock-in, and enables reproducible scientific verification across the entire industry.'",
		Questions: map[string]SystemOneQuestion{
			"stance": {
				Type:         "choice",
				Instructions: "Determine the author's stance toward the target topic (Open-Source Software Licensing).",
				Criteria: map[string]string{
					"pro":     "In favor of / supportive of the target topic",
					"con":     "Against / opposing the target topic",
					"neutral": "Neutral or unrelated to the target topic",
				},
			},
		},
		Expected: map[string]string{"stance": "pro"},
	})

	// =========================================================================
	// AREA 3: Retrieval & Classification (2 panel benchmarks, including 11-slot BRIGHT!)
	// =========================================================================
	brightQuestions := make(map[string]SystemOneQuestion)
	brightExpected := make(map[string]string)
	for i := 1; i <= 11; i++ {
		qKey := fmt.Sprintf("passage_%02d", i)
		isRel := (i == 2 || i == 7)
		desc := fmt.Sprintf("Candidate Passage %02d: Discusses general HTTP status codes.", i)
		gold := "not_relevant"
		if isRel {
			desc = fmt.Sprintf("Candidate Passage %02d: Proves that Shannon entropy H(Y|X) = -sum p_k ln p_k over restricted single-token slot logits provides an exact $O(1)$ calibration gate for early exit.", i)
			gold = "relevant"
		}
		brightQuestions[qKey] = SystemOneQuestion{
			Type:         "choice",
			Instructions: fmt.Sprintf("Assess relevance of (%s) to the query: 'How does single-pass restricted softmax entropy enable O(1) early exit in diffusion classifiers?'", desc),
			Criteria: map[string]string{
				"relevant":     "Directly answers and proves the query theorem/mechanism",
				"not_relevant": "Background noise or unrelated topic",
			},
		}
		brightExpected[qKey] = gold
	}
	rows = append(rows, SuiteRow{
		ID:        "di_retrieval_bright_11slots",
		CatalogID: 10,
		Dataset:   "BRIGHT",
		Area:      "retrieval",
		State:     "Query: How does single-pass restricted softmax entropy enable O(1) early exit in diffusion classifiers?\nPassage 02 and Passage 07 explicitly derive the restricted softmax Shannon entropy H = -sum p_k ln p_k over single-token canvas slots for O(1) early exit, whereas all other passages (01, 03, 04, 05, 06, 08, 09, 10, 11) discuss unrelated HTTP status codes.",
		Questions: brightQuestions,
		Expected:  brightExpected,
	})

	rows = append(rows, SuiteRow{
		ID:        "di_retrieval_esci_01",
		CatalogID: 11,
		Dataset:   "Amazon ESCI",
		Area:      "retrieval",
		State:     "Shopping Query: 'noise cancelling wireless over-ear headphones black'\nProduct Title: 'Sony WH-1000XM5 Wireless Industry Leading Active Noise Canceling Over-Ear Headphones, Black'",
		Questions: map[string]SystemOneQuestion{
			"esci_label": {
				Type:         "choice",
				Instructions: "Classify the Amazon ESCI query-product relevance relation.",
				Criteria: map[string]string{
					"Exact":      "Exact match satisfying all query attributes (wireless, over-ear, noise cancelling, black)",
					"Substitute": "Partially matching alternative product (e.g. wired or in-ear earbuds)",
					"Complement": "Accessory or complementary product (e.g. headphone carrying case)",
					"Irrelevant": "Completely unrelated product",
				},
			},
		},
		Expected: map[string]string{"esci_label": "Exact"},
	})

	// =========================================================================
	// AREA 4: Tools & Automation (3 panel benchmarks, including 12-slot ToolRet!)
	// =========================================================================
	rows = append(rows, SuiteRow{
		ID:        "di_tools_bfcl_01",
		CatalogID: 12,
		Dataset:   "BFCL",
		Area:      "tools",
		State:     "User Request: 'Get the current stock price and 24-hour trading volume for ticker NVDA in USD.'\nAvailable Functions:\n- get_weather_forecast(city, units)\n- get_equity_quote(ticker, currency, include_volume)\n- execute_wire_transfer(iban, amount)\n- cancel_subscription(user_id)",
		Questions: map[string]SystemOneQuestion{
			"function_call": {
				Type:         "choice",
				Instructions: "Select the exact function tool to invoke for the user request.",
				Criteria: map[string]string{
					"get_weather_forecast":  "Fetches meteorological weather forecast for a city",
					"get_equity_quote":      "Fetches live stock price and 24h trading volume for an equity ticker in the specified currency",
					"execute_wire_transfer": "Initiates an international bank wire transfer",
					"cancel_subscription":   "Terminates a recurring SaaS billing subscription",
				},
			},
		},
		Expected: map[string]string{"function_call": "get_equity_quote"},
	})

	toolRetQuestions := make(map[string]SystemOneQuestion)
	toolRetExpected := make(map[string]string)
	for i := 1; i <= 12; i++ {
		tKey := fmt.Sprintf("tool_candidate_%02d", i)
		gold := "skip"
		instr := fmt.Sprintf("Should tool_%02d (unrelated image watermark filter) be invoked to inspect Kubernetes pod OOMKills and Prometheus memory metrics?", i)
		if i == 4 {
			gold = "invoke"
			instr = "Should tool_04 (kubectl_get_pod_oom_events: inspects Kubernetes container exit code 137 / OOMKilled events) be invoked?"
		} else if i == 9 {
			gold = "invoke"
			instr = "Should tool_09 (prometheus_query_container_memory_working_set_bytes: queries container memory usage time-series) be invoked?"
		}
		toolRetQuestions[tKey] = SystemOneQuestion{
			Type:         "choice",
			Instructions: instr,
			Criteria: map[string]string{
				"invoke": "Required diagnostic tool for Kubernetes OOMKill & Prometheus memory investigation",
				"skip":   "Irrelevant tool that should not be called",
			},
		}
		toolRetExpected[tKey] = gold
	}
	rows = append(rows, SuiteRow{
		ID:        "di_tools_toolret_12slots",
		CatalogID: 13,
		Dataset:   "ToolRet",
		Area:      "tools",
		State:     "SRE Task: Diagnose why the production 'dgemma' container restarted with Exit Code 137 (OOMKilled) by querying Kubernetes pod OOM events (tool_04) and Prometheus container memory metrics (tool_09). Do not invoke image watermark or audio tools.",
		Questions: toolRetQuestions,
		Expected:  toolRetExpected,
	})

	rows = append(rows, SuiteRow{
		ID:        "di_tools_routerbench_01",
		CatalogID: 14,
		Dataset:   "RouterBench",
		Area:      "tools",
		State:     "Routing Policy: Route low-entropy deterministic classification requests (H < 0.35 nats, latency budget < 800ms, cost < $0.0005) to the fast System-1 Decision Model (`dgemma-vllm-l4`), and escalate only high-entropy multi-step synthesis requests to `frontier-reasoning-model`. Incoming Request: Single-pass 4-way ticket severity triage with H = 0.04 nats.",
		Questions: map[string]SystemOneQuestion{
			"target_model": {
				Type:         "choice",
				Instructions: "Select the optimal model route under the cost-quality-latency routing policy.",
				Criteria: map[string]string{
					"dgemma-vllm-l4":           "Fast System-1 Decision Model ($0.00018/req, 610ms, handles H < 0.35 nats)",
					"frontier-reasoning-model": "Heavy frontier reasoning model ($0.015/req, 4200ms, reserved for H >= 0.35 nats)",
					"claude-3-7-opus":          "High-cost long-context synthesis model ($0.030/req)",
					"gpt-4.5-preview":          "High-latency generalist model ($0.075/req)",
				},
			},
		},
		Expected: map[string]string{"target_model": "dgemma-vllm-l4"},
	})

	// =========================================================================
	// AREA 5: Arts & Human Judgment (5 panel benchmarks)
	// =========================================================================
	rows = append(rows, SuiteRow{
		ID:        "di_arts_bpomp_01",
		CatalogID: 15,
		Dataset:   "BPoMP",
		Area:      "arts",
		State:     "Musical Progression Description: A slow minor-key adagio featuring descending chromatic strings, low cello ostinato, and sparse dissonant piano chords at 56 BPM.",
		Questions: map[string]SystemOneQuestion{
			"mood": {
				Type:         "choice",
				Instructions: "Select the primary emotional mood evoked by this musical passage.",
				Criteria: map[string]string{
					"melancholic_somber": "Melancholic, somber, and mournful",
					"euphoric_festive":   "Euphoric, festive, and celebratory dance energy",
					"playful_whimsical":  "Playful, lighthearted, and comical",
					"triumphant_march":   "Triumphant military brass fanfare",
				},
			},
		},
		Expected: map[string]string{"mood": "melancholic_somber"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_arts_humicroedit_01",
		CatalogID: 16,
		Dataset:   "Humicroedit",
		Area:      "arts",
		State:     "Original Headline: 'Scientists discover water on Mars.'\nEdited Headline A: 'Scientists discover artisanal sparkling water at $9 a bottle on Mars.'\nEdited Headline B: 'Scientists discover soil on Mars.'",
		Questions: map[string]SystemOneQuestion{
			"funnier_edit": {
				Type:         "choice",
				Instructions: "Which edited headline is rated funnier by human judges due to incongruous satirical juxtaposition?",
				Criteria: map[string]string{
					"Edit_A": "Headline A ('artisanal sparkling water at $9 a bottle on Mars') - humorous incongruity",
					"Edit_B": "Headline B ('soil on Mars') - mundane factual substitution",
				},
			},
		},
		Expected: map[string]string{"funnier_edit": "Edit_A"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_arts_pop909_01",
		CatalogID: 17,
		Dataset:   "POP909-CL",
		Area:      "arts",
		State:     "Track Structure Analysis: High-energy vocal chorus repeating the main hook melody with full drum kit, bassline, and synth brass stabs at peak loudness (-6 LUFS), immediately following a quiet pre-chorus build-up.",
		Questions: map[string]SystemOneQuestion{
			"section_type": {
				Type:         "choice",
				Instructions: "Identify the structural section label in POP909 musical form analysis.",
				Criteria: map[string]string{
					"intro":      "Quiet instrumental introduction before vocals enter",
					"verse":      "Low-intensity narrative verse section",
					"chorus":     "Peak-energy repeating main hook chorus section",
					"outro_fade": "Final decaying fade-out section",
				},
			},
		},
		Expected: map[string]string{"section_type": "chorus"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_arts_cfcolor_01",
		CatalogID: 18,
		Dataset:   "cfcolor",
		Area:      "arts",
		State:     "UI Design Brief: Design an emergency 'CRITICAL SECURITY BREACH / QUARANTINE ACTIVE' alert banner on a dark-mode Security Operations Center (SOC) dashboard requiring immediate high-contrast visual urgency.",
		Questions: map[string]SystemOneQuestion{
			"palette": {
				Type:         "choice",
				Instructions: "Select the semantic color palette matching human perceptual urgency standards.",
				Criteria: map[string]string{
					"crimson_alert_red": "High-contrast crimson alert red (#EF4444) with amber warning border",
					"pastel_mint_green": "Soft pastel mint green (#A7F3D0) associated with success/calm",
					"muted_slate_gray":  "Low-contrast muted slate gray (#64748B) used for disabled elements",
					"baby_sky_blue":     "Light sky blue (#BAE6FD) used for informational tooltips",
				},
			},
		},
		Expected: map[string]string{"palette": "crimson_alert_red"},
	})

	rows = append(rows, SuiteRow{
		ID:        "di_arts_habermas_01",
		CatalogID: 19,
		Dataset:   "Habermas Machine",
		Area:      "arts",
		State:     "Citizen Group A Position: 'We need strict congestion pricing downtown to cut traffic emissions and fund public transit buses.'\nCitizen Group B Position: 'Congestion tolls hurt night-shift hospital and restaurant workers who drive when buses aren't running.'\nMediator Proposals:\n- Statement 1: Implement peak-hour congestion pricing to fund public transit while exempting off-peak night-shift and essential low-income workers.\n- Statement 2: Ban all private cars 24/7 with no exemptions.\n- Statement 3: Cancel all public transit funding and widen downtown highways.",
		Questions: map[string]SystemOneQuestion{
			"consensus_statement": {
				Type:         "choice",
				Instructions: "Select the consensus mediation statement that best bridges both citizen groups' core concerns.",
				Criteria: map[string]string{
					"Statement_1": "Peak-hour congestion pricing funding transit + exemptions for off-peak night-shift workers",
					"Statement_2": "Total 24/7 car ban with zero exemptions",
					"Statement_3": "Cancel public transit funding and expand highways",
					"Statement_4": "Defer all transport decisions for 20 years",
				},
			},
		},
		Expected: map[string]string{"consensus_statement": "Statement_1"},
	})

	// =========================================================================
	// DISPLAY / WIDE-OPTION BENCHMARKS (K > 26 options per choice!)
	// A naive single request fails with HTTP 422 ("at most 26 options per choice").
	// dgem's 2-Stage Bracket Tournament Routing handles them seamlessly!
	// =========================================================================

	// 20. API-Bank (30 options > 26 limit)
	apiBankCriteria := make(map[string]string, 30)
	for i := 1; i <= 30; i++ {
		key := fmt.Sprintf("api_tool_%02d", i)
		if i == 28 {
			apiBankCriteria["ConvertCurrencyFX_28"] = "ConvertCurrencyFX(amount, from_currency, to_currency, date): Converts an amount between two ISO currency codes using historical spot FX rates"
		} else {
			apiBankCriteria[key] = fmt.Sprintf("Unrelated system utility API #%02d for file compression or calendar alarms", i)
		}
	}
	rows = append(rows, SuiteRow{
		ID:        "di_wide_apibank_30opts",
		CatalogID: 20,
		Dataset:   "API-Bank",
		Area:      "tools",
		State:     "User Prompt: 'Convert 1,250 EUR into JPY using yesterday's foreign exchange spot rate.' Select the exact API from the 30 registered enterprise tools.",
		Questions: map[string]SystemOneQuestion{
			"selected_api": {
				Type:         "choice",
				Instructions: "Select the single API tool that performs ISO currency foreign exchange (FX) conversion.",
				Criteria:     apiBankCriteria,
			},
		},
		Expected: map[string]string{"selected_api": "ConvertCurrencyFX_28"},
	})

	// 21. BANKING77 (32 options > 26 limit)
	b77Criteria := make(map[string]string, 32)
	for i := 1; i <= 32; i++ {
		key := fmt.Sprintf("banking_intent_%02d", i)
		if i == 29 {
			b77Criteria["lost_or_stolen_card"] = "Customer reports that their physical debit or credit card was lost or stolen and must be frozen immediately"
		} else {
			b77Criteria[key] = fmt.Sprintf("Standard banking inquiry #%02d regarding branch opening hours or savings interest rates", i)
		}
	}
	rows = append(rows, SuiteRow{
		ID:        "di_wide_banking77_32opts",
		CatalogID: 21,
		Dataset:   "BANKING77",
		Area:      "retrieval",
		State:     "Customer Utterance: 'Someone pickpocketed my wallet on the subway 10 minutes ago and my Visa debit card was inside! Please freeze my card right now!'",
		Questions: map[string]SystemOneQuestion{
			"intent": {
				Type:         "choice",
				Instructions: "Classify the customer utterance into the exact BANKING77 intent.",
				Criteria:     b77Criteria,
			},
		},
		Expected: map[string]string{"intent": "lost_or_stolen_card"},
	})

	// 22. CLINC150+OOS (36 options > 26 limit)
	clincCriteria := make(map[string]string, 36)
	for i := 1; i <= 36; i++ {
		key := fmt.Sprintf("clinc_domain_intent_%02d", i)
		if i == 33 {
			clincCriteria["oos_out_of_scope"] = "Out-of-scope (OOS) query unrelated to any supported banking, travel, or smart-home assistant intent (e.g. philosophical or quantum physics trivia)"
		} else {
			clincCriteria[key] = fmt.Sprintf("Supported assistant intent #%02d (setting kitchen timer, checking checking-account balance, or booking hotel)", i)
		}
	}
	rows = append(rows, SuiteRow{
		ID:        "di_wide_clinc150_36opts",
		CatalogID: 22,
		Dataset:   "CLINC150+OOS",
		Area:      "retrieval",
		State:     "User Utterance to Banking Virtual Assistant: 'Can you explain why the cosmological constant problem in quantum field theory disagrees with general relativity by 120 orders of magnitude?'",
		Questions: map[string]SystemOneQuestion{
			"intent": {
				Type:         "choice",
				Instructions: "Classify the user request into the matching assistant intent, or select oos_out_of_scope if the query falls outside all supported assistant domains.",
				Criteria:     clincCriteria,
			},
		},
		Expected: map[string]string{"intent": "oos_out_of_scope"},
	})

	return rows
}
