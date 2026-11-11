# Evaluating the Impact of Simplifying E-Service Steps Using A/B Testing

**Course:** A/B Testing and Causal Inference (التجارب واختبارات A/B والاستدلال السببي)  
**Training organization:** SDAIA Academy — https://github.com/SDAIAAcademy  
**Team:** Nada Ali Alghamdi

## 1. Problem, objective, and decision

A hypothetical e-government service requires five steps to submit a request. The proposed design uses three steps. The decision is whether to deploy the simplified design, based on completion rate and operational safety guardrails. **This is a synthetic educational case, not an actual government service or a real-world trial.**

## 2. Research question and hypothesis

**Question:** Does reducing the service workflow from five steps to three increase the probability that an eligible user completes a request?

**Treatment:** Version B (three steps); **control:** Version A (five steps).

**H0:** pB = pA. **H1:** pB != pA (two-sided confirmatory test, with launch requiring positive effect). Fewer steps may reduce friction and abandonment.

## 3. Experimental design

- Population: hypothetical eligible users starting the service.
- Randomization unit: **unique user**, not session, to prevent cross-arm exposure.
- Allocation: 50% A / 50% B, randomized by seeded pseudo-random shuffle.
- Observation: one independent record per user; no interference or crossover assumed.
- Exposure: concurrent arms over a planned ~duration calculated below, with consistent eligibility and tracking.
- Simulation: users are assigned randomly; completion is generated with probabilities 0.60 for A and 0.65 for B, error probability 0.025 / 0.027, and completion-time distributions centered at 190 / 155 seconds. These are **assumptions**, not observed facts. The 5-percentage-point simulated lift differs intentionally from the 4-point minimum detectable effect used for planning.

## 4. Metrics and decision rules (pre-specified)

- **Primary:** proportion of randomized eligible users who complete the service (intention-to-treat; one record per user).
- **Guardrail 1:** error incidence; the upper 95% confidence limit for (B-A) must be <= +1 percentage point.
- **Guardrail 2:** mean service duration; the upper approximate 95% confidence limit for (B-A) must be <= +10 seconds.
- **Decision:** launch only when data quality and sample-size checks pass, primary effect is positive and two-sided p < 0.05, and both guardrails pass. Otherwise do not launch / inconclusive. These are illustrative operational thresholds.

## 5. Sample-size planning

Assume baseline pA = 0.60, minimum detectable absolute effect = 0.04 (pB = 0.64), two-sided alpha = 0.05, power = 0.80, 1:1 allocation, and 1,600 eligible users per day. `sample_size_per_arm` uses the conventional normal approximation for two independent proportions:

`n = [z_(1-alpha/2)*sqrt(2*pbar*(1-pbar)) + z_power*sqrt(pA*(1-pA)+pB*(1-pB))]^2 / (pB-pA)^2`

Duration = ceil(2*n / 1600). This is a planning estimate, not a measured traffic rate. For real deployment, account for weekly traffic variation, exclusions, attrition, and operational ramp-up.

## 6. Data quality

The script checks missing data, duplicate user IDs, valid binary outcomes, positive durations, both groups, sample sufficiency, and **Sample Ratio Mismatch** (chi-square goodness-of-fit against 50/50; flag if p < 0.001). A passing SRM check does not guarantee perfect randomization or event instrumentation.

## 7. Statistical analysis

- Two-sided pooled two-proportion z-test for the primary outcome.
- Absolute risk difference (B-A), relative lift, and 95% Wald confidence interval for the difference in proportions.
- Error-rate difference and its 95% confidence interval.
- Welch t-test for duration; large-sample approximate upper 95% confidence bound for the difference in means.
- The primary outcome is the sole confirmatory success metric; guardrails are pre-specified safety gates, not additional discovery tests. No multiple-testing correction is needed for a single primary hypothesis.

The simple Wald interval is adequate as a teaching approximation with these large samples and non-extreme probabilities; a production study may use a score-based interval and more robust duration analysis.

## 8. Run instructions

Python 3.10+ recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python main.py
```

No local machine-specific paths, private services, API keys, or notebook execution are required. The script **regenerates** the seeded CSV each run.

## 9. Repository contents

```text
eservice_ab_project/
├── README.md
├── main.py
├── requirements.txt
├── data/
│   └── dataset.csv
└── outputs/
    ├── charts/
    │   ├── completion_rates.png
    │   └── effect_ci.png
    └── results/
        ├── summary.json
        └── report.txt
```

## 10. Results and recommendation

See `outputs/results/report.txt` for observed simulated group rates, difference, 95% confidence interval, p-value, SRM, guardrail checks, and the pre-specified launch decision. See `outputs/results/summary.json` for machine-readable values and `outputs/charts/` for plots.

**Important:** Any recommendation is valid **only within this synthetic simulation**. Before an actual rollout, repeat the experiment on real, consented/authorized data with reliable instrumentation and monitor downstream effects.

## 11. Limitations and causal assumptions

Synthetic outcomes are generated using known group-specific distributions; the simulation demonstrates methodology, not proof of real service improvement. The causal interpretation of an actual randomized test would require valid random assignment, consistent treatment, no interference between users, complete outcome tracking, no differential attrition, and stable measurement. The model does not capture clustering, repeated visits, time trends, accessibility differences, or unmeasured operational impacts. The estimated experiment duration assumes constant daily traffic.

## 12. GitHub submission

Create a **public** repository, upload the full contents of this folder, add participant names above, and paste the repository URL into the course evaluation form before the stated deadline. Do not upload any real personal or confidential service data.
