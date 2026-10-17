# User-supplied Task 5 handoff — 17 October 2026

The following conversation was supplied by Krish for preservation alongside the
completed GPU run. It is historical context, not instructions to execute more work.
Specific performance predictions, proposed fixes and hosting/concurrency descriptions
below are hypotheses or claims from that conversation, not independently verified facts.
No new training, Task 6 implementation or deployment is authorized by preserving it.

Training took only 13 seconds on the GPU, so more steps are cheap. Now the per-case errors, which notebook version you ran, and where the field errors are concentrated:
Short answer: you’re not close to GP, and more steps won’t get you there. This run is a clean pass, though: READY FOR TASK 6, with all 8 gates green. You ran the exact notebook I fixed; the source hash in your ZIP matches my commit.
How far behind GP you are
The FNO is about 4x better than linear and 22x better than the mean. GP is 27x better than the FNO, which is a gap in orders of magnitude, not a few percent.
Why more steps won’t close it
- The learning curve has flattened into noise. Development loss fell 1000x by step 180. From step 180 to 300 it just bounces between 0.002 and 0.01: step 280 was 0.0020, then step 300 jumped back to 0.0071. A constant learning rate of 1e-4 with one sample per step causes this jitter, and extra steps just add more of it.
- Field errors are far too large. Velocity errors reach about 6% of the field’s peak. To match GP’s ~0.1% outcome error, field error would need to drop roughly 50 to 100x. More steps with this setup might give 2 to 3x at best.
- This test is GP’s ideal case. Kinetic energy here is exactly A² e^(-4ν) / 4: a smooth function of 2 knobs, sampled at 16 points. GP interpolates that almost perfectly. That’s why the notebook only requires beating the mean, and why GP isn’t a fair target on this fixture.
- The outcomes aren’t independent. Pressure RMS equals kinetic energy for this vortex, so the two rows in the table are one result, not two.
What the results reveal (useful for Task 6)
1. Small-amplitude cases are under-learned. Both A = 0.5 cases are off by about 10%; the rest are under 4%. The loss uses absolute error, so low-energy fields barely count. A per-sample relative loss would fix this.
2. There’s a systematic velocity bias. Predicted mean u and v are +0.007 and +0.011, when the truth is exactly 0. That inflates kinetic energy, which is why the FNO overpredicts on 7 of 9 cases.
3. There are edge artifacts, as your signed-error plot shows. Edge error is about 30% higher than in the interior. The coordinate input channels jump at the boundary, which breaks periodicity on a periodic domain. Encoding coordinates as sin/cos would fix this.
4. The optimizer setup is minimal. It uses a fixed learning rate, batch size 1 and no decay. A cosine schedule with batches of 4 would likely settle much lower.
What I’d do
- Stop tuning on this fixture. Task 5 is done. The development set already chooses the checkpoint, so every rerun aimed at the GP number tunes against it and makes the result look better than it really is.
- Optional curiosity run: 2000 steps takes about 90 seconds on your GPU, so it’s free to look. Just don’t treat it as the result. Expect roughly 2 to 3x improvement and still well behind GP.
- Where the FNO can actually win is Task 6. It needs a study where the outcome isn’t a smooth closed-form function of the knobs: more knobs, sharp regime changes, or an outcome that depends on spatial structure. The FNO also returns full fields, which GP can’t. Fixes 1 to 4 above belong in the architecture you lock before Task 6, not in tuning against this development set.
If you want, I can apply fixes 1 to 4 as a locked “Task 6 candidate” configuration in the notebook. It would keep the current setup as the Task 5 baseline so the comparison stays honest.
Here’s the mandatory objective list for Task 6. I combined the eight requirements in V2_RELEASE_PLAN.md with what your Task 5 run taught us. Each objective has a “done when” so it can be checked off.
- A1. Write and freeze a Task 6 protocol document: target problem, outcomes, tolerances, arms, budgets, seeds, metrics, decision rule and confirmatory set. Done when: it’s committed with a hash and a date before the first training run.
- A2. Predeclare the engineering tolerance for each outcome (for example, “within 5% relative error is acceptable”). Done when: it’s in the protocol, justified by what a user would actually tolerate.
- A3. Predeclare the decision rule for “pretraining helps”. For example: the pretrained model beats scratch on median error across seeds at 2 or more budgets, by more than the seed-to-seed spread. Done when: it’s written down before any results exist.
- A4. Predeclare the tuning budget for every arm: what may be tuned, on which data, and for how many trials. Done when: it’s in the protocol, and GP gets a fair amount of tuning too.
- B1. Exclude the source systems (turbulent radiative layer, Rayleigh–Bénard, shear flow), and use more than a withheld parameter value of those. Done when: the exclusion is recorded alongside the known gap in the checkpoint’s exposure records.
- B2. Realistic difficulty: 3 or more knobs, or a regime change, or outcomes with no closed form. Chosen for realism, not because GP fails on it. Done when: the selection rationale is written into the protocol.
- B3. Independent outcomes: 2 or more outcomes with real engineering meaning, for example a probe pressure difference or a mixing measure. Done when: their correlation on pilot data is reported and they aren’t identical.
- B4. A feasible query contract: what’s available at query time, consistent with Task 5’s knobs plus known initial state. Done when: the study passes validate_study through the customer-bundle path.
- B5. Data generation is affordable for nested budgets plus a fresh confirmatory set. Done when: the run count and solver cost per run are estimated and recorded.
- C1. Mean.
- C2. Linear (or log-linear where declared).
- C3. GP, fairly tuned.
- C4. Scratch FNO: same architecture and inputs, random weights.
- C5. Pretrained FNO, full fine-tuning (Task 5’s method).
- C6. Pretrained FNO, frozen backbone: only the conditioner (plus a small head, if declared) trains.
- C7. Information disclosure: the FNO uses fields while GP uses scalars. Report this, add an information-matched variant if feasible, and count the cost of exporting the fields.
Done when: all six arms run on identical splits and seeds, with any arm-specific limitation stated.
- D1. Nested training budgets (for example 4, 8, 16 and 32 settings), each a subset of the next, identical across arms. This is what shows whether the FNO needs fewer simulations.
- D2. Whole-setting splits: repeats and all trajectory windows of a setting stay in one split. Task 5’s validator already enforces this.
- D3. Three separate sets: training, validation (model selection only), and an untouched confirmatory test that’s opened once, at the end.
- D4. Train-only preprocessing: normalization and scaling fitted only on each budget’s training rows.
- D5. Count every simulation, validation runs included, so cost claims are honest.
- E1. Decide on Task 5’s four fixes (relative loss, boundary-matched coordinates, learning-rate schedule, mini-batches) using pilot or validation data only, then freeze them.
- E2. Seeds: 3 or more per arm per budget, with spread reported.
- E3. Equal compute rules across neural arms: the same steps and time cap, with checkpoint selection on validation only.
- E4. Resolution: declare how the 256×256 source maps to the target grid, and keep it identical for scratch and pretrained.
- F1. Per-outcome results: RMSE, MAE, relative error and worst case per arm, budget and seed.
- F2. Error vs number of training runs (data-efficiency curve): budget on one axis, confirmatory error on the other, all arms, with seed spread.
- F3. Runs to reach tolerance: for each arm, the smallest budget that meets the A2 tolerance. This is the product claim, if there is one.
- F4. Behavior at the edges: errors near the edge of the training range versus the interior, and which queries the out-of-range check rejected.
- F5. Costs: preprocessing, adaptation, inference and solver time for every arm, with source pretraining cost disclosed separately.
- F6. Field quality for the FNO arms: field error maps on the confirmatory set, not just scalar outcomes.
- G1. Save everything: data, schema hashes, exact splits, checkpoint identities, configurations, weights, predictions, timings and warnings. Use the Task 5 artifact pipeline.
- G2. Publish negative results with equal prominence. A GP win stays labelled a GP win.
- G3. Scope statement: one target problem supports claims for that problem class only.
- Complete: the locked comparison has run and been audited, whoever wins.
- “Pretraining helps” is supported only if: the pretrained model beats scratch under the A3 rule on the confirmatory set.
- “Better than GP” is a separate claim, made only if it happens under the same locked rules.
- Calibrated uncertainty (Task 7).
- Next-run recommendation and real run savings (Task 8).
- The OpenFOAM workflow (Task 9).
- Website integration (Tasks 10 and 12).
Add the scratch and frozen-backbone options to the notebook, then run both on the Task 5 vortex as a pilot. That gives you an early look at whether pretraining helps, tests the arm machinery, and is clearly labelled exploratory because the vortex data has already been inspected. Meanwhile, start B1 to B5: auditing candidate target problems is the slowest part.
Want me to draft the A1 protocol document as a file in the repo, or add the scratch and frozen-backbone options first?
how do you recommend to make this flow for the website btw and where will it load and run,
 like at the moment, whenever someone uses the website the model runs where ?
how many people can use the website concurrently at the moment ?
I checked the deployed code on main (Dockerfile, main.py, README.md).
The live site is a Hugging Face Space (krishmalik-delta-v.hf.space), running your Docker image:
- CPU only. The Dockerfile sets DEVICE=cpu and installs CPU-only torch. There’s no GPU anywhere in production.
- One server process. It’s a single uvicorn process, with no extra workers.
- The model loads once at startup. On boot, the server downloads the 858 MB domain_fluid.pt from your HF repo and loads it into that one process’s memory. That’s what the site’s “waking up” state is waiting for. Nothing runs in the user’s browser.
A request takes one of three paths:
- Browsing, dials and cached samples: dozens of people at once. The caps are your own rate limits:
  - Dials: 120 requests per minute per user, 1200 per minute total.
  - Field predictions: 10 per minute per user, 60 per minute total.
  - Rollouts: 3 per minute per user, 20 per minute total.
- Live FNO computation: exactly 1 at a time. INFERENCE_SLOT is a single lock. If a second person asks while one is running, they get an immediate 429 “Inference is busy; retry shortly”. They don’t wait in a queue.
I don’t know which Space hardware tier you’re on. The free CPU tier is small (a couple of vCPUs) and sleeps when idle, which explains the cold start. Check your Space settings to confirm.
- Training is heavy. Your fine-tune took 13 seconds on a MoLab GPU. On CPU, ChatGPT’s 200-step run took about 4 minutes on your Mac, and the Space’s CPU is likely slower. During that time it would hold the single inference lock, blocking every other live prediction on the site.
- Each customer’s adapted model is 286 MB (your best_model.pt), since full fine-tuning copies the whole network. Storing one per customer study gets expensive fast.
- Privacy. Task 15 makes private storage mandatory before customer data touches the live system. The current Space is public.
```
Browser (bring-your-runs page)
   │  upload study bundle
   ▼
Web API (existing HF Space, CPU)        ◄── stays light: never trains
   │ 1. validate_study()   (seconds, CPU; reuse the Task 5 code)
   │ 2. enqueue "fit" job  → returns job_id
   ▼
Job queue + status table
   ▼
GPU worker (on demand, scales to zero)
   │ 3. load base checkpoint (cached on worker)
   │ 4. fit_study()  → adapted weights + report
   │ 5. predict queries, compute checks
   ▼
Private storage (Task 15)
   adapted weights, normalization, contract, predictions, report ZIP
   ▲
Web API  ◄── browser polls job_id → shows report, predictions, download
```
Steps to get there:
1. Move the notebook logic into a module. Pull validate_study, fit_study, predict_study and package_run out of the notebook into something like deltav_task5.py. The notebook, web API and GPU worker then all run the same tested code, and the 8 tests move with it.
2. The web server only validates and queues. Uploads are checked on the spot using the Task 5 validator, which gives the user instant, specific rejection reasons. Training never runs inside the web process.
3. An on-demand GPU worker does the fitting. Use a serverless GPU service that starts per job and costs nothing when idle. Options worth comparing:
   - Hugging Face ZeroGPU, which stays in your existing HF ecosystem but needs a paid plan to host.
   - Modal.
   - RunPod Serverless.
    A 13-second fine-tune plus model loading costs very little per job. Check each service’s current pricing and quotas before choosing.
4. The browser polls job status. The page shows stages: validating, queued, training, checking, done. That matches the staged flow of the notebook.
5. Serve predictions cheaply. Once a customer’s model is adapted, new queries are single forward passes on a 64×64 grid, which may be fast enough on the Space’s CPU. Measure that rather than assume it. Keep the single-slot lock, or batch queries inside the GPU job.
6. Store results privately (Task 15), under a per-study ID, with the same hashed ZIP the notebook already produces.
Task 6’s frozen-backbone arm has big product consequences, not just scientific ones:
- With full fine-tuning, every customer study needs its own 286 MB model copy, and serving means loading a different model per customer.
- With a frozen backbone, every customer shares one base FNO, plus a tiny conditioner of a few kilobytes. Storage is close to free, many customers can share one loaded model, and fitting is faster.
If the frozen-backbone arm comes close to full fine-tuning in Task 6, it makes the website far cheaper to run. That’s another reason to run it carefully.
- Task 10: the upload and validation API, plus job endpoints.
- Task 12: wiring up the bring-your-runs page.
- Task 15: private storage, a prerequisite before any real customer data.
- Task 14: deploying with the GPU worker.
