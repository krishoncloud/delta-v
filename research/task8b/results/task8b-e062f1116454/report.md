# Delta-V Task 8b verdict

Locked rules: `e062f11164540ed06dfe952ea292f808fe84ee908ace4ba696cf6c08c619547b`. Evidence status: retrospective development evidence: Task 8 test labels were already opened.

* Objective 1, cheap calibration (CV+ joint 90% within 85 to 95% at 16 and 24 runs, at most 12 extra labels): **FAIL**
* Objective 2, FNO RMSE at most 0.8 of POD-GP and refitted GP at 8 and 16 runs: **PASS**
* Objective 3, FNO-LHS saves at least 8 runs against the best baseline at 2 of 3 tolerances: **NO SAVING**

**Task 9: NO-GO until the failed objective is addressed.**

Pitch should not claim run savings or a baseline advantage beyond what the passing objectives show.

Scope: one generated 64x64 periodic flow family, three repetitions, three-member ensembles, Task 8 acquisition paths reused. Not customer or OpenFOAM validation.