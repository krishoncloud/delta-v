"""Historical research evidence shown on the website, with scope and source on every number.

Numbers are copied from the published Task 8 / 8b result tables and checked by
tests/test_flow_evidence.py against those files. They describe one generated
research family and are kept separate from any study's own measurements.
"""
TASK8 = "research/task8/results/20261019T025823Z"
TASK8B = "research/task8b/results/task8b-e062f1116454"

EVIDENCE = {
    "status": "measured",
    "kind": "historical research evidence",
    "scope": "One generated 2D periodic forced incompressible flow family (64x64, three controls, two outcomes). "
             "Not customer CFD, OpenFOAM or a validation on arbitrary fluid studies.",
    "cards": [
        {
            "id": "task8_fno_lhs_accuracy",
            "claim": "With 16 training runs chosen for even coverage, predictions of both outcomes were within tolerance on "
                     "597 of 600 held-out settings, across three independent repeats of the study.",
            "qualification": "One generated 2D periodic flow study type. Each fit also used 12 development and "
                             "39 calibration runs. Not a validation on customer CFD.",
            "value": {"hits": 597, "cases": 600, "per_repetition": {"701": [200, 200], "1409": [197, 200], "2111": [200, 200]}},
            "metric": "both |prediction - reference| <= 0.015 (absolute, nondimensional) per held-out setting",
            "protocol": "df369db23d69771ad092d5413b07afac3a670f2d217ebc9697d6880bfc256424",
            "source": f"{TASK8}/round_metrics.csv (policy fno_lhs, budget 16)",
        },
        {
            "id": "task8_fno_lhs_rmse",
            "claim": "Average error (RMSE) at 16 training runs: kinetic energy 0.00306, pressure RMS 0.00336 "
                     "(uncertainty-driven selection: 0.00781 and 0.00661).",
            "qualification": "Nondimensional outcome units; three independent repeats of one generated study type.",
            "value": {"fno_lhs": [0.00306350807659328, 0.003361375769600272],
                      "fno_active": [0.007809970683107774, 0.0066107021023829775]},
            "metric": "root mean squared outcome error on 200 held-out settings per repetition",
            "source": f"{TASK8}/round_metrics.csv",
        },
        {
            "id": "task8_coverage",
            "claim": "Nominal 90% joint intervals covered 91.5%, 94.5% and 86.0% of held-out cases (pooled 90.7%).",
            "qualification": "Research observations at one budget. Not a confidence guarantee for an individual prediction; "
                             "one repeat covered less than intended.",
            "value": {"per_repetition": [0.915, 0.945, 0.86], "pooled": 0.9066666666666666},
            "metric": "joint split-conformal coverage, 39 calibration runs",
            "source": f"{TASK8}/round_metrics.csv",
        },
        {
            "id": "task8_stopping",
            "claim": "All three even-coverage repeats were accurate enough at 16 runs, but the automatic stopping "
                     "rule would have stopped at 24, 24 and 16.",
            "qualification": "Accuracy measured afterwards is not something a live stopping rule can see.",
            "value": {"proposed_stops": [24, 24, 16]},
            "source": f"{TASK8}/stopping_results.csv",
        },
        {
            "id": "task8b_pod_gp",
            "claim": "At 16 training runs, typical outcome error was about 31% lower for kinetic energy and 52% lower for "
                     "pressure RMS than a standard baseline (POD-GP) trained on the same runs.",
            "qualification": "A follow-up analysis of the same held-out tests, three repeats. The baseline beat "
                             "the FNO on kinetic energy in one repeat. Not evidence that pretraining alone caused it.",
            "value": {"median_ratio_16": {"kinetic_energy": 0.6936302575871508, "pressure_rms": 0.4754388753385918},
                      "per_repetition_16": {"kinetic_energy": [1.271, 0.554, 0.694], "pressure_rms": [0.402, 0.739, 0.475]},
                      "median_ratio_8": {"kinetic_energy": 0.26800315780041684, "pressure_rms": 0.2543223614844491}},
            "metric": "median over repetitions of FNO RMSE / POD-GP RMSE",
            "protocol": "e062f11164540ed06dfe952ea292f808fe84ee908ace4ba696cf6c08c619547b",
            "source": f"{TASK8B}/objective2.csv",
        },
    ],
    "details": {
        "task8b_savings": {
            "statement": "Looking back, at tolerance 0.008 even coverage needed a median of 8 fewer training runs than the best "
                         "baseline; at 0.015 it needed the same number; at 0.004 it needed 8 more.",
            "status": "Run savings were not shown: the follow-up analysis did not meet its savings goal or its cheaper "
                      "calibration goal; only the accuracy comparison with the baseline passed. No money figures.",
            "source": f"{TASK8B}/objective3.csv and verdict.json",
        },
        "task8_prospective_savings": {
            "statement": "In the original study, uncertainty-driven FNO selection saved 0, -16 and 0 runs against the baseline.",
            "status": "Run savings against the baseline were not established.",
            "source": f"{TASK8}/paired_savings.csv",
        },
        "labels_example": {"train": 16, "development": 12, "calibration": 39, "total": 67,
                           "statement": "The supported calibrated example consumes 16 + 12 + 39 = 67 labelled settings."},
        "not_claimed": ["guaranteed 90% confidence for an individual prediction", "100-to-16 or 84% fewer simulations",
                        "predictions replacing solver runs", "a causal pretraining advantage",
                        "validation on customer CFD, OpenFOAM, geometry or 3D"],
    },
}
