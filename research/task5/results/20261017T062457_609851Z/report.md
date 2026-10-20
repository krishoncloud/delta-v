# Delta-V Task 5 measured development report

Status: **READY FOR TASK 6**

Study: controlled_taylor_green_2d. Source checkpoint: 2aefbf631c8ce81f5037646b08d6e16ece9e736cbfda1b37ad4c924e261e1674.
Bundle manifest: 89d53cdf79d6406b614ab01af8dd14b1a41bd774908db1a5d1f457b105038b4d.

Training settings: 16; development settings: 9.
Completed steps: 300; selected step: 280; training-stage seconds: 13.40.
Initial development loss: 1.9359; selected development loss: 0.00198797.

## Completion gates
{
  "training_field_loss_improved": true,
  "development_loss_improved": true,
  "beats_mean_on_at_least_one_outcome": true,
  "reload_matches": true,
  "query_batch_invariant": true,
  "excluded_labels_invariant": true,
  "parameter_conditioning_changes_field": true,
  "query_requires_no_targets": true
}

## Engineering outcome errors
RMSE/MAE are in declared units; normalized RMSE uses training outcome standard deviation, not percentage error.

```csv
method,outcome,unit,rmse,mae,rmse_over_train_std
mean,kinetic_energy,nondimensional,0.16111521599515785,0.1412115107393927,1.14227502924814
mean,pressure_rms,nondimensional,0.16111522359258196,0.1412115168964697,1.142275083112375
linear,kinetic_energy,nondimensional,0.028450720516520638,0.025784438398870797,0.20170998381131164
linear,pressure_rms,nondimensional,0.028450724075126377,0.025784442979346137,0.2017100090411228
gp,kinetic_energy,nondimensional,0.00027702733362078877,0.0002023845372498957,0.0019640690276189125
gp,pressure_rms,nondimensional,0.00027703596006838343,0.0002023909338712533,0.001964130187426927
fno,kinetic_energy,nondimensional,0.007471169602163569,0.006123714231806894,0.052969115443979475
fno,pressure_rms,nondimensional,0.005768088181171576,0.0048602908258061185,0.04089460486495348
```

## Interpretation and limitations
- Real source FNO weights are strictly loaded; the input conditioner is newly trained. All backbone weights are fine-tuned.
- Known initial fields and knobs are the only query inputs. No query future frames or outcomes are consumed.
- FNO additionally uses field supervision. GP/linear/mean see the same runs' controls and scalar outcomes.
- Development selects the checkpoint; errors are exploratory. Task 6 must compare identical-input scratch and adapter arms on fresh tests.
- Its two default scalar outcomes are mathematically equal in this vortex, not two independent success cases.
- The default vortex study is an exact analytical development fixture, not expensive customer CFD. It establishes no solver-run or money savings.
- Source-training code indicates three systems; the legacy checkpoint lacks a complete exposure ledger.
- Source resolution 256x256 is adapted to 64x64 here. Generalization to other resolutions/geometries is not established.
- Reserved settings evaluated: zero. Uncertainty is not calibrated. No next-run policy is implemented.
- Baseline warnings recorded: 3. No weights or customer data uploaded remotely.

See task6_handoff.json for the next protocol and model_contract.json for the exact saved inference contract.
