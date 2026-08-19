# Full-H, LR-corrected SR, and predicted-response comparison

This folder compares four consistent references/outputs:

- actual DFT;
- the direct Full-H model;
- the original LR-corrected SR model, using fixed reference response tensors;
- the new SR model reconstructed with its predicted Born charges and
  dielectric tensor. The plotted EPC result is the geometry-dependent mode.

The regular Hamiltonian comparison uses the five locked tensor-test snapshots,
which are also members of the original 37-snapshot held-out split. No training
or validation structures are mixed into these plots. DFT has zero prediction
error by definition and is shown as the reference.

## Headline values

| Model | Full-H MAE (meV) | EPC relative L2 | EPC complex MAE (eV/A) |
|---|---:|---:|---:|
| Actual DFT | 0 | 0 | 0 |
| Direct Full-H | 0.416583 | 0.919283 | 0.552189 |
| LR-corrected SR | 0.486775 | 0.243558 | 0.267903 |
| SR + predicted Z* + epsilon_inf | 0.484999 | 0.244328 | 0.268813 |

The equilibrium-frozen and geometry-dependent predicted-tensor EPC results are
nearly identical at the 5e-6 A finite-difference displacement: relative L2
difference `5.991e-10` and complex MAE
`4.257e-10 eV/A`.

Every figure is provided as both PNG and vector PDF. Exact metrics, input paths,
and SHA-256 hashes are in `metrics.json`; rerun with
`python -m workflows.analysis.generate_plots`.

## Presentation

`MACE-H-LR_Learned_Response_Results_2026-08-17.pptx` is a 12-slide results
presentation built from these figures. It includes the model/training protocol,
locked tensor and Hamiltonian results, EPC comparisons, the current scientific
interpretation, and proposed next experiments. Speaker notes are embedded in
the PowerPoint and duplicated in
`MACE-H-LR_Learned_Response_Results_2026-08-17_speaker_notes.md`.

Regenerate the deck with:

    python -m workflows.analysis.build_results_presentation

### SURF final-talk version

`Jeremiah_Bailey_SURF_Final_Presentation_4x3_2026.pptx` is the final-talk
edition. It is a true 4:3 deck, written for a general audience and paced for a
15-minute talk followed by 2–3 minutes of questions. It adds motivation,
background, a roadmap, plain-language explanations, acknowledgments, and a
final slide containing Jeremiah Bailey's name and the supplied audience
feedback QR code. Its embedded notes are also available in
`Jeremiah_Bailey_SURF_Final_Presentation_4x3_2026_speaker_notes.md`.

Regenerate it with:

    python -m workflows.analysis.build_surf_final_4x3
