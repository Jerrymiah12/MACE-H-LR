# Jeremiah Bailey — SURF final presentation speaker notes

Format: 4:3  
Target speaking time: 15 minutes  
Q&A: 2–3 minutes  
Audience: general Caltech audience

## 1. Teaching machines how atoms and electrons move together

0:00–0:35. Hello, my name is Jeremiah Bailey. This summer I worked in the
Bernardi Group on a question at the intersection of materials physics and
machine learning: can we teach a model not only the electronic state of a
material, but also how that state changes when the atoms move? I will focus on
magnesium oxide, a simple polar crystal that makes this challenge visible.

## 2. Atoms vibrate—and electrons must respond

0:35–1:35. Atoms in a solid are always moving. When they move, the electrons
rearrange. That interaction affects resistance, energy relaxation, heat, and
optical behavior. Quantum-mechanical calculations can describe it accurately,
but repeating those calculations for every possible atomic motion is costly.
The opportunity is to learn this response from examples while keeping the
relevant physics in the model.

## 3. Four questions guide the talk

1:35–2:05. I will first define the quantities in plain language. Then I will
explain why polar magnesium oxide needs special treatment, show the model and
training protocol I built, and finish with what the data support so far and
what experiment should come next.

## 4. Three ideas connect structure to response

2:05–3:20. DFT is my physics-based reference: accurate, but expensive for
large numbers of structures. The Hamiltonian is a compact matrix that acts
like an electronic rulebook. Machine learning can predict that matrix from
atomic positions. Electron–phonon coupling is more demanding: it is the slope
of that prediction when an atom moves. A model can match many values and still
learn the wrong slope between them.

## 5. Polar crystals communicate over long distances

3:20–4:25. Magnesium and oxygen carry opposite ionic character. Moving one
atom creates an electric response that is not purely local. I compare four
things: the DFT reference, a model trained directly on the complete
Hamiltonian, a short-range model corrected with fixed reference physics, and
my new model whose effective charges and electronic screening are predicted
from the current geometry.

## 6. One structural encoder feeds three physical predictions

4:25–5:25. I built one shared equivariant encoder, meaning its outputs transform
correctly when the crystal is rotated. It feeds three predictions: the local
Hamiltonian, an effective charge for every atom, and one screening tensor for
the structure. Those pieces are combined with an analytic long-range formula.
The model enforces overall charge balance and positive screening by design.

## 7. Separate data answer separate questions

5:25–6:20. Hamiltonians are large, so the model retained all 367 short-range
training and validation structures. The response tensors are expensive DFT
labels: ten trained the heads, two guided wiring and stability, and five were
locked. I opened those five only after freezing the architecture, learning
rates, and checkpoint. The accepted model had to pass all three validation
gates ten times in a row.

## 8. The new tasks did not destabilize the original model

6:20–7:15. The left panel gives the longer training context. The center asks
whether adding new tasks damages the original short-range Hamiltonian; it
stays below the dashed limit. The right asks whether the charge and screening
heads beat simple baselines; both stay below one. After ten consecutive passes,
training stopped and I froze the checkpoint.

## 9. The learned tensors beat constant guesses on unseen structures

7:15–8:15. This is the first success criterion. On structures never used for
training or selection, the learned effective charges reduce mean absolute
error by about 26 percent compared with always predicting the training mean.
Screening error drops by about 73 percent. The outputs also satisfy the
physical constraints. This shows learned geometry-dependent signal, while the
small test set keeps the claim at prototype scale.

## 10. For electronic values, direct Full-H still leads

8:15–9:15. Next I compare matrix values on those same five structures. Zero is
the DFT reference. Direct full-H has the smallest mean error at 0.417
millielectron-volts. The fixed-physics SR result is 0.487, and learned tensors
slightly improve it to 0.485. So the composed model is complete and stable,
but the tensor heads do not close the value-accuracy gap.

## 11. For atomic-motion response, both SR models win

9:15–10:20. The ranking reverses for the response. Here lower is again better.
Direct full-H has about 92 percent relative error, while both short-range
pipelines are near 24 percent. This is the central lesson: a model can predict
Hamiltonian values well and still learn the wrong slope. Physics-guided target
design improves the derivative observable even when it does not win on values.

## 12. Longer-wavelength labels separate value and response quality

10:20–11:25. q is simply a label for a vibration's wavelength and direction.
The top panel shows coupling strength; the bottom shows error against DFT. At
q equals zero, all models are much closer. Away from zero, direct full-H
becomes too strong. The blue fixed-tensor and green learned-tensor curves
overlap across this grid, which leads to the final interpretation.

## 13. The new heads work—but their EPC effect is below present sensitivity

11:25–12:30. The heads work as predictors, and the composed reconstruction is
correct. But the learned geometry dependence changes EPC by only about four
times ten to the minus ten electron-volts per ångström on average at this
finite-difference step. That does not prove the tensors are physically
irrelevant. It means this experiment cannot yet separate their effect from
near-neutrality. The dataset and q grid also limit the scope of the claim.

## 14. Make the tensor effect measurable before scaling up

12:30–13:35. My first next step would be a displacement-sensitivity ladder,
because the current five-microångström step may make geometry-dependent tensor
changes invisible. Then I would add response-diverse labels and repeat matched
ablations with the same initialization. A denser q grid is valuable, but only
after the tensor contribution is measurable independently of numerical noise.

## 15. Physics-guided learning improves response—even before tensors help

13:35–14:35. In summary, I built the intended three-output architecture and
verified it on locked data. The response heads learn real signal. Direct
full-H remains best for matrix values, but target design around short-range
physics produces far better atomic-motion response. Learned tensors have not
yet improved EPC at this sensitivity. Thank Prof. Marco Benardi, my graduate
mentor Yao Luo, and WAVE at Caltech for their support.

## 16. Questions and audience feedback

14:35–15:00. Thank the audience and invite questions. Leave this final slide
displayed throughout the 2–3 minute Q&A so the QR code remains available.
