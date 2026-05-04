# Paper A — pilot additions from Lucian's review (May 2026)

These three additions are operational — they belong in the pilot pipeline,
not just the writeup. Tracked here so they don't get lost between
concept-note and code.

## 1. False-alarm panel for F3 (becomes F5)

**What.** A small-multiples panel showing conformal interval width across
entities at contamination level 0 only, with entities ordered by sample
size n.

**Why.** Width-vs-contamination is the diagnostic headline (F3). The
obvious reviewer confound is that width could be tracking sample size or
niche breadth, not contamination. F5 isolates this:

- If width is flat across entities at level 0, the diagnostic claim is
  much stronger — width genuinely responds to contamination, not n.
- If width rises with smaller n, we say so explicitly in Discussion and
  scope the diagnostic claim accordingly.

**Implementation.** One additional figure in the panel; computed from the
same conformal output as F3, just sliced differently. No extra Slurm time.

**Dual purpose.** This is also the falsification check for Paper B's
detector. If width tracks niche breadth more than contamination at level
0, Paper B's detection-without-ground-truth approach has a fundamental
confound and we want to know early.

## 2. Coverage target (a) vs (b) — Methods, not Supplementary

**What.** An upfront paragraph in the Methods section explicitly framing
the choice between two coverage definitions:

- (a) suitability vs held-out presence/absence labels (the standard ML
  question).
- (b) suitability vs benchmark suitability surface at the same pixel (the
  ecologically meaningful question).

We use (b) as the headline result and (a) as a robustness check, framed
as a deliberate choice motivated by conservation-decision relevance.

**Why move from Supplementary to Methods.** "Coverage" is then defined
relative to a model rather than to ground truth, and that is a real
methodological commitment. Burying it in Supplementary invites a reviewer
to call it out as if we'd hidden it. Pre-empt by being upfront.

**Implementation.** Methods paragraph + one Supplementary table comparing
(a) and (b) coverage values across the pilot cells.

## 3. Fold-count sensitivity (Supplementary)

**What.** Repeat the headline coverage analysis with 4-fold and 6-fold
basin-stratified CV in addition to the default 5-fold. Report in
Supplementary as a sensitivity table.

**Why.** The default 5-fold is convention, not principle. Reviewers will
ask whether the calibration result is fold-count-dependent. A short
sensitivity settles it without inviting deeper concerns.

**Implementation.** ~3x the conformal compute on the pilot panel. Cheap
because conformal is post-hoc — no model retraining needed, just
re-running the calibration with different fold splits.

---

## Pilot timeline (per Lucian's email)

- Two-to-three week window for the pilot.
- Checkpoint end of week 2 — share whatever data is in hand.
- Sit-down end of week 3 if anything is ambiguous.
- No binary kill on a noisy n=3 pilot — interpret cautiously.

## Paper B status

Decision parked until pilot data lands. If contamination-aware corrector
turns out to be careful-application-of-existing-methods, it slots in as
Paper A Section 5. If it grows into something genuinely novel for non-iid
spatial settings, spin it out as a standalone ML-venue paper.
