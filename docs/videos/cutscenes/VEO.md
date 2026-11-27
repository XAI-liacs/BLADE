# Veo prompts for the cutscenes

Each cutscene is a still in `frames/` (generate them first, see
`README.md`). Make the motion in Veo with image-to-video, 16:9, 1080p,
6–8 s.

## How to use them

**Mode A, "alive" (default):** use the PNG as both the first and the last
frame. Veo then animates in between and returns to the exact still, so the
text ends sharp and the clip loops. Every prompt below is written for this
mode unless it says otherwise.

**Mode B, "build":** first frame `00-blank.png`, last frame the scene.
Veo builds the scene up. Use it only where noted; with a lot of text, Veo
tends to garble letters while they appear.

Rules that make Veo behave:

- Describe motion only; the frame already carries the design.
- Always end with the **shared suffix** below. Video models invent new
  lettering and objects unless told not to.
- If any text wobbles, cross-fade to the sharp PNG for the last second
  (or the whole clip, with only the background moving).
- Generate two or three takes and pick the calmest one.

**Shared suffix** (add to every prompt):

> Smooth, slow, minimal motion-graphics animation on a dark navy background
> with a soft violet glow at the top right and a faint pink glow at the
> bottom left. Static camera, no zoom, no shake. All text stays exactly as
> it is, sharp and still. No new text, no new objects, no people.

## Background

**`00-blank`** (mode A)
> The violet glow at the top right and the pink glow at the bottom left
> breathe slowly, drifting a little and pulsing softly in brightness, like
> light through mist. Tiny faint particles float upwards.

## Benchmarks (`bench-01` … `bench-15`)

The right half (key, name, tagline) must stay still; only the illustration
in the framed tile on the left moves. Start each prompt with:

> Only the illustration inside the rounded frame on the left moves; the
> text on the right and the BLADE logo stay completely still.

**`bench-01-bbob`** (24 landscapes)
> The 24 small colour landscapes shimmer one after another in a wave from
> the top left to the bottom right, their glowing amber minima pulsing
> softly, as if each function is being evaluated in turn.

**`bench-02-bbob-groups`** (five group landscapes)
> The five landscapes light up one at a time, from the top left to the
> bottom right, each brightening and then settling back, while the colours
> flow slowly along their contour lines.

**`bench-03-ma-bbob`** (one landscape)
> The colourful landscape slowly morphs: its bright amber ridges and violet
> valleys flow and twist gently, as if two functions are being blended
> together, and then settle back into the original pattern. The white dot
> at the optimum glows steadily.

**`bench-04-packing`** (circles in a square)
> The coloured circles gently wobble and nudge against each other inside
> the square, like bubbles settling into a tight packing, then come to rest
> in their original positions.

**`bench-05-heilbronn`** (points in a triangle)
> The amber points drift slightly inside the triangle; the highlighted pink
> triangle between three of them pulses and grows a little, as if being
> made larger, and the faint lines flicker softly.

**`bench-06-spheres`** (points on a sphere)
> The sphere rotates slowly around its vertical axis; the amber points on
> its surface turn with it, glowing brighter on the front, and the thin
> pink links between neighbouring points stay attached.

**`bench-07-analysis`** (step function and autoconvolution)
> The violet bars of the step function rise and fall gently, one after
> another, and the pink curve below ripples in response, its peak dipping
> towards the amber line.

**`bench-08-number-theory`** (sums and differences)
> The glowing dots on the three number lines pulse in sequence from left
> to right, row by row, like counting the sums and differences.

**`bench-09-matrix`** (A × B = C, seven products)
> The violet and pink matrix cells glow in pairs, and a soft light travels
> into the seven mint boxes one after another, then on into the amber
> result cells.

**`bench-10-graphs`** (Steiner tree and coloured graph)
> Light pulses along the mint Steiner tree from the four amber endpoints to
> the two diamond junctions. Below, the coloured nodes of the graph glow in
> turn, and the edges shimmer faintly.

**`bench-11-routing`** (vehicle routes)
> Small bright pulses travel along each coloured route, out from the white
> depot in the centre and back again, like vehicles driving their tours.
> The customer dots glow as a pulse passes them.

**`bench-12-automl`** (data → pipeline → ROC)
> Light flows from the data table into the pipeline boxes, one box after
> another from top to bottom, and the mint ROC curve below draws itself
> from the bottom left to the top right.

**`bench-13-kerneltuner`** (heatmap of configurations)
> The grid of configurations flickers softly as if being measured, a wave
> of brightness sweeps across it, and the outlined fastest cell glows
> steadily brighter.

**`bench-14-photonics`** (layer stack and spectrum)
> The amber light ray reflects off the striped layer stack with a gentle
> shimmer, the layers glint as light passes through them, and the pink
> spectrum curve below ripples slightly around its flat stop band.

**`bench-15-multi-objective`** (Pareto front)
> The grey points drift a little, while the pink points on the stepped
> front glow brighter and the front line shimmers from left to right.

## Analysis

**`ceg`** (code evolution graph, mode A or B)
> The graph grows from left to right: glowing dots appear one after another
> with thin lines connecting each to its parent, and the bright pink line
> of descent lights up last, ending at the circled best algorithm. The
> colour legend on the right stays still.

(For mode B, use the prompt as written; for mode A, replace the first
sentence by "Light travels along the lines from the first dot on the left
to the circled best algorithm, and the dots pulse softly as it passes.")

**`convergence`** (mode A)
> The four glowing curves pulse softly with light running along them from
> left to right, and their shaded bands breathe slightly. The end points
> glow. The legend and the axes stay still.

## How BLADE works

**`framework`** (five stages)
> A soft light travels from left to right through the five coloured
> columns, following the arrows: each column brightens in turn as the
> light passes, then settles. The item boxes inside stay still.

**`framework-detailed`** (the platform figure)
> A slow, gentle light sweep moves from left to right across the figure,
> brightening each panel as it passes. The figure itself does not move.

**`aad-loop`** (the design loop, mode A or B)
> A bright pulse of light runs along the arrows: from the task prompt into
> the LLM box, on to the population, then round the loop through scoring,
> selection and LLM mutation, and back to the population. Each box glows
> as the pulse enters it. The three code cards in the middle shuffle
> slightly, and the highlighted one rises to the front.

**`isolation`** (sandbox)
> The code on the left glows briefly and a pulse travels along the arrow
> into the dashed sandbox, whose border shimmers like a force field. Then
> three pulses leave the sandbox along the curved arrows, one into each
> outcome box, mint first, then pink, then amber.

**`methods`** (nine methods)
> The nine boxes light up one after another, row by row, each with a soft
> glow that fades again, like a roll call.

**`llms`** (provider logos)
> The white logo circles float up and down very slightly, out of step with
> each other, and their coloured halos pulse softly. The logos themselves
> do not change.

**`yaml-terminal`** (mode A)
> A soft light sweeps down the YAML file line by line, then jumps across
> the arrow and runs down the terminal, ending with a gentle blinking glow
> after the last command. The text does not change.

## Episode cards

**`episode-1` … `episode-7`** (title cards)
> The faint line graphs in the background climb slowly upwards in steps,
> glowing softly, and the leaf logo shimmers once. The title text stays
> still.

**`next-1` … `next-6`** (up next)
> The faint line graphs in the background drift slowly to the left, as if
> moving on to the next chapter, and the "UP NEXT" label glows softly. All
> text stays still.

## Not for Veo

**`lower-third`** is a transparent overlay: animate it in the video editor
(slide in from the left over 0.4 s, hold, fade out), not in Veo.
