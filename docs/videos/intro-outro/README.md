# Intro and outro: key frames for Veo

First and last frames (1920×1080) for a short animated intro and outro,
in three styles. Generate the clip in Veo with the first and last frame
(image-to-video with a last frame), 6–8 s, 16:9, 1080p.

The intro's last frame is the same as the outro's first frame, so the
intro ends and the outro starts on the same logo shot.

| Style | Intro: first → last | Outro: first → last |
|---|---|---|
| A · Constellation | scattered points (generated algorithms) → the points gather into the leaf, BLADE wordmark | leaf of points → points drift apart, end card with links |
| B · Code to leaf | `experiment.yml` in an editor → code fades, leaf and wordmark | logo → terminal (`pip install`, `init`, `check`) and links |
| C · Convergence | four best-so-far curves start → they rise and meet in the leaf | logo over curves → curves dim, end card with links |

Frames are in `frames/` (`<style>-intro-first.png`, `<style>-intro-last.png`,
`<style>-outro-last.png`; the outro starts from `<style>-intro-last.png`).
Regenerate them with:

```bash
python docs/videos/intro-outro/make_frames.py
NODE_PATH=$(npm root -g) node docs/videos/intro-outro/render.js
```

## Veo prompts

Describe the motion only; the frames carry the design. Ask for no new text,
because video models tend to garble lettering while it moves.

**A · intro** (first `a-intro-first`, last `a-intro-last`)
> Slow, smooth cinematic motion on a dark navy background with soft violet
> and pink light. Hundreds of small glowing points connected by faint lines
> drift gently, then flow towards the left of the frame and settle into the
> shape of a leaf, like a swarm finding a pattern. The wordmark on the right
> fades in at the end. Subtle parallax, no camera shake, no new text, no
> extra objects.

**A · outro** (first `a-intro-last`, last `a-outro-last`)
> The glowing points that form the leaf loosen and drift outwards across the
> dark background, dimming as they spread, while the centred end card fades
> in. Calm, slow, elegant. No new text, no extra objects.

**B · intro** (first `b-intro-first`, last `b-intro-last`)
> A code editor window on a dark navy background. The camera slowly pushes
> in; the code lines softly blur and fade into the background while a white
> leaf logo and the large wordmark rise and sharpen in the centre. Smooth
> dissolve, soft violet glow, no camera shake, no new text.

**B · outro** (first `b-intro-last`, last `b-outro-last`)
> The centred logo glides to the right and shrinks while a terminal window
> slides in from the left and the links fade in below the logo. Smooth,
> minimal motion design, dark background, no new text.

**C · intro** (first `c-intro-first`, last `c-intro-last`)
> Four glowing line graphs in violet, pink, amber and mint climb in steps
> from the bottom left towards the upper right of a dark chart, like
> optimisation progress, and converge into a bright point that becomes a
> white leaf logo with a soft violet halo. The wordmark fades in at the top
> left. Smooth, steady motion, no camera shake, no new text.

**C · outro** (first `c-intro-last`, last `c-outro-last`)
> The leaf and wordmark glide to the centre while the line graphs dim into
> the background and the links fade in below. Slow and calm, dark background,
> no new text.

## Tips

- Put the episode title on its own title card in the editor (2–3 s after the
  intro), not in the Veo frames; it can change per episode without
  regenerating the clip.
- If Veo distorts the wordmark or links, generate the motion with the
  frames as they are and overlay the final frame (sharp) for the last
  second, or cross-fade to the PNG.
- The standard opening narration (about 15 s) runs over the intro clip and
  the title card.
