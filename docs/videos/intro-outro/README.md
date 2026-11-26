# Intro and outro: key frames for Veo

Style **Convergence**: four best-so-far curves (violet, pink, amber, mint)
climb in steps from the bottom left and meet in the BLADE leaf; the outro
dims them behind an end card with the links. Generate each clip in Veo with
a first and a last frame (image-to-video with a last frame), 6–8 s, 16:9,
1080p.

| Clip | First frame | Last frame |
|---|---|---|
| Intro | `frames/intro-first.png`: the curves just starting | `frames/intro-last.png`: the curves meet in the leaf; BLADE and the tagline at the top left |
| Outro | `frames/intro-last.png` (the same shot, so intro and outro match) | `frames/outro-last.png`: curves dimmed; leaf, BLADE, GitHub, `pip install iohblade`, docs |

Regenerate the frames (1920×1080) with:

```bash
python docs/videos/intro-outro/make_frames.py
NODE_PATH=$(npm root -g) node docs/videos/intro-outro/render.js
```

## Veo prompts

Describe the motion only; the frames carry the design. Ask for no new text,
because video models tend to garble lettering while it moves.

**Intro** (first `intro-first.png`, last `intro-last.png`)
> Four glowing line graphs in violet, pink, amber and mint climb in steps
> from the bottom left towards the upper right of a dark chart, like
> optimisation progress, and converge into a bright point that becomes a
> white leaf logo with a soft violet halo. The wordmark fades in at the top
> left. Smooth, steady motion on a dark navy background with soft violet
> light, no camera shake, no new text, no extra objects.

**Outro** (first `intro-last.png`, last `outro-last.png`)
> The leaf logo and the wordmark glide to the centre while the line graphs
> dim and sink into the dark background, and the links fade in below the
> wordmark. Slow and calm, dark navy background with a soft violet glow, no
> new text, no extra objects.

## Tips

- Put the episode title on its own title card in the editor (2–3 s after the
  intro), not in the Veo frames, so one intro clip serves every episode.
- If Veo distorts the wordmark or the links, cross-fade to the sharp PNG of
  the last frame for the final second.
- The standard opening narration (about 15 s) runs over the intro clip and
  the title card.
