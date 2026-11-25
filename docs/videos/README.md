# BLADE video tutorials

Source material for the tutorial series (issue #234): one folder per
episode, each with a narration script and a [VHS](https://github.com/charmbracelet/vhs)
tape for the terminal parts, so an episode can be re-recorded when the CLI
changes.

| # | Episode | Status |
|---|---------|--------|
| 1 | BLADE in 5 minutes | script draft |
| 2 | Experiments in YAML | planned |
| 3 | Local and hosted LLMs | planned |
| 4 | Your own problem | planned |
| 5 | Your own method | planned |
| 6 | Analysing results | planned |
| 7 | BLADE with a coding agent | planned |

Recording order: 1, 2, 6, 4, 3, 5, 7.

## Format

- Terminal recorded with VHS (1920×1080, font size 22), narration recorded
  separately in your own voice and laid over the recording; captions from
  the narration text.
- Slides or figures (the platform figure, the banner) as stills between
  terminal scenes.
- 3 to 8 minutes per episode. Long-running steps (a real run) are recorded
  live and shown as a time-lapse with an on-screen note.

## Standard opening (every episode, 10–15 s)

Every episode starts with the same short opening, recorded once and reused,
followed by the episode's own title card.

**Screen:** the banner with a name lower-third, then a simple loop
animation: *LLM writes code → evaluate → score fed back*.

**Narration (about 38 words):**

> I'm Niki van Stein, professor of explainable AI and optimisation at
> Leiden University. In LLM-driven algorithm design, a language model
> writes optimisation algorithms, we score them, and the model uses those
> scores to write better ones, generation after generation.

Then the title card, e.g. "Episode 1 · BLADE in 5 minutes", for 2–3 s
without narration.

## Recording conventions

- Record against the released version (`uv add iohblade`), not a checkout.
- Start from an empty folder and clear `~/.cache` only if the episode is
  meant to show the first-time environment build.
- Narration in the script is in the first person plural ("we"), matching
  the campaign copy.

## Rendering a tape

```bash
vhs docs/videos/01-blade-in-5-minutes/episode.tape
```

VHS needs `ttyd` and `ffmpeg`. The tape writes `episode.mp4` next to
itself; the narration is added in the video editor.
