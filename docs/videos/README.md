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

## Recording conventions

- Record against the released version (`uv add iohblade`), not a checkout.
- Set `MLFLOW_DISABLE_AGENT_HINT=1` in the recording shell; otherwise
  MLflow prints a log line on every command.
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
