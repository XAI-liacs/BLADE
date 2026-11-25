# Episode 2: Experiments in YAML - terminal parts of scenes 4 to 8.
# Run from a folder with experiment.yml, typos.yml and episode 1's
# my-first-benchmark/ (with its short results). Needs GEMINI_API_KEY set.
# The YAML writing (scenes 2 to 5) is a screen recording of the editor.

Output episode.mp4

Set Shell "bash"
Set FontSize 22
Set Width 1920
Set Height 1080
Set Padding 40
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

# Scene 4 and 5: what is built in
Type "uv run iohblade list methods"
Enter
Sleep 6s
Ctrl+L
Type "uv run iohblade list suites"
Enter
Sleep 8s
Ctrl+L

# Scene 6: validate
Type "uv run iohblade validate typos.yml"
Enter
Sleep 8s
Type "uv run iohblade validate experiment.yml"
Enter
Sleep 10s
Ctrl+L

# Scene 7: --set and --dry-run
Type "uv run iohblade run experiment.yml --dry-run --set experiment.runs=1"
Enter
Sleep 10s
Ctrl+L

# Scene 8: resuming
Type "uv run iohblade run my-first-benchmark/experiment.yml --dry-run"
Enter
Sleep 10s
Ctrl+L
Type "uv run iohblade run my-first-benchmark/experiment.yml --dry-run --set experiment.runs=1 --set methods.1.name=Random"
Enter
Sleep 12s
