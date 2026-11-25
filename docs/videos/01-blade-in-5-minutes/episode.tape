# Episode 1: BLADE in 5 minutes - terminal scenes 3 to 7.
# Render with: vhs docs/videos/01-blade-in-5-minutes/episode.tape
# Needs: uv, Ollama with qwen2.5-coder:14b, an empty working folder.
# Scene 4's YAML walk-through and scene 8 (web app) are screen recordings.

Output episode.mp4

Set Shell "bash"
Set FontSize 22
Set Width 1920
Set Height 1080
Set Padding 40
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

# Scene 3: install
Type "uv init blade-demo && cd blade-demo"
Enter
Sleep 2s
Type "uv add iohblade"
Enter
Sleep 15s
Ctrl+L

# Scene 4: a first experiment
Type "uv run iohblade init my-first-benchmark"
Enter
Sleep 4s
Type "cat my-first-benchmark/experiment.yml"
Enter
Sleep 12s
Ctrl+L

# Scene 5: validate
Type "uv run iohblade validate my-first-benchmark/experiment.yml"
Enter
Sleep 8s
Ctrl+L

# Scene 6: check
Type "uv run iohblade check my-first-benchmark/experiment.yml"
Enter
Sleep 40s
Ctrl+L

# Scene 7: run (shown as a time-lapse in the edit)
Type "uv run iohblade run my-first-benchmark/experiment.yml --set experiment.runs=1 --set experiment.budget=10"
Enter
# Adjust to the real duration of the run on the recording machine.
Sleep 600s
