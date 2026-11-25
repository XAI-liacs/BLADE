# Episode 5: Your own method - terminal parts of scenes 2 and 5.
# Run from an empty folder; Ollama with qwen2.5-coder:14b for the run.

Output episode.mp4

Set Shell "bash"
Set FontSize 22
Set Width 1920
Set Height 1080
Set Padding 40
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

# Scene 2: the template
Type "uv run iohblade init my-method --template custom-method"
Enter
Sleep 4s
Ctrl+L

# Scene 5: check, then the real run (time-lapse in the edit)
Type "uv run iohblade check my-method/experiment.yml"
Enter
Sleep 40s
Ctrl+L
Type "uv run iohblade run my-method/experiment.yml"
Enter
# Adjust to the real duration of the run on the recording machine.
Sleep 1800s
