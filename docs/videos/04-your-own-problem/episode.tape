# Episode 4: Your own problem - terminal parts of scenes 2, 5 and 6.
# Run from an empty folder. The code walk-through (scenes 3, 4, 7) and the
# typo in scene 6 are made in the editor between the Hide/Show blocks.

Output episode.mp4

Set Shell "bash"
Set FontSize 22
Set Width 1920
Set Height 1080
Set Padding 40
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

# Scene 2: the template
Type "uv run iohblade init my-problem --template custom-problem"
Enter
Sleep 4s
Ctrl+L

# Scene 5: check passes
Type "uv run iohblade check my-problem/experiment.yml"
Enter
Sleep 40s
Ctrl+L

# Scene 6: introduce the typo off screen, check fails, fix it, check passes
Hide
Type "sed -i 's/- shift)/- shft)/' my-problem/my_problem.py && clear"
Enter
Show
Type "uv run iohblade check my-problem/experiment.yml"
Enter
Sleep 20s
Hide
Type "sed -i 's/- shft)/- shift)/' my-problem/my_problem.py && clear"
Enter
Show
Type "uv run iohblade check my-problem/experiment.yml"
Enter
Sleep 20s
