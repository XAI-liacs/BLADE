# Adapted for BLADE (iohblade) from ShinkaEvolve 0.0.8
# (https://github.com/SakanaAI/ShinkaEvolve, shinka/prompts and
# shinka/defaults.py), Copyright Sakana AI, licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Changes: the
# whole program is editable (no EVOLVE-BLOCK markers), fitness is shown with
# the problem's optimisation direction instead of a "combined score to
# maximize", the BLADE problem description is part of the system text, and
# the error feedback of BLADE evaluations replaces stdout/stderr logs.
"""Prompt texts of ShinkaEvolve."""

from __future__ import annotations

DEFAULT_TASK_SYS_MSG = (
    "You are an expert optimization and algorithm design assistant. "
    "Improve the program while preserving correctness."
)

_NAME_DESCRIPTION = """<NAME>
A shortened name summarizing the {what} you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
{description}
</DESCRIPTION>"""

DIFF_SYS_FORMAT = (
    """
You MUST respond using an edit name, description, and the exact SEARCH/REPLACE diff format shown below to indicate changes:

"""
    + _NAME_DESCRIPTION.format(
        what="edit",
        description="A description and argumentation process of the edit you are proposing.",
    )
    + """

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

</DIFF>


Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

</DIFF>

* Every block's SEARCH section must be copied **verbatim** from the current program, including indentation.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the program still runs after your changes.
"""
).rstrip()

DIFF_TASK = """# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to {goal}.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements."""

_CODE_FORMAT = """

<CODE>
```python
# {placeholder}
```
</CODE>

* Make sure your {kind} program maintains the same inputs and outputs (class interface) as the original program{suffix}.
* Make sure the program still runs after your changes.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards."""


def _full_format(intro: str, description: str, placeholder: str, bullets=""):
    return (
        intro.strip()
        + "\nYou MUST respond using a short summary name, description and the full code:\n\n"
        + _NAME_DESCRIPTION.format(what="code", description=description)
        + _CODE_FORMAT.format(
            placeholder=placeholder,
            kind="rewritten",
            suffix=", but with improved internal implementation",
        )
        + bullets
    )


# Full rewrites: Shinka samples one of five variants uniformly.
FULL_SYS_FORMATS = {
    "default": _full_format(
        "Rewrite the program to improve its performance on the specified "
        "metrics.\nProvide the complete new program code.",
        "A description and argumentation process of the code you are proposing.",
        "The new rewritten program here.",
    ),
    "different_algorithm": _full_format(
        "Design a completely different algorithm approach to solve the same "
        "problem.\nIgnore the current implementation and think of alternative "
        "algorithmic strategies that could achieve better performance.",
        "Explain the completely different algorithmic approach you are taking "
        "and why it should perform better than the current implementation.",
        "The completely new algorithm implementation here.",
        "\n* Your algorithm should solve the same problem but use a "
        "fundamentally different approach.\n* Think outside the box - consider "
        "different data structures, algorithms, or paradigms.",
    ),
    "context_motivated": _full_format(
        "Create a novel algorithm that draws inspiration from the provided "
        "context programs but implements a fundamentally different approach.\n"
        "Study the patterns and techniques from the examples, then design "
        "something new.",
        "Explain how you drew inspiration from the context programs and what "
        "novel approach you are implementing. Detail the key insights that led "
        "to this design.",
        "The inspired but novel algorithm implementation here.",
        "\n* Learn from the context programs but don't copy their approaches "
        "directly.\n* Combine ideas in novel ways or apply insights to "
        "different algorithmic paradigms.",
    ),
    "structural_redesign": _full_format(
        "Redesign the program with a different structural approach while "
        "potentially using similar core concepts.\nFocus on changing the "
        "overall architecture, data flow, or program organization.",
        "Describe the structural changes you are making and how they improve "
        "the program's performance, maintainability, or efficiency.",
        "The structurally redesigned program here.",
        "\n* Focus on changing the program's structure: modularization, data "
        "flow, control flow, or architectural patterns.\n* The core "
        "problem-solving approach may be similar but organized differently.",
    ),
    "parametric_design": _full_format(
        "Analyze the current program to identify its key parameters and "
        "algorithmic components, then design a new algorithm with different "
        "parameter settings and configurations.",
        "Identify the key parameters in the current approach and explain how "
        "your new parameter choices or algorithmic configuration will lead to "
        "better performance.",
        "The new parametric algorithm implementation here.",
        "\n* Identify parameters like: learning rates, iteration counts, "
        "thresholds, weights, selection criteria, etc.\n* Design a new "
        "algorithm with different parameter values or configurations.\n* "
        "Consider adaptive parameters, different optimization strategies, or "
        "alternative heuristics.",
    ),
}

FULL_TASK = """# Task

Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation."""

CROSS_SYS_FORMAT = _full_format(
    "You are given multiple code scripts implementing the same algorithm.\n"
    "You are tasked with generating a new code snippet that combines these "
    "code scripts in a way that is more efficient.\nI.e. perform crossover "
    "between the code scripts.\nProvide the complete new program code.",
    "A description and argumentation process of the code you are proposing.",
    "The new rewritten program here.",
)

CROSS_TASK = """# Task

Perform a cross-over between the code script above and the one below. Aim to combine the best parts of both code implementations that improves the score.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation."""

FIX_SYS_FORMAT = (
    "You are debugging and fixing an incorrect program that has failed "
    "validation.\nYour task is to analyze the error output and fix the "
    "program so it passes validation.\nYou MUST respond using a short "
    "summary name, description and the full code:\n\n"
    + _NAME_DESCRIPTION.format(
        what="fix",
        description="Describe the bug you identified and the fix you are "
        "applying. Include your analysis of the error messages.",
    )
    + _CODE_FORMAT.format(
        placeholder="The fixed program here.", kind="fixed", suffix=""
    )
    + "\n* Focus on making the program correct first - performance "
    "optimization is secondary."
)

FIX_ITER_MSG = """# Incorrect Program to Fix

The following program has failed validation and needs to be fixed:

```python
{code}
```

## Error Information

The program is marked as **incorrect** and did not pass validation tests.

{error}

# Task

Analyze the error output above and fix the program. Focus on:
1. Understanding why the program failed validation
2. Identifying the root cause from the error messages
3. Implementing a fix that addresses the issue

IMPORTANT: Make the program correct first. Performance improvements can come later."""

PATCH_RETRY_MSG = (
    "The previous edit was not successful. Error: {error}\n\n"
    "Please try again, following the response format exactly."
)

NOVELTY_SYSTEM_MSG = """You are an expert code reviewer tasked with determining if two code snippets are meaningfully different from each other.

Your job is to analyze both programs and determine if the proposed code introduces meaningful changes compared to the existing code. Consider:

1. **Algorithmic differences**: Different approaches, logic, or strategies
2. **Structural changes**: Different data structures, control flow, or organization
3. **Functional improvements**: New features, optimizations, or capabilities
4. **Implementation variations**: Different ways of achieving the same goal that could lead to different performance characteristics
5. **Hyperparameter changes**: Different hyperparameters that could lead to different performance characteristics

Ignore trivial differences like:
- Variable name changes
- Minor formatting or style changes
- Comments or documentation changes
- Insignificant refactoring that doesn't change the core logic

Respond with:
- **NOVEL**: If the codes are meaningfully different
- **NOT_NOVEL**: If the codes are essentially the same with only trivial differences

After your decision, provide a brief explanation of your reasoning."""

NOVELTY_USER_MSG = """Please analyze these two code snippets:

**EXISTING CODE:**
```python
{existing_code}
```

**PROPOSED CODE:**
```python
{proposed_code}
```

Are these codes meaningfully different? Respond with NOVEL or NOT_NOVEL followed by your explanation."""

META_STEP1 = (
    "You are an expert programming assistant analyzing an individual program. "
    "Create a standalone summary focusing on implementation details and "
    "evaluation feedback. Consider how this specific program performs and "
    "what implementation choices were made.\n\n"
    "# Program to Analyze\n{program}\n\n"
    "# Instructions\n\n"
    "Create a standalone summary for this program using the following "
    "exact format:\n\n"
    "**Program Name: [Short summary name of the algorithm (up to 10 words)]**\n"
    "- **Implementation**: [Key implementation details (1-2 sentences)]\n"
    "- **Performance**: [Score/metrics summary (1 sentence)]\n"
    "- **Feedback**: [Key insights from evaluation (1-2 sentences)]\n\n"
    "Focus on:\n"
    "1. What specific implementation details were done\n"
    "2. How these details affected performance\n"
    "3. Implementation details that are relevant to the approach\n"
    "4. Any evaluation feedback that provides insights\n\n"
    "Keep the program summary concise but informative. Follow the format exactly."
)

META_STEP2 = (
    "You are an expert programming assistant analyzing specific program "
    "evaluation results to extract actionable optimization insights. Focus "
    "on concrete performance data and implementation details from the actual "
    "programs that were evaluated.\n\n"
    "# Individual Program Summaries\n{summaries}\n\n"
    "# Previous Global Insights (if any)\n{previous}\n\n"
    "# Current Best Program\n{best}\n\n"
    "# Instructions\n\n"
    "Analyze the SPECIFIC program evaluation results above to extract "
    "concrete optimization insights. Look at the actual performance scores, "
    "implementation details, and evaluation feedback to identify patterns. "
    "Reference specific programs and their results.\n"
    "Make sure to incorporate the previous insights into the new insights.\n\n"
    "**CRITICAL: Pay special attention to the current best program and how it "
    "compares to other evaluated programs. Ensure the best program's successful "
    "patterns are prominently featured in your analysis.**\n\n"
    "Update or create insights in these sections:\n\n"
    "## Successful Algorithmic Patterns\n"
    "- Identify specific implementation changes that led to score improvements\n"
    "- Reference which programs achieved these improvements and their scores\n"
    "- **Highlight patterns from the current best program**\n"
    "- Note the specific techniques or approaches that worked (2-4 bullet points)\n\n"
    "## Ineffective Approaches\n"
    "- Identify specific implementation changes that worsened performance\n"
    "- Reference which programs had these issues and how scores were affected\n"
    "- Note why these approaches failed based on evaluation feedback "
    "(2-4 bullet points)\n\n"
    "## Implementation Insights\n"
    "- Extract specific coding patterns/techniques from the evaluated programs\n"
    "- **Analyze what makes the current best program effective**\n"
    "- Connect implementation details to their performance impact\n"
    "- Reference concrete examples from the program summaries (2-4 bullet points)\n\n"
    "## Performance Analysis\n"
    "- Analyze actual score changes and trends from the evaluated programs\n"
    "- **Compare other programs' performance against the current best**\n"
    "- Compare performance between different implementation approaches\n"
    "- Identify score patterns and correlations (2-4 bullet points)\n\n"
    "CRITICAL: Base insights on the ACTUAL individual program summaries, "
    "previous insights, and the current best program information. "
    "Reference specific program names, scores, and implementation details. "
    "Build upon previous insights with concrete evidence from the new "
    "evaluations. IMPORTANT: Make sure that the best results are not ignored "
    "and are prominently featured in your analysis. Do not make "
    "recommendations for the next steps. ONLY PERFORM THE ANALYSIS."
)

META_STEP3 = (
    "You are an expert programming assistant generating actionable "
    "recommendations for future program mutations based on successful "
    "patterns and insights.\n\n"
    "# Global Insights\n{insights}\n\n"
    "# Previous Recommendations (if any)\n{previous}\n\n"
    "# Current Best Program\n{best}\n\n"
    "# Instructions\n\n"
    "Based on the global insights above and the current best program, "
    "generate {n} actionable recommendations for future program mutations. "
    "Each recommendation should be:\n\n"
    "1. **Specific**: Clear about what to implement or try\n"
    "2. **Actionable**: Something that can be directly applied\n"
    "3. **Evidence-based**: Grounded in the successful patterns identified\n"
    "4. **Diverse**: Cover different types of optimizations\n"
    "5. **Best-program informed**: Consider what makes the current best "
    "program successful\n\n"
    "Format as a numbered list:\n\n"
    "1. [Specific recommendation based on successful patterns]\n"
    "2. [Another recommendation focusing on a different aspect]\n"
    "...\n\n"
    "**CRITICAL: Prioritize recommendations that build upon or extend the "
    "successful patterns from the current best program. Consider both "
    "incremental improvements to the best program's approach and novel "
    "variations that could surpass it.**\n\n"
    "Focus on the most promising approaches that have shown success in "
    "recent evaluations, especially those demonstrated by the best program. "
    "Avoid generic advice - provide 2-3 sentences per recommendation. "
    "DO NOT RECOMMEND CHANGING THE EVALUATION CODE. ONLY MAKE ALGORITHMIC "
    "RECOMMENDATIONS."
)


def direction_text(minimisation: bool) -> str:
    return "lower is better" if minimisation else "higher is better"


def goal_text(minimisation: bool) -> str:
    return "minimize the fitness" if minimisation else "maximize the fitness"
