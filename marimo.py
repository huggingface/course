import marimo as mo

app = mo.app()


@app.cell
def cell_1():
    import marimo as mo

    return mo


@app.cell
def cell_2(mo):
    # Create a slider for the ideal length
    ideal_length = mo.ui.slider(
        min=5, max=50, value=20, step=1, label="Ideal Length (characters)"
    )

    mo.md("## Length Reward Function")

    return ideal_length


@app.cell
def cell_3(mo, ideal_length):
    # Toy dataset with 5 samples of different lengths
    completions = [
        "Short",  # 5 chars
        "Medium length text",  # 18 chars
        "This is about twenty chars",  # 25 chars
        "This is a slightly longer completion",  # 36 chars
        "This is a much longer completion with more words",  # 45 chars
    ]

    def length_reward(completions, ideal_length=ideal_length.value):
        """
        Calculate rewards based on the length of completions.

        Args:
            completions: List of text completions
            ideal_length: Target length in characters

        Returns:
            List of reward scores for each completion
        """
        rewards = []

        for completion in completions:
            length = len(completion)
            # Simple reward function: negative absolute difference
            reward = -abs(ideal_length - length)
            rewards.append(reward)

        return rewards

    # Calculate rewards for the examples
    rewards = length_reward(completions)

    # Display the examples and their rewards
    results = []
    for completion, reward in zip(completions, rewards):
        results.append(
            {"Completion": completion, "Length": len(completion), "Reward": reward}
        )

    mo.table(results)

    mo.md(f"""
    **How it works:**
    - Reward = -|ideal_length - actual_length|
    - Higher reward (closer to 0) is better
    - Current ideal length: {ideal_length.value} characters
    """)

    return length_reward, completions
