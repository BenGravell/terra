def pct_fmt(x: float):
    """Format float x between 0.0 and 1.0 as an integer percentage."""
    return f"{round(100*x, 2):.0f}%"
