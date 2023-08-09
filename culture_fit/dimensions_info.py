DIMENSIONS = ["pdi", "idv", "mas", "uai", "lto", "ind"]

DIMENSIONS_INFO = {
    "pdi": {
        "name": "Power Distance",
        "question": (
            "To which extent do you accept that power should not be distributed equally amongst individuals in a"
            " society?"
        ),
        "hofstede_youtube_video_url": "https://youtu.be/DqAJclwfyCw",
    },
    "idv": {
        "name": "Individualism",
        "question": "How important is it for you to be an independent person in society?",
        "hofstede_youtube_video_url": "https://youtu.be/zQj1VPNPHlI",
    },
    "mas": {
        "name": "Masculinity",
        "question": (
            "How much are you driven by competition, achievement, and success, while being willing to forego caring for"
            " others and quality of life?"
        ),
        "hofstede_youtube_video_url": "https://youtu.be/Pyr-XKQG2CM",
    },
    "uai": {
        "name": "Uncertainty Avoidance",
        "abbreviation": "UAI",
        "question": "To which extent do you feel threatened by ambiguous or unknown situations and try to avoid them?",
        "hofstede_youtube_video_url": "https://youtu.be/fZF6LyGne7Q",
    },
    "lto": {
        "name": "Long Term Orientation",
        "question": (
            "How much are you willing to adapt and try new approaches to solving societal problems that may arise in"
            " the future?"
        ),
        "hofstede_youtube_video_url": "https://youtu.be/H8ygYIGsIQ4",
    },
    "ind": {
        "name": "Indulgence",
        "question": "To which extent is it important for you to be able to express your desires and impulses?",
        "hofstede_youtube_video_url": "https://youtu.be/V0YgGdzmFtA",
    },
}

# Add uppercase abbreviations programmatically
for dimension in DIMENSIONS:
    DIMENSIONS_INFO[dimension]["abbreviation"] = dimension.upper()

# Add long descriptions programmatically by reading in text files
for dimension in DIMENSIONS:
    with open(f"./culture_fit/dimensions_descriptions/{dimension}.txt") as f:
        description_text = f.read()

    DIMENSIONS_INFO[dimension]["description"] = description_text
