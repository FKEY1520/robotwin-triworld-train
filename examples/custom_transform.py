"""Example episode transformation loaded by --transform path:function.

Keep this file in source control so transformed datasets can be reproduced.
"""


def customize_episode(episode):
    """Modify a prompt while preserving aligned images and actions.

    ``episode`` is triworld_train.converter.EpisodeData. Its public fields are
    actions (NumPy array), prompt (str), images (dict of view to image paths),
    and metadata (dict). Return the EpisodeData object after your changes.
    """
    original_prompt = episode.prompt
    episode.prompt = original_prompt.strip()
    # Replace the line above with your project-specific text transformation.
    # For example: episode.prompt = "Robot task: " + original_prompt.strip()
    episode.metadata["custom_transform"] = {
        "name": "customize_episode",
        "description": "Strip leading and trailing whitespace from prompt",
        "original_prompt": original_prompt,
    }
    return episode
