import json, os


def get_paths(human_only=False) -> dict:
    """
    Get a dict object that contains all paths
    to samples trajectory files. They will be
    used in the JavaScript code so paths must
    be relative to these JS file locations.

    The returned dict is something like :
    {
      "human": ["../../python/human_traj1.txt", ...],
      "bot": ["../../python/bot_traj1.txt", ...]
    }

    Parameters
    ----------
    human_only : boolean
        Default to false, if true then the "bot" list
        will be empty.

    Returns
    -------
    dict
        The dict object, you can write it as JSON file.
    """
    paths = {"human":[],"bot":[]}

    for typ in paths:
        if typ == "human":
            folders = ["circles_human_pc1/", "circles_human_pc2/",
                       "circles_human_pc2_pad/", "circles_human_tel/",
                       "circles_human_vm/", "circles_human_fast/"]
        elif typ == "bot":
            folders = ["circles_bot_pynput/", "circles_bot_gan/",
                       "circles_bot_pyhm/", "circles_bot_naturalmousemotion/"]
            if human_only: folders = []
        for folder in folders:
            for file in os.listdir("./" + folder):
                if ".jpg" in file or not ".txt" in file: continue
                paths[typ].append("../../python/" + folder + file)

    return paths


if "__main__" == __name__:
    paths = get_paths(human_only=False)
    with open("sessions.json", "w") as f:
        f.write(json.dumps(paths, indent=4))

    paths = get_paths(human_only=True)
    with open("sessions_human_only.json", "w") as f:
        f.write(json.dumps(paths, indent=4))