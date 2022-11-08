import json, os

def get_humain_bot(human_only=False):
    paths = {"human":[],"bot":[]}

    for typ in paths:
        if typ == "human":
            folders = ["circles_human_pc1/", "circles_human_pc2/", "circles_human_pc2_pad/", "circles_human_tel/", "circles_human_vm/", "circles_human_fast/"]
        elif typ == "bot":
            folders = ["circles_bot_pynput/", "circles_bot_gan/", "circles_bot_pyhm/", "circles_bot_naturalmousemotion/"]
            if human_only: folders = []
        for folder in folders:
            for file in os.listdir("./" + folder):
                if ".jpg" in file or not ".txt" in file: continue
                paths[typ].append("../../python/" + folder + file)

    return paths


def get_balabit():
    paths = {"training":{}, "testing":{}}

    for typ in paths:
        folder = "./Mouse-Dynamics-Challenge-master/" + ("training_files" if typ == "training" else "test_files") + "/"
        for user in os.listdir(folder):
            if user not in paths[typ]: paths[typ][user] = []
            for file in os.listdir(folder + user):
                paths[typ][user].append(folder + user + "/" + file)

    return paths


if "__main__" == __name__:
    paths = get_humain_bot(human_only=False)
    with open("sessions.json", "w") as f:
        f.write(json.dumps(paths, indent=4))

    paths = get_humain_bot(human_only=True)
    with open("sessions_human_only.json", "w") as f:
        f.write(json.dumps(paths, indent=4))