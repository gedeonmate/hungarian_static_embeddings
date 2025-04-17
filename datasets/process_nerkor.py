import os

# Define the root folder containing subfolders with .conllup files
train_root_folder = "../datasets/NYTK-NerKor/data/train"
train_output_file = "nerkor_train.conllup"

dev_root_folder = "../datasets/NYTK-NerKor/data/devel"
dev_output_file = "nerkor_val.conllup"

test_root_folder = "../datasets/NYTK-NerKor/data/test"
test_output_file = "nerkor_test.conllup"

def concatenate_conllup_files(root_folder, output_file):
    conllup_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".conllup"):
                conllup_files.append(os.path.join(root, file))
    conllup_files.sort()
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file in conllup_files:
            with open(file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read().strip() + "\n\n")
    print(f"Concatenation complete. Output saved as {output_file}")

concatenate_conllup_files("../datasets/NYTK-NerKor/data/train", "nerkor_train.conllup")
concatenate_conllup_files("../datasets/NYTK-NerKor/data/devel", "nerkor_val.conllup")
concatenate_conllup_files("../datasets/NYTK-NerKor/data/test", "nerkor_test.conllup")
