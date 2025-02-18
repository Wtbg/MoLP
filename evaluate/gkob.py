

from pathlib import Path
import pdb


def glob_data_files(data_dir):
    data_dir = Path(data_dir)
    data_files = sorted(list(data_dir.glob("*.pt")))
    pdb.set_trace()
    return data_files


if __name__ == "__main__":
    data_files = glob_data_files("data/processed/embeddings")
    with open("training/split/all_question.txt", "w") as f:
        for file in data_files:
            f.write(f"{file}\n")