import os
import shutil
from glob import glob


def clean_project():
    """Remove generated files and directories"""
    base_dir = os.path.dirname(os.path.dirname(__file__))

    items_to_clean = [
        "model",
        "result",
        "__pycache__",
        "*/__pycache__",
        "output.csv",
        "result.csv",
        ".pytest_cache",
        "*.pyc",
        "*.pyo",
        "*.log",
    ]

    for item in items_to_clean:
        path = os.path.join(base_dir, item)

        if "*" in item:
            for file in glob(os.path.join(base_dir, "**", item.split("/")[-1]), recursive=True):
                try:
                    if os.path.isdir(file):
                        shutil.rmtree(file, ignore_errors=True)
                        print(f"Removed directory: {file}")
                    else:
                        os.remove(file)
                        print(f"Removed: {file}")
                except OSError as e:
                    print(f"Error removing {file}: {e}")
        elif os.path.isdir(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
                print(f"Removed directory: {path}")
            except OSError as e:
                print(f"Error removing directory {path}: {e}")
        elif os.path.isfile(path):
            try:
                os.remove(path)
                print(f"Removed file: {path}")
            except OSError as e:
                print(f"Error removing file {path}: {e}")

    print("\nCleanup complete!")


if __name__ == "__main__":
    confirmation = input("This will remove all generated files (models, results, cache). Continue? (y/n): ")
    if confirmation.lower() == "y":
        clean_project()
    else:
        print("Cleanup cancelled.")
