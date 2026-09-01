from pathlib import Path


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Prefix to add to each file name.
# Leave this as an empty string "" if no prefix should be added.
PREFIX = "chalet_"

# Text to remove from each file name.
# Leave this as an empty string "" if no text should be removed.
TEXT_TO_REMOVE = ""

# Folder containing the files to rename.
FOLDER_PATH = Path(r"C:\Users\bwindapo\polybox\Reality Capture Data\Effretikon Chalet\Exports\RTC\Cube Map Images")


# ---------------------------------------------------------------------------
# File-renaming function
# ---------------------------------------------------------------------------

def rename_files(
    folder_path: Path,
    prefix: str,
    text_to_remove: str,
) -> None:
    """Remove selected text and add a prefix to files in a folder.

    Args:
        folder_path: Path to the folder containing the files.
        prefix: Text to add to the beginning of each file name.
        text_to_remove: Text to remove from each file name.
    """
    # Check that the supplied folder exists.
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Check that the supplied path points to a folder.
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder_path}")

    # Loop through each item directly inside the folder.
    for file_path in folder_path.iterdir():
        # Skip subfolders because only files should be renamed.
        if not file_path.is_file():
            continue

        original_name = file_path.name

        # Separate the file name from its extension.
        file_stem = file_path.stem
        file_extension = file_path.suffix

        # Remove the selected text only when the parameter is not empty
        # and the text exists in the file name.
        if text_to_remove and text_to_remove in file_stem:
            file_stem = file_stem.replace(text_to_remove, "")

        # Add the prefix only when it is not empty and is not already present.
        if prefix and not file_stem.startswith(prefix):
            file_stem = f"{prefix}{file_stem}"

        # Reconstruct the full file name with its original extension.
        new_file_name = f"{file_stem}{file_extension}"
        new_file_path = file_path.with_name(new_file_name)

        # Skip the file if no changes are required.
        if new_file_name == original_name:
            print(f"No change: {original_name}")
            continue

        # Prevent an existing file from being overwritten.
        if new_file_path.exists():
            print(
                f"Skipped: {original_name}. "
                f"{new_file_name} already exists."
            )
            continue

        # Rename the original file in the same folder.
        file_path.rename(new_file_path)
        print(f"Renamed: {original_name} -> {new_file_name}")


# ---------------------------------------------------------------------------
# Run the script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rename_files(
        folder_path=FOLDER_PATH,
        prefix=PREFIX,
        text_to_remove=TEXT_TO_REMOVE,
    )