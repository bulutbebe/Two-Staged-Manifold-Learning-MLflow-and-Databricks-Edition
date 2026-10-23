from pathlib import Path

import scipy.io

# Project root is the parent of the scripts folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Filename -> expected MATLAB variable name
MAT_FILES = {
    "H_strongNL_MC.mat": "H_nom_strong_all",
    "H_test_strongNL_MC.mat": "H_nom_strong_test_all",
    "H_s_dmg1_MC.mat": "H_anom_dmg1_strong",
    "H_s_dmg2_MC.mat": "H_anom_dmg2_strong",
}

def inspect_mat_file(filename: str, expected_variable: str) -> None:
    path = DATA_DIR / filename

    print(f"\n{'=' * 65}")
    print(f"File: {filename}")

    if not path.is_file():
        print(f"ERROR: File not found: {path}")
        return

    print(f"Size: {path.stat().st_size / (1024**2):.2f} MB")

    try:
        # Reads MATLAB file metadata without loading full arrays.
        variables = scipy.io.whosmat(path)

        for name, shape, dtype in variables:
            marker = " <-- expected variable" if name == expected_variable else ""
            print(f"  {name}: shape={shape}, type={dtype}{marker}")

        names = {name for name, _, _ in variables}

        if expected_variable not in names:
            print(f"WARNING: Expected variable {expected_variable!r} was not found.")

    except (NotImplementedError, ValueError) as scipy_error:
        # MATLAB v7.3 files use HDF5 and may need h5py.
        try:
            import h5py
        except ImportError:
            print("This may be a MATLAB v7.3 file.")
            print("Install the optional reader with: python -m pip install h5py")
            print(f"Original reader message: {scipy_error}")
            return

        try:
            with h5py.File(path, "r") as file:
                print("HDF5/MATLAB v7.3 format detected.")
                print("Note: h5py may display MATLAB dimensions in reverse order.")

                def show_item(name, item):
                    if isinstance(item, h5py.Dataset):
                        marker = (
                            " <-- expected variable"
                            if name == expected_variable
                            else ""
                        )
                        print(
                            f"  {name}: shape={item.shape}, "
                            f"type={item.dtype}{marker}"
                        )

                file.visititems(show_item)

        except OSError as error:
            print(f"Could not inspect this file: {error}")

def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset directory: {DATA_DIR}")

    for filename, variable in MAT_FILES.items():
        inspect_mat_file(filename, variable)

if __name__ == "__main__":
    main()
