import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import h5py

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from input_output import create_h5_file  # noqa: E402


class AppVersionsTests(unittest.TestCase):
    def test_output_preserves_upstream_versions_and_adds_angioeye_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "ef_input.h5"
            output_path = tmp_path / "ae_output.h5"

            with h5py.File(input_path, "w") as h5file:
                h5file.create_dataset(
                    "app_versions",
                    data=json.dumps(
                        {
                            "HD_version": "v0.5.0",
                            "DV_version": "v1.17.2",
                            "EF_version": "v0.17.2",
                        }
                    ),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

            create_h5_file(output_path, source_file=input_path)

            expected_version = re.search(
                r'^version\s*=\s*"([^"]+)"$',
                (SRC_DIR.parent / "pyproject.toml").read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            assert expected_version is not None

            with h5py.File(output_path, "r") as h5file:
                versions = h5file["/app_versions"]
                self.assertIsInstance(versions, h5py.Dataset)
                self.assertEqual(
                    json.loads(versions[()]),
                    {
                        "HD_version": "v0.5.0",
                        "DV_version": "v1.17.2",
                        "EF_version": "v0.17.2",
                        "AE_version": expected_version.group(1),
                    },
                )
                self.assertEqual(versions.shape, ())

    def test_output_creates_version_dataset_without_upstream_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "legacy_input.h5"
            output_path = tmp_path / "ae_output.h5"
            with h5py.File(input_path, "w"):
                pass

            create_h5_file(output_path, source_file=input_path)

            with h5py.File(output_path, "r") as h5file:
                versions = h5file["/app_versions"]
                self.assertIsInstance(versions, h5py.Dataset)
                self.assertEqual(set(json.loads(versions[()])), {"AE_version"})
                self.assertEqual(versions.shape, ())


if __name__ == "__main__":
    unittest.main()
