"""Standard-library safety tests; these do not load or execute checkpoints."""
import unittest
from pathlib import Path
from check_frozen_checkpoints import scan_metadata,unsafe_archive_name,validate_output


class CheckpointValidatorSafetyTest(unittest.TestCase):
    def test_archive_names(self):
        for name in ("/absolute.pt","../escape.pt","a/../../escape.pt","C:/escape.pt","a\\b.pt"):
            self.assertTrue(unsafe_archive_name(name))
        self.assertFalse(unsafe_archive_name("results/iclr_mnist/model/final.pt"))

    def test_metadata_values_not_reported(self):
        result = scan_metadata({"history":[{"source":"/home/example/private/model.pt"}],"seed":2718})
        self.assertEqual(result["private_absolute_path_fields"],["$.history[0].source"])
        self.assertNotIn("example",str(result))
        self.assertFalse(result["credential_like_key_fields"])

    def test_credential_keys(self):
        result = scan_metadata({"api_key":"placeholder","random_seed":2718})
        self.assertEqual(result["credential_like_key_fields"],["$.api_key"])
        self.assertNotIn("placeholder",str(result))

    def test_output_protection(self):
        root = Path(".").resolve()
        for name in ("results/model.pt", "data/file.json", "validation/checkpoint_validation.json", "configs/grid.json"):
            with self.assertRaises(ValueError):
                validate_output(root/name,root)
        self.assertEqual(validate_output(root/"outputs/checkpoint_validation.json",root),root/"outputs/checkpoint_validation.json")


if __name__ == "__main__":
    unittest.main()
