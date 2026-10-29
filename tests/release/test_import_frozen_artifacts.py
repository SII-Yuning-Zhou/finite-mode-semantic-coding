"""Small synthetic archive safety tests; never read/hash the transferred tar."""
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from neural.import_frozen_artifacts import APPROVED, MAX_WEIGHT, destination, import_archive, validate_members, write_new_or_identical


def ordinary(name,size=1):
    value=tarfile.TarInfo(name)
    value.size=size
    return value


class ImportSafetyTests(unittest.TestCase):
    def test_path_traversal(self):
        with self.assertRaises(ValueError):
            validate_members([ordinary("../outside")])

    def test_absolute_path(self):
        with self.assertRaises(ValueError):
            validate_members([ordinary("/outside")])

    def test_backslash(self):
        with self.assertRaises(ValueError):
            validate_members([ordinary("results\\outside")])

    def test_symbolic_link(self):
        value=ordinary(next(iter(APPROVED)))
        value.type=tarfile.SYMTYPE
        value.linkname="../outside"
        with self.assertRaises(ValueError):
            validate_members([value])

    def test_duplicate(self):
        value=ordinary(next(iter(APPROVED)))
        with self.assertRaises(ValueError):
            validate_members([value,value])

    def test_unapproved(self):
        with self.assertRaises(ValueError):
            validate_members([ordinary(".env")])

    def test_oversized(self):
        name=next(name for name in APPROVED if name.endswith(".pt"))
        with self.assertRaises(ValueError):
            validate_members([ordinary(name,MAX_WEIGHT+1)])

    def test_missing(self):
        with self.assertRaises(ValueError):
            validate_members([ordinary(name) for name in sorted(APPROVED)[:-1]])

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="fm-import-test-") as folder:
            root=Path(folder).resolve()
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve())
            target=root/"existing"
            target.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                write_new_or_identical(target,b"replacement")
            self.assertEqual(target.read_bytes(),b"original")

    def test_destination_escape(self):
        with tempfile.TemporaryDirectory(prefix="fm-import-test-") as folder:
            root=Path(folder).resolve()
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve())
            with self.assertRaises(ValueError):
                destination(root,"../outside")

    def test_valid_synthetic_import_and_idempotency(self):
        with tempfile.TemporaryDirectory(prefix="fm-import-test-") as folder:
            base=Path(folder).resolve()
            self.assertEqual(base.parent,Path(tempfile.gettempdir()).resolve())
            root=base/"release"
            root.mkdir()
            archive=base/"synthetic.tar.gz"
            with tarfile.open(archive,"w:gz") as output:
                for name in sorted(APPROVED):
                    data=b"synthetic weights" if name.endswith(".pt") else (b"{}" if name.endswith(".json") else b"synthetic ledger\n")
                    output.addfile(ordinary(name,len(data)),io.BytesIO(data))
            first=import_archive(archive,root)
            second=import_archive(archive,root)
            self.assertEqual(first,second)
            self.assertEqual(first["members"],26)


if __name__ == "__main__":
    unittest.main()
