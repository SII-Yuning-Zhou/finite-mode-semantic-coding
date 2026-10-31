"""Fast checks for the project landing pages; no network or neural runtime."""
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DOI = "10.5281/zenodo.23774113"


class PublicDocumentationTests(unittest.TestCase):
    def test_landing_pages_use_project_names(self):
        for name in ("README.md", "README_zh-CN.md", "START_HERE_ZH.md"):
            text = (ROOT/name).read_text(encoding="utf-8")
            self.assertNotIn("iclr", text.lower(), name)
            self.assertIn(DOI, text, name)

    def test_local_links_and_images_resolve(self):
        for name in ("README.md", "README_zh-CN.md", "START_HERE_ZH.md"):
            text = (ROOT/name).read_text(encoding="utf-8")
            targets = re.findall(r"\]\(([^)]+)\)", text)
            targets += re.findall(r'(?:src|href)="([^"]+)"', text)
            for target in targets:
                address = urlsplit(target)
                if address.scheme or address.netloc or not address.path:
                    continue
                path = ROOT/unquote(address.path)
                self.assertTrue(path.is_file(), (name, target))

    def test_citation_and_dependency_notices_remain(self):
        citation = (ROOT/"CITATION.cff").read_text(encoding="utf-8")
        self.assertRegex(citation, r"(?m)^type:\s*software\s*$")
        self.assertIn(DOI, citation)
        notices = (ROOT/"THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("CompressAI", notices)
        self.assertIn("PyTorch", notices)
        self.assertIn("MNIST", notices)
        self.assertIn("CC BY 4.0", (ROOT/"README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
