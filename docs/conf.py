import sys
from pathlib import Path

from sphinx.config import is_serializable

project = "KS shadowing"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]
autodoc_typehints = "none"
autodoc_member_order = "bysource"
napoleon_use_rtype = False
html_theme = "furo"

# Sphinx-Gallery imports the sort key by fully qualified name, and Sphinx does
# not put the config directory on sys.path.
sys.path.insert(0, str(Path(__file__).parent))

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "remove_config_comments": True,
    "within_subsection_order": "gallery_order.example_order",
}

# Sphinx caches the config by pickling it; a raw callable here would silently
# disable that cache and force a full rebuild every time.
assert is_serializable(sphinx_gallery_conf)
