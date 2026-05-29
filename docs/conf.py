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

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "remove_config_comments": True,
}
