"""soilcement: predict the unconfined strength of cement-admixed clay from its Atterberg limits and a few
trial mixes, with a worldwide corpus of 38 clay deposits as context.

Youwai S., Jongpradist P. Predicting the strength of cement-admixed clay from a few trial mixes using a
worldwide corpus of 38 clay deposits."""
__version__ = "1.0.0"
from .corpus import load_corpus, equivalent_cement, features   # noqa: F401
from .law import HierarchicalLaw                               # noqa: F401
