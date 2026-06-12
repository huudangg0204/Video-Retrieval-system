"""
Desktop GUI applications for AIC-26.

Provides graphical interfaces for:
- EVA02 Image Retrieval (main search GUI)
- Object Filter (filter keyframes by detected objects)

Usage:
    from aic26.gui.retrieval_app import EVA02GUI
    app = EVA02GUI()
    app.mainloop()
"""

from aic26.gui.retrieval_app import EVA02GUI
from aic26.gui.filter_app import ObjectFilterApp

__all__ = ["EVA02GUI", "ObjectFilterApp"]