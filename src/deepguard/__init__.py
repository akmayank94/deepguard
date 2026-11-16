"""DeepGuard v2.

One package shared by the preprocessing notebooks and the deployed app (D-018.2), so that
training and inference see identical inputs. Modules are added phase by phase:
Phase 1.2 adds video decoding, face tracking, alignment and cropping.
"""
__version__ = "2.0.0.dev0"
