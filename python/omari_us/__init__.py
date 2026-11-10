"""omari_us — utilities for the OMARI 'From Pixels to Predictions' workshop.

Subpackages
-----------
datasets        FETAL_PLANES_DB metadata, patient-level splits, transforms, perturbations, duplicates
models          DenseNet-161 (torchvision), ViT-B/16 (timm), USF-MAE (key-compatible MAE + checkpoint loader)
training        fine-tuning loop (frozen / full), losses, schedules, label-fraction subsets
evaluation      metrics, calibration, bootstrap (patient-cluster) CIs, McNemar, subgroup analysis
explainability  CAM, Grad-CAM, Grad-CAM++, ViT token Grad-CAM, attention rollout, faithfulness tests, method registry
export          artifact bundle writer for the website (schema v1)
"""
__version__ = "0.1.0"
