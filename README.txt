## Baseline Model

We use nnU-Net v2 (2D) for prostate MRI segmentation,
with patient-level five-fold cross-validation on RUNMC.

The experiment configuration is specified in
`baseline_config.yaml`, and Python dependencies are
listed in `requirements.txt`.

## References

- Isensee et al. (2021). nnU-Net: a self-configuring
  method for deep learning-based biomedical image
  segmentation. Nature Methods.
  https://doi.org/10.1038/s41592-020-01008-z

- SAML Project:
  https://liuquande.github.io/SAML/
