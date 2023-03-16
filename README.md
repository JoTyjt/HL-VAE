# HL-VAE
## Abstract
Data-driven prediction of remaining useful life (RUL) has emerged as one of the most sought-after research in prognostics and health management (PHM). Nevertheless, most RUL prediction methods based on deep learning are black-box models that lack a visual interpretation to understand the RUL degradation process. To remedy the deficiency, we propose an intrinsically interpretable RUL prediction method based on three main modules: a temporal fusion separable convolutional network (TF-SCN), a hierarchical latent space variational auto-encoder (HLS-VAE), and a regressor. TF-SCN is used to extract the local feature information of the temporal signal. HLS-VAE is based on a transformer backbone that mines long-term temporal dependencies and compresses features into a hierarchical latent space. To enhance the streaming representation of the latent space, the temporal degradation information, i.e., health indicators (HI), is incorporated into the latent space in the form of inductive bias by using intermediate latent variables. The latent space can be used as a visual representation with self-interpretation to evaluate RUL degradation patterns visually. Experiments based on turbine engines show that the proposed approach achieves the same high-quality RUL prediction as black-box models while providing a latent space in which degradation rate can be captured to provide the interpretable evaluation.
## Environment
- python == 3.8
- pytorch == 1.8.1
- torch-geometric == 1.6.3
- torch-scatter == 2.0.9
- torch-sparse == 0.6.13
- torchvision == 0.9.1
- pandas == 1.4.2
- matplotlib == 3.5.1
