/**
 * Facts transcribed from publications (NOT computed here). Each value carries its source.
 * When notebook 01 exports dataset_stats.json from the real metadata CSV, the website prefers those values.
 */

export const CLASSES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other'];
export const CLASS_SHORT = ['Abdomen', 'Brain', 'Femur', 'Thorax', 'Cervix', 'Other'];

export const FETAL_PLANES_LITERATURE = {
  source: 'Burgos-Artizzu et al., Sci. Rep. 10:10200 (2020), Table 1 and Methods',
  status:
    'Transcribed from the publication. Totals (12,400 images; 1,792 patients; "Other" = 4,213; 7,129 / 5,271 train/test images; 896 / 896 patients; 2,949 fine-grained brain images) were cross-checked against the paper text. Per-class counts have not yet been re-derived from FETAL_PLANES_DB_data.csv in this environment.',
  totalImages: 12400,
  totalPatients: 1792,
  studies: 2087,
  classCounts: [711, 3092, 1040, 1718, 1626, 4213],
  brainPlanes: { 'Trans-thalamic': 1638, 'Trans-cerebellum': 714, 'Trans-ventricular': 597, 'Other / not classifiable': 143 },
  split: { trainImages: 7129, testImages: 5271, trainPatients: 896, testPatients: 896 },
  splitProtocol: 'Patients ordered chronologically by first visit; first half → training, second half → test; 10% of training held out for validation.',
  machines: ['Voluson E6 (×3)', 'Voluson S8', 'Voluson S10', 'Aloka'],
  hospitals: ['BCNatal — Hospital Clínic', 'BCNatal — Hospital Sant Joan de Déu'],
  storage: 'Original DICOM; released as uncompressed grayscale PNG with headers removed.',
  license: 'CC BY 4.0 (Zenodo record 3904280)',
  url: 'https://zenodo.org/records/3904280',
  doi: 'https://doi.org/10.1038/s41598-020-67076-5',
  paperBest: {
    model: 'DenseNet-169 (not DenseNet-161)',
    top1Error: 0.062,
    avgClassAccuracy: 0.936,
    technicianTop1Error: [0.051, 0.065],
    note: 'Reported for the authors\' own split and training protocol; not directly comparable with results produced by this workshop.',
  },
};

export const USF_MAE_LITERATURE = {
  source: 'Megahed et al., Biomedical Signal Processing and Control 122:110313 (2026); arXiv:2510.22990',
  pretrainingImages: '≈370,000 2D and 3D ultrasound images from 46 open datasets (OpenUS-46)',
  architecture: 'MAE with ViT-B/16 encoder and 8-block, 512-d decoder (mae_vit_base_patch16_dec512d8b)',
  repo: 'https://github.com/Yusufii9/USF-MAE',
  reportedDownstream: 'BUS-BRA, MMOTU-2D and GIST514-DB (F1 81.6%, 79.6%, 82.4% as reported). No FETAL_PLANES_DB downstream result is reported.',
  overlap: 'FETAL_PLANES_DB is listed as dataset 25 of OpenUS-46, i.e. its images (without labels) were part of USF-MAE pretraining.',
  initialization: 'The public pretraining notebook loads mae_pretrain_vit_base_full.pth (Facebook MAE, ImageNet-1k) before ultrasound pretraining.',
  preprocessing: 'Resize((224, 224), NEAREST) — aspect ratio not preserved; ImageNet mean/std; grayscale replicated to RGB.',
};

export const CITATIONS = [
  { key: 'burgos2020', text: 'Burgos-Artizzu X.P., Coronado-Gutiérrez D., Valenzuela-Alcaraz B., et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Scientific Reports 10, 10200 (2020).', url: 'https://doi.org/10.1038/s41598-020-67076-5' },
  { key: 'fpdb', text: 'Burgos-Artizzu X.P., et al. FETAL_PLANES_DB: Common maternal-fetal ultrasound images. Zenodo (2020), CC BY 4.0.', url: 'https://zenodo.org/records/3904280' },
  { key: 'densenet', text: 'Huang G., Liu Z., van der Maaten L., Weinberger K.Q. Densely connected convolutional networks. CVPR (2017).', url: 'https://arxiv.org/abs/1608.06993' },
  { key: 'vit', text: 'Dosovitskiy A., et al. An image is worth 16×16 words: Transformers for image recognition at scale. ICLR (2021).', url: 'https://arxiv.org/abs/2010.11929' },
  { key: 'augreg', text: 'Steiner A., et al. How to train your ViT? Data, augmentation, and regularization in vision transformers. TMLR (2022).', url: 'https://arxiv.org/abs/2106.10270' },
  { key: 'mae', text: 'He K., Chen X., Xie S., Li Y., Dollár P., Girshick R. Masked autoencoders are scalable vision learners. CVPR (2022).', url: 'https://arxiv.org/abs/2111.06377' },
  { key: 'usfmae', text: 'Megahed Y., Ducharme R., Erman A., Walker M.C., Hawken S., Chan A.D.C. USF-MAE: Ultrasound self-supervised foundation model with masked autoencoding. Biomedical Signal Processing and Control 122, 110313 (2026).', url: 'https://doi.org/10.1016/j.bspc.2026.110313' },
  { key: 'cam', text: 'Zhou B., Khosla A., Lapedriza A., Oliva A., Torralba A. Learning deep features for discriminative localization. CVPR (2016).', url: 'https://arxiv.org/abs/1512.04150' },
  { key: 'gradcam', text: 'Selvaraju R.R., et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV (2017).', url: 'https://arxiv.org/abs/1610.02391' },
  { key: 'gradcampp', text: 'Chattopadhay A., Sarkar A., Howlader P., Balasubramanian V.N. Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks. WACV (2018).', url: 'https://arxiv.org/abs/1710.11063' },
  { key: 'rollout', text: 'Abnar S., Zuidema W. Quantifying attention flow in transformers. ACL (2020).', url: 'https://arxiv.org/abs/2005.00928' },
  { key: 'chefer', text: 'Chefer H., Gur S., Wolf L. Transformer interpretability beyond attention visualization. CVPR (2021).', url: 'https://arxiv.org/abs/2012.09838' },
  { key: 'jain', text: 'Jain S., Wallace B.C. Attention is not explanation. NAACL (2019).', url: 'https://arxiv.org/abs/1902.10186' },
  { key: 'adebayo', text: 'Adebayo J., et al. Sanity checks for saliency maps. NeurIPS (2018).', url: 'https://arxiv.org/abs/1810.03292' },
  { key: 'rise', text: 'Petsiuk V., Das A., Saenko K. RISE: Randomized input sampling for explanation of black-box models. BMVC (2018) — deletion/insertion metrics.', url: 'https://arxiv.org/abs/1806.07421' },
  { key: 'guo', text: 'Guo C., Pleiss G., Sun Y., Weinberger K.Q. On calibration of modern neural networks. ICML (2017).', url: 'https://arxiv.org/abs/1706.04599' },
  { key: 'cui', text: 'Cui Y., Jia M., Lin T.-Y., Song Y., Belongie S. Class-balanced loss based on effective number of samples. CVPR (2019).', url: 'https://arxiv.org/abs/1901.05555' },
  { key: 'geirhos', text: 'Geirhos R., et al. Shortcut learning in deep neural networks. Nature Machine Intelligence 2, 665–673 (2020).', url: 'https://arxiv.org/abs/2004.07780' },
  { key: 'distill', text: 'Hohman F., Conlen M., Heer J., Chau D.H. Communicating with interactive articles. Distill (2020).', url: 'https://distill.pub/2020/communicating-with-interactive-articles/' },
];
