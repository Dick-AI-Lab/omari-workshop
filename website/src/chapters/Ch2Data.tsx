import { ChapterHead } from '../components/shell';
import { Ask, Code, Deep, Tex } from '../components/ui';
import { FETAL_PLANES_LITERATURE as LIT, USF_MAE_LITERATURE as USF } from '../data/literature';
import { ClassBalance, LeakageLab } from '../visualizations/DataLabs';
import { PreprocessLab } from '../visualizations/PreprocessLab';

export default function Ch2Data() {
  return (
    <section className="chapter" id="data">
      <ChapterHead id="data" />
      <p className="lede">A network never sees “an ultrasound”. It sees a 224 × 224 × 3 tensor of floating-point numbers, produced by a chain of physical and software decisions that started at the transducer.</p>

      <h3>What makes an ultrasound image</h3>
      <p>
        A B-mode image is a map of echo amplitude. Sub-resolution scatterers interfere coherently, which produces <b>speckle</b>: a granular texture that is a deterministic function of the tissue and probe, not additive sensor noise. In fully developed speckle the envelope amplitude is Rayleigh-distributed and the intensity is exponential, so the noise is <i>multiplicative</i> and its relative variance does not fall with brightness. The machine then log-compresses a dynamic range of roughly 40–80 dB into 8 bits, applies vendor-specific filtering and gain curves, and burns in text, calipers and depth markers.
      </p>
      <p>
        Axial resolution (along the beam) is set by pulse length and is usually finer than lateral resolution, which degrades with depth outside the focal zone. On top of the physics sits <b>acquisition variability</b>: probe frequency, gain, depth and zoom chosen by the operator; fetal position and gestational age; maternal habitus; and the machine itself. FETAL_PLANES_DB contains images from {LIT.machines.join(', ')} across two hospitals, and the metadata records operator and machine for every image. This is useful, because those variables can be tested as sources of shortcut learning and domain shift.
      </p>

      <PreprocessLab />

      <Deep kind="math">
        <p>Speckle intensity with <Tex>{'L'}</Tex> independent looks, and the multiplicative model used by the simulator:</p>
        <Tex display>{'I_\\text{obs}(x) = I(x)\\,\\big(1 + s\\,(n(x) - 1)\\big),\\qquad n(x) \\sim \\mathrm{Gamma}\\!\\left(L, \\tfrac{1}{L}\\right),\\quad \\mathbb{E}[n]=1,\\ \\mathrm{Var}[n] = \\tfrac{1}{L}'}</Tex>
        <p>Normalisation maps intensities to the distribution a pretrained network expects. For channel <Tex>c</Tex> with grey replicated to RGB:</p>
        <Tex display>{'\\tilde x_c = \\frac{x - \\mu_c}{\\sigma_c},\\quad \\mu = (0.485, 0.456, 0.406),\\ \\sigma = (0.229, 0.224, 0.225)\\ \\ \\text{(ImageNet)}'}</Tex>
        <p>Per-image z-scoring <Tex>{'(x-\\bar x)/s_x'}</Tex> removes global gain differences but also discards absolute echogenicity, which can be diagnostic. CLAHE equalises local histograms with a clip limit, amplifying speckle in dark regions.</p>
      </Deep>
      <Deep kind="code">
        <Code>{`from torchvision import transforms as T

# USF-MAE (official loader): stretch to 224x224, NEAREST, ImageNet statistics
usf_mae_eval = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# timm vit_base_patch16_224.augreg_in1k expects mean = std = 0.5 (check timm.data.resolve_data_config)
vit_eval = T.Compose([T.Grayscale(3), T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

# torchvision DenseNet-161 IMAGENET1K_V1: ImageNet statistics; we keep the same geometry as the ViTs
densenet_eval = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# python/omari_us/datasets/transforms.py builds these from configs/*.yaml so notebooks and export agree.`}</Code>
      </Deep>

      <Ask q="The official USF-MAE loader stretches every image to 224 × 224. FETAL_PLANES_DB images vary in size and aspect ratio. What could go wrong?"
        options={['Nothing: CNNs are scale-invariant', 'Aspect ratio becomes a class cue', 'Head circumference shapes distort', 'Padding would be better in every case']} multi
        reveal={<>Stretching changes object shape in a way that depends on the original frame size, which depends on machine and zoom. If some planes are mostly acquired at one zoom setting, the distortion itself can become predictive, a shortcut. Letterboxing preserves geometry but wastes resolution and introduces a padding border the model can also exploit. Neither is universally right. The defensible practice is to use the transform each model was pretrained with (here, {USF.preprocessing}) and to test sensitivity to the choice, which you can do in the lab above.</>} />

      <h3>Class imbalance</h3>
      <p>The six classes are far from balanced. Imbalance changes what an unweighted loss optimises: a model can reach high accuracy by doing well on “Other” and the brain while neglecting the abdomen. Weighting the loss is one remedy; we compare it with macro-averaged metrics in Chapter 4.</p>
      <ClassBalance />

      <h3>Splitting without leakage</h3>
      <p>
        Images from one patient share anatomy, gestational age, machine, operator and often consecutive frames. If those images straddle the train/test boundary, the test set stops measuring generalisation to new patients and starts measuring recognition of patients already seen. FETAL_PLANES_DB ships the authors’ patient-level split; we keep it, carve a patient-disjoint validation set from the training half, and check for near-duplicate images across splits in notebook 01.
      </p>
      <LeakageLab />
      <Deep kind="code">
        <Code>{`from sklearn.model_selection import StratifiedGroupKFold

# keep the authors' test split; make a patient-disjoint validation fold inside the training half
train = meta[meta.Train == 1]
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
tr_idx, va_idx = next(sgkf.split(train, y=train.label, groups=train.Patient_num))
assert set(train.iloc[tr_idx].Patient_num).isdisjoint(train.iloc[va_idx].Patient_num)
# full implementation: python/omari_us/datasets/fetal_planes.py::make_splits`}</Code>
      </Deep>
    </section>
  );
}
