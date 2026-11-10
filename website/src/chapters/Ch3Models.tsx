import { ChapterHead } from '../components/shell';
import { Ask, Code, Deep, ModelTag, Tex } from '../components/ui';
import { USF_MAE_LITERATURE as USF } from '../data/literature';
import { ConvExplorer, DenseBlockFigure, DenseNetActivations, ReceptiveFieldFigure } from '../visualizations/CnnExplorers';
import { MaskingExplorer, PretrainFinetuneFigure } from '../visualizations/MaeExplorer';
import { ModelComparisonTable } from '../visualizations/ModelCards';
import { AttentionExplorer, PatchExplorer } from '../visualizations/VitExplorers';

export default function Ch3Models() {
  return (
    <section className="chapter" id="models">
      <ChapterHead id="models" />
      <p className="lede">All three models end the same way, with a vector of features and a linear map to six logits. They differ in how that vector is computed and in what the weights learned before they ever saw a fetal image.</p>
      <ModelComparisonTable />

      <h3 id="densenet" data-beat>3A · <ModelTag id="densenet161" />: local features, reused</h3>
      <p>
        A convolutional layer slides a small kernel <Tex>{'W \\in \\mathbb{R}^{C_\\text{out}\\times C_\\text{in}\\times k\\times k}'}</Tex> over the image. Weight sharing makes the operation <b>translation-equivariant</b>, and locality means each output depends only on a small neighbourhood. These are strong, useful assumptions for images: a femur is a femur wherever it sits in the sector.
      </p>
      <ConvExplorer />
      <Deep kind="math">
        <Tex display>{'Y_{o}(i,j) = b_o + \\sum_{c=1}^{C_\\text{in}}\\sum_{u=0}^{k-1}\\sum_{v=0}^{k-1} W_{o,c,u,v}\\; X_c\\big(s\\,i + d\\,u - p,\\; s\\,j + d\\,v - p\\big)'}</Tex>
        <p>Receptive field <Tex>r</Tex> and cumulative stride (jump) <Tex>J</Tex> after layer <Tex>l</Tex> with kernel <Tex>k_l</Tex> and stride <Tex>s_l</Tex>:</p>
        <Tex display>{'r_l = r_{l-1} + (k_l - 1)\\,J_{l-1},\\qquad J_l = J_{l-1}\\,s_l,\\qquad r_0 = J_0 = 1'}</Tex>
      </Deep>
      <p>
        Stacking layers grows the region each unit depends on. DenseNet-161 starts with a 7×7 stride-2 convolution and max-pool, then four <b>dense blocks</b> (6, 12, 36 and 24 layers) separated by transitions that halve resolution and channel count.
      </p>
      <ReceptiveFieldFigure />
      <p>
        The defining idea of DenseNet is connectivity. Instead of adding a layer’s output to its input (ResNet), each layer receives the <i>concatenation</i> of every earlier feature map in the block. Features computed early are reused directly by later layers. This encourages feature reuse and gives short gradient paths, with fewer parameters than a comparably deep ResNet.
      </p>
      <DenseBlockFigure />
      <Deep kind="math">
        <Tex display>{'\\mathbf{x}_\\ell = H_\\ell\\big([\\mathbf{x}_0, \\mathbf{x}_1, \\ldots, \\mathbf{x}_{\\ell-1}]\\big),\\qquad H_\\ell = \\mathrm{Conv}_{3\\times3}\\circ\\mathrm{ReLU}\\circ\\mathrm{BN}\\circ\\mathrm{Conv}_{1\\times1}\\circ\\mathrm{ReLU}\\circ\\mathrm{BN}'}</Tex>
        <p>With growth rate <Tex>k</Tex>, layer <Tex>\ell</Tex> of a block with input width <Tex>c_0</Tex> reads <Tex>{'c_0 + (\\ell-1)k'}</Tex> channels. The classifier is global average pooling followed by a linear layer: <Tex>{'y_c = \\sum_k w_k^c \\, \\frac{1}{Z}\\sum_{i,j} A_k(i,j) + b_c'}</Tex>. This structure is exactly what makes CAM applicable in Chapter 5.</p>
      </Deep>
      <Deep kind="code">
        <Code>{`import torch.nn as nn
from torchvision.models import densenet161, DenseNet161_Weights

model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 6)  # 2208 -> 6

# freeze everything except the head for linear probing
for n, p in model.named_parameters():
    p.requires_grad = n.startswith("classifier")

# feature maps for the website: forward hooks on each dense block
feats = {}
for name in ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]:
    getattr(model.features, name).register_forward_hook(lambda m, i, o, n=name: feats.__setitem__(n, o.detach()))`}</Code>
      </Deep>
      <DenseNetActivations />

      <h3 id="vit" data-beat>3B · <ModelTag id="vit_b16" />: global context from the first layer</h3>
      <p>
        A vision transformer drops convolution almost entirely. It cuts the image into 16×16 patches, embeds each as a token, adds a position code, and lets every token exchange information with every other token through <b>self-attention</b>, in every one of its 12 blocks. Nothing in the architecture says that neighbouring patches are related; that has to be learned from data, which is one reason ViTs need large pretraining corpora.
      </p>
      <PatchExplorer />
      <Deep kind="math">
        <Tex display>{'\\mathbf{z}_0 = \\big[\\mathbf{x}_\\text{cls};\\; \\mathbf{x}_p^1 E;\\; \\ldots;\\; \\mathbf{x}_p^{N} E\\big] + E_\\text{pos},\\qquad E \\in \\mathbb{R}^{(P^2 C)\\times D},\\ E_\\text{pos}\\in\\mathbb{R}^{(N+1)\\times D}'}</Tex>
        <Tex display>{'\\mathbf{z}\'_\\ell = \\mathrm{MSA}(\\mathrm{LN}(\\mathbf{z}_{\\ell-1})) + \\mathbf{z}_{\\ell-1},\\qquad \\mathbf{z}_\\ell = \\mathrm{MLP}(\\mathrm{LN}(\\mathbf{z}\'_\\ell)) + \\mathbf{z}\'_\\ell'}</Tex>
        <Tex display>{'\\mathrm{Attn}(Q,K,V) = \\mathrm{softmax}\\!\\Big(\\frac{QK^\\top}{\\sqrt{d_h}}\\Big)V,\\quad Q = \\mathbf{z}W_Q^{(h)},\\ K = \\mathbf{z}W_K^{(h)},\\ V = \\mathbf{z}W_V^{(h)},\\quad d_h = D/H = 64'}</Tex>
        <p>MAE’s fixed 2D sin-cos code uses half the channels for the column index and half for the row, each <Tex>{'[\\sin(p\\,\\omega_i), \\cos(p\\,\\omega_i)]'}</Tex> with <Tex>{'\\omega_i = 10000^{-i/(D/4)}'}</Tex>. Supervised ViT-B/16 learns <Tex>{'E_\\text{pos}'}</Tex> freely.</p>
      </Deep>
      <AttentionExplorer />
      <p>
        Attention maps are seductive because they look like explanations. They are not. A weight <Tex>{'A_{ij}'}</Tex> says how much token <i>j</i>’s value vector is mixed into token <i>i</i> at one layer and one head. It ignores the magnitude of those value vectors, the residual stream that bypasses attention, the MLPs, and every other layer. Different heads can attend to completely different places with similar effect on the output. Methods that account for this (attention rollout, gradient-weighted relevance) come in Chapter 5.
      </p>
      <Deep kind="code">
        <Code>{`import timm, torch

vit = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True, num_classes=6)
# timm uses fused scaled-dot-product attention by default; disable it so weights are materialised
for blk in vit.blocks:
    blk.attn.fused_attn = False
attn = {}
for i, blk in enumerate(vit.blocks):
    # attn_drop is applied to softmax(QK^T / sqrt(d)); its input is the attention matrix
    blk.attn.attn_drop.register_forward_hook(lambda m, inp, out, i=i: attn.__setitem__(i, inp[0].detach()))
logits = vit(x)            # attn[i]: (B, heads=12, 197, 197)
cls_to_patches = attn[11][:, :, 0, 1:]   # (B, 12, 196) -> reshape to 14x14`}</Code>
      </Deep>

      <h3 id="usfmae" data-beat>3C · <ModelTag id="usf_mae" />: an encoder that has seen ultrasound before</h3>
      <p>
        ImageNet pretraining teaches features for colour photographs of objects: textures, edges and shapes under natural lighting. Ultrasound has no colour, its texture is speckle, its geometry is set by the probe, and its classes differ in subtle anatomy. Ultrasound labels are also scarce and expensive. Self-supervised pretraining avoids labels by inventing a task whose answer is in the data itself.
      </p>
      <p>
        USF-MAE applies masked autoencoding (He et al., 2022) to {USF.pretrainingImages}. The encoder is a standard ViT-B/16; the decoder is used only during pretraining. To solve the task the encoder has to model anatomy, speckle statistics and acquisition geometry well enough to predict what is hidden.
      </p>
      <MaskingExplorer />
      <Deep kind="math">
        <p>With patch set <Tex>{'\\{\\mathbf{x}_i\\}_{i=1}^{N}'}</Tex>, random mask <Tex>{'\\mathcal{M}'}</Tex> with <Tex>{'|\\mathcal{M}| = rN'}</Tex>, encoder <Tex>f_\theta</Tex> on visible patches and decoder <Tex>g_\phi</Tex>:</p>
        <Tex display>{'\\mathcal{L}_\\text{MAE}(\\theta,\\phi) = \\frac{1}{|\\mathcal{M}|}\\sum_{i\\in\\mathcal{M}} \\big\\lVert g_\\phi\\big(f_\\theta(\\mathbf{x}_{\\bar{\\mathcal{M}}}),\\, \\mathbf{m}\\big)_i - \\mathbf{x}_i \\big\\rVert_2^2'}</Tex>
        <p>Encoder cost scales with the number of <i>visible</i> tokens: attention is <Tex>{'O(n^2 D)'}</Tex> and the MLP is <Tex>{'O(nD^2)'}</Tex>, so 75% masking cuts pretraining compute by about 3–4×.</p>
      </Deep>
      <PretrainFinetuneFigure />
      <Deep kind="code">
        <Code>{`from omari_us.models.usf_mae import load_usf_mae_checkpoint, mae_to_vit_classifier

ckpt = load_usf_mae_checkpoint("checkpoints/usf_mae_100ep.pth")   # verifies keys and shapes, reports decoder presence
model = mae_to_vit_classifier(ckpt, num_classes=6, global_pool="token")  # timm VisionTransformer, MAE encoder weights
# MAE fine-tuning recipe: AdamW, layer-wise lr decay 0.65-0.75, drop-path 0.1, warm-up + cosine schedule
param_groups = layer_decay_groups(model, base_lr=5e-4, layer_decay=0.65, weight_decay=0.05)`}</Code>
      </Deep>
      <div className="aside">
        <b>Data overlap to keep in mind.</b> {USF.overlap} The labels were never used, but the encoder may have seen the very test images during pretraining. This is a transductive advantage that ImageNet-initialised models do not have. It must be stated whenever USF-MAE results on this dataset are reported, and ideally tested with a pretraining run that excludes FETAL_PLANES_DB.
      </div>

      <Ask q="All three encoders produce a 768- or 2208-dimensional vector for the same image. Which statement is defensible before any training on FETAL_PLANES_DB?"
        options={['USF-MAE will win because it is domain-specific', 'DenseNet will win because CNNs suit small data', 'It depends on labels, regime and evaluation', 'ViT-B/16 and USF-MAE will match: same architecture']}
        reveal={<>Only the third. Architecture (ViT-B/16 ≡ USF-MAE encoder) fixes the function class, not the function; the weights encode what each model has learned. Domain-specific pretraining tends to help most when labels are scarce and the encoder is frozen or lightly tuned. With thousands of labels and full fine-tuning, differences often shrink. Chapter 4 measures this instead of assuming it.</>} />
    </section>
  );
}
