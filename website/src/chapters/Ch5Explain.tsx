import { ChapterHead } from '../components/shell';
import { Code, Deep, Tex } from '../components/ui';
import { CamToy, CompatibilityTable, MetaCamPanel, XaiLab } from '../visualizations/XaiLabs';

export default function Ch5Explain() {
  return (
    <section className="chapter" id="explain">
      <ChapterHead id="explain" />
      <p className="lede">An explanation method is itself a model of the model. It makes assumptions, it can be wrong, and it needs testing like anything else we put in front of a clinician.</p>
      <p>
        We want explanations for at least three different reasons: to <b>debug</b> (is the model using burned-in text?), to <b>discover</b> (which structures distinguish planes?), and to <b>support trust</b> (should a sonographer accept this label?). The class activation family answers one narrow question: which spatial positions in a feature map contribute to a class score? Everything here builds on that question; none of it answers <i>why</i> in a causal sense.
      </p>

      <h3>CAM, and what it assumes</h3>
      <p>
        Class Activation Mapping works when the network ends in convolutional feature maps <Tex>{'A_k'}</Tex>, global average pooling and a linear layer. Then the class score decomposes exactly over space: the weights that combine pooled channels can be applied to the maps themselves. DenseNet-161 has exactly this head.
      </p>
      <Tex display>{'y^c = \\sum_k w_k^c \\,\\frac{1}{Z}\\sum_{i,j} A_k(i,j) \\;=\\; \\frac{1}{Z}\\sum_{i,j}\\underbrace{\\sum_k w_k^c A_k(i,j)}_{M^c_\\text{CAM}(i,j)}'}</Tex>
      <p>
        <b>Grad-CAM</b> removes the architectural restriction by replacing <Tex>{'w_k^c'}</Tex> with the spatially averaged gradient of the class score, then keeping only positive evidence. For a GAP–linear head the two coincide up to scale. <b>Grad-CAM++</b> weights each position’s positive gradient by a factor derived from higher-order derivatives, under the assumption that the score is an exponential of the logit. It was designed to cover several instances of a class and more of each object.
      </p>
      <Tex display>{'\\alpha_k^c = \\frac{1}{Z}\\sum_{i,j}\\frac{\\partial y^c}{\\partial A_k(i,j)},\\qquad M^c_\\text{Grad-CAM} = \\mathrm{ReLU}\\Big(\\sum_k \\alpha_k^c A_k\\Big)'}</Tex>
      <Tex display>{'w_k^{c,++} = \\sum_{i,j} a^{kc}_{ij}\\,\\mathrm{ReLU}\\!\\Big(\\frac{\\partial S^c}{\\partial A_k(i,j)}\\Big),\\qquad a^{kc}_{ij} = \\frac{(\\partial_{ij} y^c)^2}{2(\\partial_{ij} y^c)^2 + \\sum_{a,b} A_k(a,b)\\,(\\partial_{ij} y^c)^3},\\quad S^c = e^{y^c}'}</Tex>
      <CamToy />
      <Deep kind="note" title="Limitations to keep in view">
        <ul>
          <li><b>Resolution.</b> Maps live at feature-map resolution (7×7 for DenseNet-161’s last block, 14×14 for ViT tokens) and are upsampled. Fine structures such as the cavum septi pellucidi cannot be resolved; earlier layers are sharper but less class-specific.</li>
          <li><b>Gradient saturation and sign.</b> ReLU discards negative evidence; saturated logits give small, noisy gradients.</li>
          <li><b>Normalisation hides magnitude.</b> Min–max scaling makes a weak, diffuse map look as confident as a strong one. The exporter stores the raw range for this reason.</li>
          <li><b>Not a test of faithfulness.</b> A map can look anatomical while the model relies on something else, and edge-like maps can survive weight randomisation (Adebayo et al., 2018). Chapter 6 tests this.</li>
        </ul>
      </Deep>
      <Deep kind="code">
        <Code>{`from omari_us.explainability import cam, gradcam, gradcampp

# DenseNet-161: target = final dense features (after norm5 + ReLU), 2208 x 7 x 7
m_cam  = cam.cam_densenet(model, x, target_class=c)                       # exact CAM, uses classifier.weight
m_gc   = gradcam.gradcam(model, x, c, target_layer=model.features.norm5, post=torch.relu)
m_gcpp = gradcampp.gradcampp(model, x, c, target_layer=model.features.norm5, post=torch.relu)
# all return (7, 7) maps; omari_us.explainability.upsample(m, (224, 224)) for overlays`}</Code>
      </Deep>

      <h3>Real attribution maps</h3>
      <XaiLab />

      <h3>Transformers need different machinery</h3>
      <p>
        A ViT has no convolutional feature maps and, in timm’s default configuration, classifies from the [CLS] token rather than an average over space. CAM therefore does not apply as written. Two adaptations are common. First, treat the 196 patch tokens of a late block as a 14×14 “feature map” with 768 channels, apply Grad-CAM to it, and reshape. This requires choosing the input to the last block’s first LayerNorm, because after the final block only [CLS] feeds the head and patch-token gradients vanish. Second, trace how information flows from patches to [CLS] through attention: <b>attention rollout</b> multiplies head-averaged attention matrices (with the residual identity) across blocks, and <b>gradient-weighted rollout</b> keeps only attention that increases the target class.
      </p>
      <Tex display>{'\\tilde A^{(\\ell)} = \\tfrac12\\big(\\bar A^{(\\ell)} + I\\big),\\qquad R = \\tilde A^{(L)}\\cdots\\tilde A^{(1)},\\qquad M_\\text{rollout} = R_{[\\text{CLS}],\\,1:N}\\ \\to\\ 14\\times 14'}</Tex>
      <Tex display>{'\\bar A^{(\\ell)}_\\text{grad} = \\mathbb{E}_h\\Big[\\mathrm{ReLU}\\Big(\\frac{\\partial y^c}{\\partial A^{(\\ell)}_h}\\odot A^{(\\ell)}_h\\Big)\\Big]'}</Tex>
      <p>
        Keep the distinction from Chapter 3 sharp. <b>Attention visualisation</b> shows routing inside the network and is the same whichever class you ask about. <b>Class-specific gradient attribution</b> asks what would change a particular score. A model can attend to the skull and still decide from the depth marker.
      </p>
      <Deep kind="code">
        <Code>{`from omari_us.explainability import vit as vitx

# reshape transform used by token Grad-CAM: drop [CLS], (B, 196, 768) -> (B, 768, 14, 14)
def tokens_to_map(t, grid=14):
    return t[:, 1:, :].reshape(t.shape[0], grid, grid, t.shape[-1]).permute(0, 3, 1, 2)

m_tok  = vitx.token_gradcam(vit, x, c, block=-1)           # hooks blocks[-1].norm1 input
m_roll = vitx.attention_rollout(vit, x, discard_ratio=0.0)
m_grad = vitx.grad_rollout(vit, x, c)`}</Code>
      </Deep>
      <CompatibilityTable />

      <h3>MetaCAM++</h3>
      <p>
        MetaCAM++ is a research extension in this family, currently under development by the presenter. Its formulation and results will be presented when available. The interface below shows how it plugs into the same pipeline, so that it can be evaluated with exactly the same images, checkpoints and faithfulness tests as the established methods.
      </p>
      <MetaCamPanel />
    </section>
  );
}
