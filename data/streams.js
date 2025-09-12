// Research streams of SARDINE Lab

const RESEARCH_STREAMS = {
  "resources": {
  "keyword": "resources",
  "name": "Models & Datasets",
  "color": "#2563eb",
  "icon": "fas fa-database",
  "description": `Datasets, models, toolkits, benchmarks, and technical reports released by the lab.`
},
  "shared-task": {
  "keyword": "shared-task",
  "name": "Shared Tasks",
  "color": "#7c3aed",
  "icon": "fas fa-flag-checkered",
  "description": `Shared task submissions, findings reports, and challenge overviews (e.g., WMT, IWSLT).`
},
  "multilingual-translation": {
  "keyword": "multilingual-translation",
  "name": "Multilingual & Translation",
  "color": "#0891b2",
  "icon": "fas fa-language",
  "description": `Machine translation and multilingual/cross-lingual methods, data, and evaluation.`
},
  "multimodal": {
  "keyword": "multimodal",
  "name": "Multimodal (Speech/Vision)",
  "color": "#dc2626",
  "icon": "fas fa-images",
  "description": `Speech- and vision-language modeling: ASR/ST, V+L, audio-text, video understanding.`
},
  "interpretability": {
  "keyword": "interpretability",
  "name": "Interpretability & Analysis",
  "color": "#ea580c",
  "icon": "fas fa-search",
  "description": `Probing, explanations, saliency, analysis of model behaviors and components.`
},
  "fairness": {
  "keyword": "fairness",
  "name": "Fairness & Bias",
  "color": "#16a34a",
  "icon": "fas fa-balance-scale",
  "description": `Bias, harms, demographic disparities, and responsible evaluation of MT/QE.`
},
  "memory": {
  "keyword": "memory",
  "name": "Memory & Long Context",
  "color": "#9333ea",
  "icon": "fas fa-memory",
  "description": `Long-context models, associative memory, continuous-time memory, infinite memory.`
},
  "attention": {
  "keyword": "attention",
  "name": "Attention Mechanisms",
  "color": "#e11d48",
  "icon": "fas fa-bullseye",
  "description": `Attention mechanisms, sparsity (entmax/sparsemax), structured attention and variants.`
},
  "theory": {
  "keyword": "theory",
  "name": "Theory & Optimization",
  "color": "#0369a1",
  "icon": "fas fa-flask",
  "description": `Foundations: generalized losses, optimization, variational methods, bounds, formal analyses.`
},
  "evaluation-metrics": {
  "keyword": "evaluation-metrics",
  "name": "Evaluation & Metrics",
  "color": "#059669",
  "icon": "fas fa-chart-line",
  "description": `Automatic evaluation & QE: metrics, confidence/uncertainty, error span detection.`
},
  "retrieval": {
  "keyword": "retrieval",
  "name": "Retrieval & RAG",
  "color": "#7c2d12",
  "icon": "fas fa-magnifying-glass",
  "description": `Retrieval-augmented generation and kNN/kNN-MT, datastores, RAG pipelines.`
},
  "uncertainty": {
  "keyword": "uncertainty",
  "name": "Uncertainty & Robustness",
  "color": "#4f46e5",
  "icon": "fas fa-circle-question",
  "description": `Conformal prediction, risk control, calibration, and uncertainty quantification.`
},
  "efficiency": {
  "keyword": "efficiency",
  "name": "Efficiency & Scaling",
  "color": "#15803d",
  "icon": "fas fa-bolt",
  "description": `Efficient training/inference: kernels, pruning/quantization, caching, fast attention.`
},
  "dialogue-context": {
  "keyword": "dialogue-context",
  "name": "Dialogue & Context",
  "color": "#0e7490",
  "icon": "fas fa-comments",
  "description": `Document/chat translation, discourse phenomena, context usage and modeling.`
},
  "code-generation": {
  "keyword": "code-generation",
  "name": "Code Generation",
  "color": "#be185d",
  "icon": "fas fa-code",
  "description": `Program synthesis, execution-based decoding, debugging, and evaluation.`
}
};
if (typeof module !== 'undefined') module.exports = { streamsData };
