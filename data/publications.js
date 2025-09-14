// Publications data for SARDINE Lab website
// Add new publications to maintain chronological order (newest first = higher id)

const publicationsData = [
  {
  "id": 152,
  "title": "Long-Context Generalization with Sparse Attention",
  "authors": "Pavlo Vasylenko, Marcos V. Treviso, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>Transformer-based architectures traditionally employ softmax to compute attention weights, which produces dense distributions over all tokens in a sequence. While effective in many settings, this density has been shown to be detrimental for tasks that demand precise focus on fixed-size patterns: as sequence length increases, non-informative tokens accumulate attention probability mass, leading to dispersion and representational collapse. We show in this paper that sparse attention mechanisms using α-entmax can avoid these issues, due to their ability to assign exact zeros to irrelevant tokens. Furthermore, we introduce Adaptive-Scalable Entmax (ASEntmax), which endows α-entmax with a learnable temperature parameter, allowing the attention distribution to interpolate between sparse (pattern-focused) and dense (softmax-like) regimes. Finally, we show that the ability to locate and generalize fixed-size patterns can be further improved through a careful design of position encodings, which impacts both dense and sparse attention methods. By integrating ASEntmax into standard transformer layers alongside proper positional encodings, we show that our models greatly outperform softmax, scalable softmax, and fixed-temperature α-entmax baselines on long-context generalization.</p>`,
  "streams": [
  "attention",
  "memory",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2506.16640",
  "bibtex": `@article{Vasylenko2025LongContextGW,
 author = {Pavlo Vasylenko and Marcos V. Treviso and Andr'e F. T. Martins},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Long-Context Generalization with Sparse Attention},
 volume = {abs/2506.16640},
 year = {2025}
}`
}
},
  {
  "id": 151,
  "title": "LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models",
  "authors": "Hugo Pitorro, Marcos V. Treviso",
  "venue": "ACL",
  "year": 2025,
  "type": "conference",
  "award": "Outstanding Paper Award",
  "abstract": `<p>State space models (SSMs), such as Mamba, have emerged as an efficient alternative to transformers for long-context sequence modeling. However, despite their growing adoption, SSMs lack the interpretability tools that have been crucial for understanding and improving attention-based architectures. While recent efforts provide insights into Mamba's internal mechanisms, they do not explicitly decompose token-wise contributions, leaving gaps in understanding how Mamba selectively processes sequences across layers. In this work, we introduce LaTIM, a novel token-level decomposition method for both Mamba-1 and Mamba-2 that enables fine-grained interpretability. We extensively evaluate our method across diverse tasks, including machine translation, copying, and retrieval-based generation, demonstrating its effectiveness in revealing Mamba's token-to-token interaction patterns.</p>`,
  "streams": [
  "interpretability",
  "memory"
],
  "links": {
  "paper": "https://aclanthology.org/2025.acl-long.1194/",
  "code": "https://github.com/deep-spin/latim",
  "bibtex": `@inproceedings{pitorro-treviso-2025-latim,
    title = "{L}a{TIM}: Measuring Latent Token-to-Token Interactions in Mamba Models",
    author = "Pitorro, Hugo  and
      Treviso, Marcos Vinicius",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1194/",
    doi = "10.18653/v1/2025.acl-long.1194",
    pages = "24478--24493",
    ISBN = "979-8-89176-251-0"
}`
}
},
  {
  "id": 150,
  "title": "AdaSplash: Adaptive Sparse Flash Attention",
  "authors": "Nuno Gon\u00e7alves, Marcos V. Treviso, Andr\u00e9 F. T. Martins",
  "venue": "ICML",
  "year": 2025,
  "type": "conference",
  "award": "Spotlight & Oral Presentation",
  "abstract": `<p>The computational cost of softmax-based attention in transformers limits their applicability to long-context tasks. Adaptive sparsity, of which $\\alpha$-entmax attention is an example, offers a flexible data-dependent alternative, but existing implementations are inefficient and do not leverage the sparsity to obtain runtime and memory gains. In this work, we propose AdaSplash, which combines the efficiency of GPU-optimized algorithms with the sparsity benefits of $\\alpha$-entmax. We first introduce a hybrid Halley-bisection algorithm, resulting in a 7-fold reduction in the number of iterations needed to compute the $\\alpha$-entmax transformation. Then, we implement custom Triton kernels to efficiently handle adaptive sparsity. Experiments with RoBERTa and ModernBERT for text classification and single-vector retrieval, along with GPT-2 for language modeling, show that our method achieves substantial improvements in runtime and memory efficiency compared to existing $\\alpha$-entmax implementations. It approaches -- and in some cases surpasses -- the efficiency of highly optimized softmax implementations like FlashAttention-2, enabling long-context training while maintaining strong task performance.</p>`,
  "streams": [
  "attention",
  "efficiency",
  "theory"
],
  "links": {
  "paper": "https://openreview.net/forum?id=OWIPDWhUcO",
  "code": "https://github.com/deep-spin/adasplash",
  "bibtex": `@inproceedings{GonalvesAdasplashAdaptiveSparseF,
  title={ AdaSplash: Adaptive Sparse Flash Attention },
  author={ Nuno Gonçalves and Marcos Treviso and Andre Martins },
  booktitle={ International Conference on Machine Learning },
  year={ 2025},
  url={ https://openreview.net/forum?id=OWIPDWhUcO }
}`
}
},
  {
  "id": 149,
  "title": "Should We Still Pretrain Encoders with Masked Language Modeling?",
  "authors": "Hippolyte Gisserot-Boukhlef, Nicolas Boizard, Manuel Faysse, Duarte M. Alves, Emmanuel Malherbe, Andr\u00e9 F. T. Martins, C\u00e9line Hudelot, Pierre Colombo",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>Learning high-quality text representations is fundamental to a wide range of NLP tasks. While encoder pretraining has traditionally relied on Masked Language Modeling (MLM), recent evidence suggests that decoder models pretrained with Causal Language Modeling (CLM) can be effectively repurposed as encoders, often surpassing traditional encoders on text representation benchmarks. However, it remains unclear whether these gains reflect an inherent advantage of the CLM objective or arise from confounding factors such as model and data scale. In this paper, we address this question through a series of large-scale, carefully controlled pretraining ablations, training a total of 38 models ranging from 210 million to 1 billion parameters, and conducting over 15,000 fine-tuning and evaluation runs. We find that while training with MLM generally yields better performance across text representation tasks, CLM-trained models are more data-efficient and demonstrate improved fine-tuning stability. Building on these findings, we experimentally show that a biphasic training strategy that sequentially applies CLM and then MLM, achieves optimal performance under a fixed computational training budget. Moreover, we demonstrate that this strategy becomes more appealing when initializing from readily available pretrained CLM models, reducing the computational burden needed to train best-in-class encoder models. We release all project artifacts at <a href="https://hf.co/MLMvsCLM">[https://hf.co/MLMvsCLM](https://hf.co/MLMvsCLM)</a> to foster further research.</p>`,
  "streams": [
  "resources",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2507.00994",
  "code": "https://huggingface.co/blog/Nicolas-BZRD/encoders-should-not-be-only-pre-trained-with-mlm",
  "bibtex": `@article{Gisserot-Boukhlef2025ShouldWS,
 author = {Hippolyte Gisserot-Boukhlef and Nicolas Boizard and Manuel Faysse and Duarte M. Alves and Emmanuel Malherbe and André Martins and C'eline Hudelot and Pierre Colombo},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Should We Still Pretrain Encoders with Masked Language Modeling?},
 volume = {abs/2507.00994},
 year = {2025}
}`
}
},
  {
  "id": 148,
  "title": "Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs",
  "authors": "Ricardo Rei, Nuno M. Guerreiro, Jos\u00e9 Pombal, Jo\u00e3o Alves, Pedro Teixeirinha, Amin Farajian, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>Fine-tuning pretrained LLMs has been shown to be an effective strategy for reaching state-of-the-art performance on specific tasks like machine translation. However, this process of adaptation often implies sacrificing general-purpose capabilities, such as conversational reasoning and instruction-following, hampering the utility of the system in real-world applications that require a mixture of skills. In this paper, we introduce Tower+, a suite of models designed to deliver strong performance across both translation and multilingual general-purpose text capabilities. We achieve a Pareto frontier between translation specialization and multilingual general-purpose capabilities by introducing a novel training recipe that builds on Tower (Alves et al., 2024), comprising continued pretraining, supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards. At each stage of training, we carefully generate and curate data to strengthen performance on translation as well as general-purpose tasks involving code generation, mathematics problem solving, and general instruction-following. We develop models at multiple scales: 2B, 9B, and 72B. Our smaller models often outperform larger general-purpose open-weight and proprietary LLMs (e.g., Llama 3.3 70B, GPT-4o). Our largest model delivers best-in-class translation performance for high-resource languages and top results in multilingual Arena Hard evaluations and in IF-MT, a benchmark we introduce for evaluating both translation and instruction-following. Our findings highlight that it is possible to rival frontier models in general capabilities, while optimizing for specific business domains, such as translation and localization.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://arxiv.org/abs/2506.17080",
  "code": "https://huggingface.co/Unbabel/Tower-Plus-9B",
  "bibtex": `@article{Rei2025TowerBG,
 author = {Ricardo Rei and Nuno M. Guerreiro and José P. Pombal and João Alves and Pedro Teixeirinha and Amin Farajian and Andr'e F. T. Martins},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs},
 volume = {abs/2506.17080},
 year = {2025}
}`
}
},
  {
  "id": 147,
  "title": "Instituto de Telecomunica\u00e7\u00f5es at IWSLT 2025: Aligning Small-Scale Speech and Language Models for Speech-to-Text Learning",
  "authors": "Giuseppe Attanasio, Sonal Sannigrahi, Ben Peters, Andr\u00e9 Filipe Torres Martins",
  "venue": "IWSLT",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>This paper presents Instituto de Telecomunicações’s submission to the IWSLT 2025 Shared Task on Instruction Following Speech Processing. We submit results for the Short Track, i.e., speech recognition, translation, and spoken question answering. Our model is a unified speech-to-text model that integrates a pretrained continuous speech encoder and text decoder through a first phase of modality alignment and a second phase of instruction fine-tuning. Crucially, we focus on using small-scale language model backbones (&lt; 2B) and restrict to high-quality, CC-BY data along with synthetic data generation to supplement existing resources.</p>`,
  "streams": [
  "multilingual-translation",
  "multimodal",
  "resources",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2025.iwslt-1.36/",
  "bibtex": `@inproceedings{attanasio-etal-2025-instituto,
    title = "Instituto de Telecomunica{\\c{c}}{\\~o}es at {IWSLT} 2025: Aligning Small-Scale Speech and Language Models for Speech-to-Text Learning",
    author = "Attanasio, Giuseppe  and
      Sannigrahi, Sonal  and
      Peters, Ben  and
      Filipe Torres Martins, Andr{\\'e}",
    editor = "Salesky, Elizabeth  and
      Federico, Marcello  and
      Anastasopoulos, Antonis",
    booktitle = "Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.iwslt-1.36/",
    doi = "10.18653/v1/2025.iwslt-1.36",
    pages = "347--353",
    ISBN = "979-8-89176-272-5"
}`
}
},
  {
  "id": 146,
  "title": "An Interdisciplinary Approach to Human-Centered Machine Translation",
  "authors": "Marine Carpuat, Omri Asscher, Kalika Bali, Luisa Bentivogli, Fr\u00e9d\u00e9ric Blain, Lynne Bowker, Monojit Choudhury, Hal Daum\u00e9 III, Kevin Duh, Ge Gao, Alvin Grissom II, Marzena Karpinska, Elaine C. Khoong, William D. Lewis, Andr\u00e9 F. T. Martins, Mary Nurminen, Douglas W. Oard, Maja Popovic, Michel Simard, Fran\u00e7ois Yvon",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>Machine Translation (MT) tools are widely used today, often in contexts where professional translators are not present. Despite progress in MT technology, a gap persists between system development and real-world usage, particularly for non-expert users who may struggle to assess translation reliability. This paper advocates for a human-centered approach to MT, emphasizing the alignment of system design with diverse communicative goals and contexts of use. We survey the literature in Translation Studies and Human-Computer Interaction to recontextualize MT evaluation and design to address the diverse real-world scenarios in which MT is used today.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://arxiv.org/abs/2506.13468",
  "bibtex": `@article{Carpuat2025AnIA,
 author = {Marine Carpuat and Omri Asscher and Kalika Bali and Luisa Bentivogli and Fr'ed'eric Blain and Lynne Bowker and M. Choudhury and Hal Daum'e and Kevin Duh and Ge Gao and Alvin Grissom and Marzena Karpinska and Elaine C. Khoong and William D. Lewis and Andr'e F. T. Martins and Mary Nurminen and Douglas W. Oard and Maja Popovic and Michel Simard and Franccois Yvon},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {An Interdisciplinary Approach to Human-Centered Machine Translation},
 volume = {abs/2506.13468},
 year = {2025}
}`
}
},
  {
  "id": 145,
  "title": "Movie Facts and Fibs (MF\u00b2): A Benchmark for Long Movie Understanding",
  "authors": "Emmanouil Zaranis, Ant\u00f3nio Farinhas, Saul Santos, Beatriz Canaverde, Miguel Moura Ramos, Aditya K Surikuchi, Andr\u00e9 Viveiros, Baohao Liao, Elena Bueno-Benito, Nithin Sivakumaran, Pavlo Vasylenko, Shoubin Yu, Sonal Sannigrahi, Wafaa Mohammed, Ben Peters, Danae S\u00e1nchez Villegas, Elias Stengel-Eskin, Giuseppe Attanasio, Jaehong Yoon, Stella Frank, Alessandro Suglia, Chrysoula Zerva, Desmond Elliott, Mariella Dimiccoli, Mohit Bansal, Oswald Lanz, Raffaella Bernardi, Raquel Fern\u00e1ndez, Sandro Pezzelle, Vlad Niculae, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>Recommender system is one of the most important information services on today’s Internet. Recently, graph neural networks have become the new state-of-the-art approach to recommender systems. In this survey, we conduct a comprehensive review of the literature on graph neural network-based recommender systems. We first introduce the background and the history of the development of both recommender systems and graph neural networks. For recommender systems, in general, there are four aspects for categorizing existing works: stage, scenario, objective, and application. For graph neural networks, the existing methods consist of two categories: spectral models and spatial ones. We then discuss the motivation of applying graph neural networks into recommender systems, mainly consisting of the high-order connectivity, the structural property of data and the enhanced supervision signal. We then systematically analyze the challenges in graph construction, embedding propagation/aggregation, model optimization, and computation efficiency. Afterward and primarily, we provide a comprehensive overview of a multitude of existing works of graph neural network-based recommender systems, following the taxonomy above. Finally, we raise discussions on the open problems and promising future directions in this area. We summarize the representative papers along with their code repositories in <a href="https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems">[https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems](https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems)</a> .</p>`,
  "streams": [
  "multimodal",
  "resources"
],
  "links": {
  "paper": "https://arxiv.org/abs/2506.06275",
  "code": "https://github.com/deep-spin/MF2",
  "bibtex": `@article{Zaranis2025MovieFA,
 author = {Emmanouil Zaranis and António Farinhas and Saul Santos and Beatriz Canaverde and Miguel Moura Ramos and Aditya K Surikuchi and Andr'e Viveiros and Baohao Liao and E. Bueno-Benito and Nithin Sivakumaran and Pavlo Vasylenko and Shoubin Yu and Sonal Sannigrahi and Wafaa Mohammed and Ben Peters and Danae S'anchez Villegas and Elias Stengel-Eskin and Giuseppe Attanasio and Jaehong Yoon and Stella Frank and Alessandro Suglia and Chrysoula Zerva and Desmond Elliott and Mariella Dimiccoli and Mohit Bansal and Oswald Lanz and Raffaella Bernardi and R. Fern'andez and Sandro Pezzelle and Vlad Niculae and Andr'e F. T. Martins},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Movie Facts and Fibs (MF2): A Benchmark for Long Movie Understanding},
 volume = {abs/2506.06275},
 year = {2025}
}`
}
},
  {
  "id": 144,
  "title": "Construction-Based Reduction of Translationese for Low-Resource Languages: A Pilot Study on Bavarian",
  "authors": "Peiqin Lin, Marion Thaler, Daniela Goschala, Amir Hossein Kargaran, Yihong Liu, Andr\u00e9 F. T. Martins, Hinrich Sch\u00fctze",
  "venue": "SIGTYP",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>When translating into a low-resource language, a language model can have a tendency to produce translations that are close to the source (e.g., word-by-word translations) due to a lack of rich low-resource training data in pretraining. Thus, the output often is translationese that differs considerably from what native speakers would produce naturally. To remedy this, we synthetically create a training set in which the frequency of a construction unique to the low-resource language is artificially inflated. For the case of Bavarian, we show that, after training, the language model has learned the unique construction and that native speakers judge its output as more natural. Our pilot study suggests that construction-based mitigation of translationese is a promising approach. Code and artifacts are available at https://github.com/cisnlp/BayernGPT.</p>`,
  "streams": [
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2025.sigtyp-1.13/",
  "bibtex": `@inproceedings{lin-etal-2025-construction,
    title = "Construction-Based Reduction of Translationese for Low-Resource Languages: A Pilot Study on {B}avarian",
    author = {Lin, Peiqin  and
      Thaler, Marion  and
      Goschala, Daniela  and
      Kargaran, Amir Hossein  and
      Liu, Yihong  and
      Martins, Andr{\\'e} F. T.  and
      Sch{\\"u}tze, Hinrich},
    editor = "Hahn, Michael  and
      Rani, Priya  and
      Kumar, Ritesh  and
      Shcherbakov, Andreas  and
      Sorokin, Alexey  and
      Serikov, Oleg  and
      Cotterell, Ryan  and
      Vylomova, Ekaterina",
    booktitle = "Proceedings of the 7th Workshop on Research in Computational Linguistic Typology and Multilingual NLP",
    month = aug,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.sigtyp-1.13/",
    doi = "10.18653/v1/2025.sigtyp-1.13",
    pages = "114--121",
    ISBN = "979-8-89176-281-7"
}`
}
},
  {
  "id": 143,
  "title": "EuroLLM-9B: Technical Report",
  "authors": "Pedro Henrique Martins, Jo\u00e3o Alves, Patrick Fernandes, Nuno M. Guerreiro, Ricardo Rei, Amin Farajian, Mateusz Klimaszewski, Duarte M. Alves, Jos\u00e9 Pombal, Nicolas Boizard, Manuel Faysse, Pierre Colombo, Fran\u00e7ois Yvon, Barry Haddow, Jos\u00e9 G. C. de Souza, Alexandra Birch, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>This report presents EuroLLM-9B, a large language model trained from scratch to support the needs of European citizens by covering all 24 official European Union languages and 11 additional languages. EuroLLM addresses the issue of European languages being underrepresented and underserved in existing open large language models. We provide a comprehensive overview of EuroLLM-9B's development, including tokenizer design, architectural specifications, data filtering, and training procedures. We describe the pre-training data collection and filtering pipeline, including the creation of EuroFilter, an AI-based multilingual filter, as well as the design of EuroBlocks-Synthetic, a novel synthetic dataset for post-training that enhances language coverage for European languages. Evaluation results demonstrate EuroLLM-9B's competitive performance on multilingual benchmarks and machine translation tasks, establishing it as the leading open European-made LLM of its size. To support open research and adoption, we release all major components of this work, including the base and instruction-tuned models, the EuroFilter classifier, and the synthetic post-training dataset.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://arxiv.org/abs/2506.04079",
  "bibtex": `@article{Martins2025EuroLLM9BTR,
 author = {P. Martins and João Alves and Patrick Fernandes and Nuno M. Guerreiro and Ricardo Rei and Amin Farajian and Mateusz Klimaszewski and Duarte M. Alves and José P. Pombal and Manuel Faysse and Pierre Colombo and Franccois Yvon and Barry Haddow and J. G. C. D. Souza and Alexandra Birch and André Martins},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {EuroLLM-9B: Technical Report},
 volume = {abs/2506.04079},
 year = {2025}
}`
}
},
  {
  "id": 142,
  "title": "Different Speech Translation Models Encode and Translate Speaker Gender Differently",
  "authors": "Dennis Fucci, Marco Gaido, Matteo Negri, Luisa Bentivogli, Andre Martins, Giuseppe Attanasio",
  "venue": "ACL",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>Recent studies on interpreting the hidden states of speech models have shown their ability to capture speaker-specific features, including gender. Does this finding also hold for speech translation (ST) models? If so, what are the implications for the speaker's gender assignment in translation? We address these questions from an interpretability perspective, using probing methods to assess gender encoding across diverse ST models. Results on three language directions (English-French/Italian/Spanish) indicate that while traditional encoder-decoder models capture gender information, newer architectures -- integrating a speech encoder with a machine translation system via adapters -- do not. We also demonstrate that low gender encoding capabilities result in systems' tendency toward a masculine default, a translation bias that is more pronounced in newer architectures.</p>`,
  "streams": [
  "fairness",
  "multimodal",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2025.acl-short.78/",
  "bibtex": `@inproceedings{fucci-etal-2025-different,
    title = "Different Speech Translation Models Encode and Translate Speaker Gender Differently",
    author = "Fucci, Dennis  and
      Gaido, Marco  and
      Negri, Matteo  and
      Bentivogli, Luisa  and
      Martins, Andre  and
      Attanasio, Giuseppe",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-short.78/",
    doi = "10.18653/v1/2025.acl-short.78",
    pages = "1005--1019",
    ISBN = "979-8-89176-252-7"
}`
}
},
  {
  "id": 141,
  "title": "Discrete Latent Structure in Neural Networks",
  "authors": "Vlad Niculae, Caio Corro, Nikita Nangia, Tsvetomila Mihaylova, Andr\u00e9 F. T. Martins",
  "venue": "FTSP",
  "year": 2025,
  "type": "book",
  "abstract": `<p>Many types of data from fields including natural language processing, computer vision, and bioinformatics, are well represented by discrete, compositional structures such as trees, sequences, or matchings. Latent structure models are a powerful tool for learning to extract such representations, offering a way to incorporate structural bias, discover insight about the data, and interpret decisions. However, effective training is challenging, as neural networks are typically designed for continuous computation. This text explores three broad strategies for learning with discrete latent structure: continuous relaxation, surrogate gradients, and probabilistic estimation. Our presentation relies on consistent notations for a wide range of models. As such, we reveal many new connections between latent structure learning strategies, showing how most consist of the same small set of fundamental building blocks, but use them differently, leading to substantially different applicability and properties.</p>`,
  "streams": [
  "interpretability",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2301.07473",
  "bibtex": `@article{Niculae2025DiscreteLS,
 author = {Vlad Niculae and Caio F. Corro and Nikita Nangia and Tsvetomila Mihaylova and André F. T. Martins},
 booktitle = {Foundations and Trends® in Signal Processing},
 journal = {Found. Trends Signal Process.},
 pages = {99-211},
 title = {Discrete Latent Structure in Neural Networks},
 volume = {19},
 year = {2025}
}`
}
},
  {
  "id": 140,
  "title": "Multilingual Contextualization of Large Language Models for Document-Level Machine Translation",
  "authors": "Miguel Moura Ramos, Patrick Fernandes, Sweta Agrawal, Andr\u00e9 F. T. Martins",
  "venue": "COLM",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>This collaboratively edited knowledgebase provides a common source of data for Wikipedia, and everyone else.</p>`,
  "streams": [
  "dialogue-context",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://openreview.net/forum?id=Ah0U1r5Ldq",
  "bibtex": `@inproceedings{ramos2025multilingual,
title={Multilingual Contextualization of Large Language Models for Document-Level Machine Translation},
author={Miguel Moura Ramos and Patrick Fernandes and Sweta Agrawal and Andre Martins},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=Ah0U1r5Ldq}
}`
}
},
  {
  "id": 139,
  "title": "Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering",
  "authors": "Patrick Fernandes, Sweta Agrawal, Emmanouil Zaranis, Andr\u00e9 F.T. Martins, Graham Neubig",
  "venue": "COLM",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more “pragmatic” approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets.</p>`,
  "streams": [
  "dialogue-context",
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://openreview.net/forum?id=Zfa9jCYGCz",
  "bibtex": `@inproceedings{fernandes2025do,
title={Do {LLM}s Understand Your Translations? Evaluating Paragraph-level {MT} with Question Answering},
author={Patrick Fernandes and Sweta Agrawal and Emmanouil Zaranis and Andre Martins and Graham Neubig},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=Zfa9jCYGCz}
}`
}
},
  {
  "id": 138,
  "title": "M-Prometheus: A Suite of Open Multilingual LLM Judges",
  "authors": "Jos\u00e9 Pombal, Dongkeun Yoon, Patrick Fernandes, Ian Wu, Seungone Kim, Ricardo Rei, Graham Neubig, Andr\u00e9 F. T. Martins",
  "venue": "COLM",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>The use of language models for automatically evaluating long-form text (LLM-as-a-judge) is becoming increasingly common, yet most LLM judges are optimized exclusively for English, with strategies for enhancing their multilingual evaluation capabilities remaining largely unexplored in the current literature. This has created a disparity in the quality of automatic evaluation methods for non-English languages, ultimately hindering the development of models with better multilingual capabilities. To bridge this gap, we introduce M-Prometheus, a suite of open-weight LLM judges ranging from 3B to 14B parameters that can provide both direct assessment and pairwise comparison feedback on multilingual outputs. M-Prometheus models outperform state-of-the-art open LLM judges on multilingual reward benchmarks spanning more than 20 languages, as well as on literary machine translation (MT) evaluation covering 4 language pairs. Furthermore, M-Prometheus models can be leveraged at decoding time to significantly improve generated outputs across all 3 tested languages, showcasing their utility for the development of better multilingual models. Lastly, through extensive ablations, we identify the key factors for obtaining an effective multilingual judge, including backbone model selection and training on natively multilingual feedback data instead of translated data. We release our models, training dataset, and code.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://openreview.net/forum?id=Atyk8lnIQQ",
  "code": "https://huggingface.co/collections/Unbabel/m-prometheus-67f3b17e6409b2550b698822",
  "bibtex": `@inproceedings{
pombal2025mprometheus,
title={M-Prometheus: A Suite of Open Multilingual {LLM} Judges},
author={Jos{\\'e} Pombal and Dongkeun Yoon and Patrick Fernandes and Ian Wu and Seungone Kim and Ricardo Rei and Graham Neubig and Andre Martins},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=Atyk8lnIQQ}
}`
}
},
  {
  "id": 137,
  "title": "Zero-shot Benchmarking: A Framework for Flexible and Scalable Automatic Evaluation of Language Models",
  "authors": "Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig",
  "venue": "COLM",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>As language models improve and grow capable of performing more complex tasks across modalities, evaluating them automatically becomes increasingly challenging. Developing strong and robust task-specific automatic metrics gets harder, and human-annotated test sets—which are expensive to create—saturate more quickly. A compelling alternative is to design reliable strategies to automate the creation of test data and evaluation, but previous attempts either rely on pre-existing data, or focus solely on individual tasks. We present Zero-shot Benchmarking (ZSB), a framework for creating high-quality benchmarks for any task by leveraging language models for both synthetic test data creation and evaluation. ZSB is simple and flexible: it requires only the creation of a prompt for data generation and one for evaluation; it is scalable to tasks and languages where collecting real-world data is costly or impractical; it is model-agnostic, allowing the creation of increasingly challenging benchmarks as models improve. To assess the effectiveness of our framework, we create benchmarks for five text-only tasks and a multi-modal one: general capabilities in four languages (English, Chinese, French, and Korean), translation, and general vision-language capabilities in English. We then rank a broad range of open and closed systems on our benchmarks. ZSB rankings consistently correlate strongly with human rankings, outperforming widely-adopted standard benchmarks. Through ablations, we find that strong benchmarks can be created with open models, and that judge model size and dataset variety are crucial drivers of performance. We release all our benchmarks, and code to reproduce our experiments and to produce new benchmarks.</p>`,
  "streams": [
  "evaluation-metrics",
  "resources"
],
  "links": {
  "paper": "https://openreview.net/forum?id=WARZwyDf17",
  "code": "https://github.com/deep-spin/zsb",
  "bibtex": `@inproceedings{
pombal2025zeroshot,
title={Zero-shot Benchmarking: A Framework for Flexible and Scalable Automatic Evaluation of Language Models},
author={Jos{\\'e} Pombal and Nuno M Guerreiro and Ricardo Rei and Andre Martins},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=WARZwyDf17}
}`
}
},
  {
  "id": 136,
  "title": "From TOWER to SPIRE: Adding the Speech Modality to a Text-Only LLM",
  "authors": "Kshitij Ambilduke, Ben Peters, Sonal Sannigrahi, Anil Keshwani, Tsz Kin Lam, Bruno Martins, Marcely Zanon Boito, Andr\u00e9 F.T. Martins",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>Large language models (LLMs) have shown remarkable performance and generalization capabilities across multiple languages and tasks, making them very attractive targets for multi-modality integration (e.g., images or speech). In this work, we extend an existing LLM to the speech modality via speech discretization and continued pre-training. In particular, we are interested in multilingual LLMs, such as TOWER, as their pre-training setting allows us to treat discretized speech input as an additional translation language. The resulting open-source model, SPIRE, is able to transcribe and translate English speech input while maintaining TOWER's original performance on translation-related tasks, showcasing that discretized speech input integration as an additional language is feasible during LLM adaptation. We make our code and models available to the community.</p>`,
  "streams": [
  "multilingual-translation",
  "multimodal",
  "resources"
],
  "links": {
  "paper": "https://arxiv.org/abs/2503.10620",
  "code": "https://huggingface.co/papers/2503.10620",
  "bibtex": `@article{ambilduke2025tower,
  title={From tower to spire: Adding the speech modality to a text-only llm},
  author={Ambilduke, Kshitij and Peters, Ben and Sannigrahi, Sonal and Keshwani, Anil and Lam, Tsz Kin and Martins, Bruno and Boito, Marcely Zanon and Martins, Andr{\\'e} FT},
  journal={arXiv preprint arXiv:2503.10620},
  year={2025}
}`
}
},
  {
  "id": 135,
  "title": "Adding Chocolate to Mint: Mitigating Metric Interference in Machine Translation",
  "authors": "Jos\u00e9 Pombal, Nuno M. Guerreiro, Ricardo Rei, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>As automatic metrics become increasingly stronger and widely adopted, the risk of unintentionally "gaming the metric" during model development rises. This issue is caused by metric interference (MINT), i.e., the use of the same or related metrics for both model tuning and evaluation. MINT can misguide practitioners into being overoptimistic about the performance of their systems: as system outputs become a function of the interfering metric, their estimated quality loses correlation with human judgments. In this work, we analyze two common cases of MINT in machine translation-related tasks: filtering of training data, and decoding with quality signals. Importantly, we find that MINT strongly distorts instance-level metric scores, even when metrics are not directly optimized for-questioning the common strategy of leveraging a different, yet related metric for evaluation that is not used for tuning. To address this problem, we propose MINTADJUST, a method for more reliable evaluation under MINT. On the WMT24 MT shared task test set, MINTADJUST ranks translations and systems more accurately than state-of-the-art metrics across a majority of language pairs, especially for high-quality systems. Furthermore, MINTADJUST outperforms AUTORANK, the ensembling method used by the organizers.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://arxiv.org/abs/2503.08327",
  "bibtex": `@article{Pombal2025AddingCT,
 author = {José P. Pombal and Nuno M. Guerreiro and Ricardo Rei and Andr'e F. T. Martins},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Adding Chocolate to Mint: Mitigating Metric Interference in Machine Translation},
 volume = {abs/2503.08327},
 year = {2025}
}`
}
},
  {
  "id": 134,
  "title": "EuroBERT: Scaling Multilingual Encoders for European Languages",
  "authors": "Nicolas Boizard, Hippolyte Gisserot-Boukhlef, Duarte M. Alves, Andr\u00e9 Martins, Ayoub Hammal, Caio Corro, C\u00e9line Hudelot, Emmanuel Malherbe, Etienne Malaboeuf, Fanny Jourdan, Gabriel Hautreux, Jo\u00e3o Alves, Kevin El-Haddad, Manuel Faysse, Maxime Peyrard, Nuno M. Guerreiro, Patrick Fernandes, Ricardo Rei, Pierre Colombo",
  "venue": "COLM",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>General-purpose multilingual vector representations, used in retrieval, regression and classification, are traditionally obtained from bidirectional encoder models. Despite their wide applicability, encoders have been recently overshadowed by advances in generative decoder-only models. However, many innovations driving this progress are not inherently tied to decoders. In this paper, we revisit the development of multilingual encoders through the lens of these advances, and introduce EuroBERT, a family of multilingual encoders covering European and widely spoken global languages. Our models outperform existing alternatives across a diverse range of tasks, spanning multilingual capabilities, mathematics, and coding, and natively supporting sequences of up to 8,192 tokens. We also examine the design decisions behind EuroBERT, offering insights into our dataset composition and training pipeline. We publicly release the EuroBERT models, including intermediate training checkpoints, together with our training framework.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://openreview.net/forum?id=jdOC24msVq",
  "code": "https://huggingface.co/papers/2503.05500",
  "bibtex": `@inproceedings{boizard2025eurobert,
title={Euro{BERT}: Scaling Multilingual Encoders for European Languages},
author={Nicolas Boizard and Hippolyte Gisserot-Boukhlef and Duarte Miguel Alves and Andre Martins and Ayoub Hammal and Caio Corro and CELINE HUDELOT and Emmanuel Malherbe and Etienne Malaboeuf and Fanny Jourdan and Gabriel Hautreux and Jo{\\~a}o Alves and Kevin El Haddad and Manuel Faysse and Maxime Peyrard and Nuno M Guerreiro and Patrick Fernandes and Ricardo Rei and Pierre Colombo},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=jdOC24msVq}
}`
}
},
  {
  "id": 133,
  "title": "LegalBench.PT: A Benchmark for Portuguese Law",
  "authors": "Beatriz Canaverde, Telmo Pessoa Pires, Leonor Melo Ribeiro, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>The recent application of LLMs to the legal field has spurred the creation of benchmarks across various jurisdictions and languages. However, no benchmark has yet been specifically designed for the Portuguese legal system. In this work, we present LegalBench.PT, the first comprehensive legal benchmark covering key areas of Portuguese law. To develop LegalBench.PT, we first collect long-form questions and answers from real law exams, and then use GPT-4o to convert them into multiple-choice, true/false, and matching formats. Once generated, the questions are filtered and processed to improve the quality of the dataset. To ensure accuracy and relevance, we validate our approach by having a legal professional review a sample of the generated questions. Although the questions are synthetically generated, we show that their basis in human-created exams and our rigorous filtering and processing methods applied result in a reliable benchmark for assessing LLMs' legal knowledge and reasoning abilities. Finally, we evaluate the performance of leading LLMs on LegalBench.PT and investigate potential biases in GPT-4o's responses. We also assess the performance of Portuguese lawyers on a sample of questions to establish a baseline for model comparison and validate the benchmark.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://arxiv.org/abs/2502.16357",
  "code": "https://huggingface.co/datasets/BeatrizCanaverde/LegalBench.PT",
  "bibtex": `@article{canaverde2025legalbench,
  title={Legalbench. pt: a benchmark for portuguese law},
  author={Canaverde, Beatriz and Pires, Telmo Pessoa and Ribeiro, Leonor Melo and Martins, Andr{\\'e} FT},
  journal={arXiv preprint arXiv:2502.16357},
  year={2025}
}`
}
},
  {
  "id": 132,
  "title": "Sparse Activations as Conformal Predictors",
  "authors": "Margarida M Campos, Jo\u00e3o C\u00e1lem, Sophia Sklaviadis, Mario A. T. Figueiredo, Andre Martins",
  "venue": "AISTATS",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>Conformal prediction is a distribution-free framework for uncertainty quantification that replaces point predictions with sets, offering marginal coverage guarantees (i.e., ensuring that the prediction sets contain the true label with a specified probability, in expectation). In this paper, we uncover a novel connection between conformal prediction and sparse softmax-like transformations, such as sparsemax and $\\gamma$-entmax (with $\\gamma &gt; 1$), which may assign nonzero probability only to a subset of labels. We introduce new non-conformity scores for classification that make the calibration process correspond to the widely used temperature scaling method. At test time, applying these sparse transformations with the calibrated temperature leads to a support set (i.e., the set of labels with nonzero probability) that automatically inherits the coverage guarantees of conformal prediction. Through experiments on computer vision and text classification benchmarks, we demonstrate that the proposed method achieves competitive results in terms of coverage, efficiency, and adaptiveness compared to standard non-conformity scores based on softmax.</p>`,
  "streams": [
  "theory",
  "uncertainty"
],
  "links": {
  "paper": "https://arxiv.org/abs/2502.14773",
  "code": "https://github.com/deep-spin/sparse-activations-cp",
  "demo": "https://sparse-activations-conformal-predictors.streamlit.app/",
  "bibtex": `@InProceedings{campos2025sparse,
  title =    {Sparse Activations as Conformal Predictors},
  author =       {Campos, Margarida M and C{\\'a}lem, Jo{\\~a}o and Sklaviadis, Sophia and Figueiredo, Mario A. T. and Martins, Andre},
  booktitle =    {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages =    {2674--2682},
  year =   {2025},
  editor =   {Li, Yingzhen and Mandt, Stephan and Agrawal, Shipra and Khan, Emtiyaz},
  volume =   {258},
  series =   {Proceedings of Machine Learning Research},
  month =    {03--05 May},
  publisher =    {PMLR},
  pdf =    {https://raw.githubusercontent.com/mlresearch/v258/main/assets/campos25a/campos25a.pdf},
  url =    {https://proceedings.mlr.press/v258/campos25a.html},
}`
}
},
  {
  "id": 131,
  "title": "Translate Smart, not Hard: Cascaded Translation Systems with Quality-Aware Deferral",
  "authors": "Ant\u00f3nio Farinhas, Nuno M. Guerreiro, Sweta Agrawal, Ricardo Rei, Andr\u00e9 F.T. Martins",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>Larger models often outperform smaller ones but come with high computational costs. Cascading offers a potential solution. By default, it uses smaller models and defers only some instances to larger, more powerful models. However, designing effective deferral rules remains a challenge. In this paper, we propose a simple yet effective approach for machine translation, using existing quality estimation (QE) metrics as deferral rules. We show that QE-based deferral allows a cascaded system to match the performance of a larger model while invoking it for a small fraction (30% to 50%) of the examples, significantly reducing computational costs. We validate this approach through both automatic and human evaluation.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://arxiv.org/abs/2502.12701",
  "bibtex": `@article{farinhas2025translate,
  title={Translate Smart, not Hard: Cascaded Translation Systems with Quality-Aware Deferral},
  author={Farinhas, Ant{\\'o}nio and Guerreiro, Nuno M and Agrawal, Sweta and Rei, Ricardo and Martins, Andr{\\'e} FT},
  journal={arXiv preprint arXiv:2502.12701},
  year={2025}
}`
}
},
  {
  "id": 130,
  "title": "Modern Hopfield Networks with Continuous-Time Memories",
  "authors": "Saul Santos, Ant\u00f3nio Farinhas, Daniel McNamee, Andr\u00e9 F. T. Martins",
  "venue": "NFAM",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>Recent research has established a connection between modern Hopfield networks (HNs) and transformer attention heads, with guarantees of exponential storage capacity. However, these models still face challenges scaling storage efficiently. Inspired by psychological theories of continuous neural resource allocation in working memory, we propose an approach that compresses large discrete Hopfield memories into smaller, continuous-time memories. Leveraging continuous attention, our new energy function modifies the update rule of HNs, replacing the traditional softmax-based probability mass function with a probability density, over the continuous memory. This formulation aligns with modern perspectives on human executive function, offering a principled link between attractor dynamics in working memory and resource-efficient memory allocation. Our framework maintains competitive performance with HNs while leveraging a compressed memory, reducing computational costs across synthetic and video datasets.</p>`,
  "streams": [
  "memory",
  "theory"
],
  "links": {
  "paper": "https://openreview.net/forum?id=bU4dyLTNp3",
  "code": "https://github.com/deep-spin/CHM-Net",
  "bibtex": `@inproceedings{santos2025modern,
title={Modern Hopfield Networks with Continuous-Time Memories},
author={Saul Santos and Ant{\\'o}nio Farinhas and Daniel C McNamee and Andre Martins},
booktitle={New Frontiers in Associative Memories},
year={2025},
url={https://openreview.net/forum?id=bU4dyLTNp3}
}`
}
},
  {
  "id": 129,
  "title": "Fenchel-Young Variational Learning",
  "authors": "Sophia Sklaviadis, Sweta Agrawal, Ant\u00f3nio Farinhas, Andr\u00e9 F. T. Martins, M\u00e1rio A. T. Figueiredo",
  "venue": "arXiv",
  "year": 2025,
  "type": "preprint",
  "abstract": `<p>From a variational perspective, many statistical learning criteria involve seeking a distribution that balances empirical risk and regularization. In this paper, we broaden this perspective by introducing a new general class of variational methods based on Fenchel-Young (FY) losses, treated as divergences that generalize (and encompass) the familiar Kullback-Leibler divergence at the core of classical variational learning. Our proposed formulation -- FY variational learning -- includes as key ingredients new notions of FY free energy, FY evidence, FY evidence lower bound, and FY posterior. We derive alternating minimization and gradient backpropagation algorithms to compute (or lower bound) the FY evidence, which enables learning a wider class of models than previous variational formulations. This leads to generalized FY variants of classical algorithms, such as an FY expectation-maximization (FYEM) algorithm, and latent-variable models, such as an FY variational autoencoder (FYVAE). Our new methods are shown to be empirically competitive, often outperforming their classical counterparts, and most importantly, to have qualitatively novel features. For example, FYEM has an adaptively sparse E-step, while the FYVAE can support models with sparse observations and sparse posteriors.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2502.10295",
  "bibtex": `@article{sklaviadis2025fenchel,
  title={Fenchel-Young Variational Learning},
  author={Sklaviadis, Sophia and Agrawal, Sweta and Farinhas, Antonio and Martins, Andre and Figueiredo, Mario},
  journal={arXiv preprint arXiv:2502.10295},
  year={2025}
}`
}
},
  {
  "id": 128,
  "title": "\u221e-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation",
  "authors": "Saul Santos, Ant\u00f3nio Farinhas, Daniel McNamee, Andr\u00e9 F. T. Martins",
  "venue": "ICML",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>Current video-language models struggle with long-video understanding due to limited context lengths and reliance on sparse frame subsampling, often leading to information loss. This paper introduces $ınfty$-Video, which can process arbitrarily long videos through a continuous-time long-term memory (LTM) consolidation mechanism. Our framework augments video Q-formers by allowing them to process unbounded video contexts efficiently and without requiring additional training. Through continuous attention, our approach dynamically allocates higher granularity to the most relevant video segments, forming "sticky" memories that evolve over time. Experiments with Video-LLaMA and VideoChat2 demonstrate improved performance in video question-answering tasks, showcasing the potential of continuous-time LTM mechanisms to enable scalable and training-free comprehension of long videos.</p>`,
  "streams": [
  "memory",
  "multimodal"
],
  "links": {
  "paper": "https://openreview.net/forum?id=afDHwQ1ZDO",
  "code": "https://github.com/deep-spin/Infinite-Video",
  "bibtex": `@inproceedings{santos2025inftyvideo,
title={\\\${\\textbackslash}infty\\$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation},
author={Saul Santos and Ant{\\'o}nio Farinhas and Daniel C McNamee and Andre Martins},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=afDHwQ1ZDO}
}`
}
},
  {
  "id": 127,
  "title": "EuroLLM: Multilingual Language Models for Europe",
  "authors": "Pedro Henrique Martins , Patrick Fernandes, Jo\u00e3o Alves, Nuno Miguel Guerreiro, Ricardo Rei, Duarte M. Alves, Jos\u00e9 Pombal, M. Amin Farajian, Manuel Faysse, Mateusz Klimaszewski, Pierre Colombo, Barry Haddow, Jos\u00e9 G. C. de Souza, Alexandra Birch, Andr\u00e9 F. T. Martins",
  "venue": "Procedia CS",
  "year": 2025,
  "type": "journal",
  "abstract": `<p>The quality of open-weight LLMs has seen significant improvement, yet they remain predominantly focused on English. In this paper, we introduce the EuroLLM project, aimed at developing a suite of open-weight multilingual LLMs capable of understanding and generating text in all official European Union languages, as well as several additional relevant languages. We outline the progress made to date, detailing our data collection and filtering process, the development of scaling laws, the creation of our multilingual tokenizer, and the data mix and modeling configurations. Additionally, we release our initial models: EuroLLM-1.7B and EuroLLM-1.7B-Instruct and report their performance on multilingual general benchmarks and machine translation.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://arxiv.org/abs/2409.16235",
  "bibtex": `@article{martins2025eurollm,
  title={Eurollm: Multilingual language models for europe},
  author={Martins, Pedro Henrique and Fernandes, Patrick and Alves, Jo{\\~a}o and Guerreiro, Nuno M and Rei, Ricardo and Alves, Duarte M and Pombal, Jos{\\'e} and Farajian, Amin and Faysse, Manuel and Klimaszewski, Mateusz and others},
  journal={Procedia Computer Science},
  volume={255},
  pages={53--62},
  year={2025},
  publisher={Elsevier}
}`
}
},
  {
  "id": 126,
  "title": "Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation",
  "authors": "Shivalika Singh, Angelika Romanou, Cl\u00e9mentine Fourrier, David Ifeoluwa Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, Raymond Ng, Shayne Longpre, Sebastian Ruder, Wei-Yin Ko, Antoine Bosselut, Alice Oh, Andre Martins, Leshem Choshen, Daphne Ippolito, Enzo Ferrante, Marzieh Fadaee, Beyza Ermis, Sara Hooker",
  "venue": "ACL",
  "year": 2025,
  "type": "conference",
  "abstract": `<p>Reliable multilingual evaluation is difficult, and culturally appropriate evaluation is even harder to achieve.A common practice to fill this gap is to machine-translate English evaluation sets. However, translation introduces language bias and carries over cultural and regional assumptions from the original questions – often testing knowledge irrelevant to the target audience. In this work, we highlight the extent and impact of these biases and present a multilingual evaluation framework that aims to mitigate them through improved translations and annotation practices.Through a large-scale study involving professional and community translators and annotators, we show that state-of-the-art models excel primarily by learning Western-centric concepts. Notably, we find that model rankings on the full MMLU change when evaluated on a subset of questions explicitly marked as culturally sensitive.We release Global MMLU, a multilingual extension of MMLU across 42 languages, featuring improved translation quality, expanded language coverage, and designated subsets labeled as culturally sensitive and culturally agnostic to enable a more comprehensive and equitable benchmark for evaluating language models across diverse linguistic and cultural contexts.</p>`,
  "streams": [
  "fairness",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2025.acl-long.919/",
  "bibtex": `@inproceedings{singh-etal-2025-global,
    title = "Global {MMLU}: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation",
    author = "Singh, Shivalika  and
      Romanou, Angelika  and
      Fourrier, Cl{\\'e}mentine  and
      Adelani, David Ifeoluwa  and
      Ngui, Jian Gang  and
      Vila-Suero, Daniel  and
      Limkonchotiwat, Peerat  and
      Marchisio, Kelly  and
      Leong, Wei Qi  and
      Susanto, Yosephine  and
      Ng, Raymond  and
      Longpre, Shayne  and
      Ruder, Sebastian  and
      Ko, Wei-Yin  and
      Bosselut, Antoine  and
      Oh, Alice  and
      Martins, Andre  and
      Choshen, Leshem  and
      Ippolito, Daphne  and
      Ferrante, Enzo  and
      Fadaee, Marzieh  and
      Ermis, Beyza  and
      Hooker, Sara",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.919/",
    doi = "10.18653/v1/2025.acl-long.919",
    pages = "18761--18799",
    ISBN = "979-8-89176-251-0",
}`
}
},
  {
  "id": 125,
  "title": "xTower: A Multilingual LLM for Explaining and Correcting Translation Errors",
  "authors": "Marcos V Treviso, Nuno M Guerreiro, Sweta Agrawal, Ricardo Rei, Jos\u00e9 Pombal, Tania Vaz, Helena Wu, Beatriz Silva, Daan Van Stigt, Andre Martins",
  "venue": "EMNLP Findings",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>While machine translation (MT) systems are achieving increasingly strong performance on benchmarks, they often produce translations with errors and anomalies. Understanding these errors can potentially help improve the translation quality and user experience. This paper introduces xTower, an open large language model (LLM) built on top of TowerBase designed to provide free-text explanations for translation errors in order to guide the generation of a corrected translation. The quality of the generated explanations by xTower are assessed via both intrinsic and extrinsic evaluation. We ask expert translators to evaluate the quality of the explanations across two dimensions: relatedness towards the error span being explained and helpfulness in error understanding and improving translation quality. Extrinsically, we test xTower across various experimental setups in generating translation corrections, demonstrating significant improvements in translation quality. Our findings highlight xTower’s potential towards not only producing plausible and helpful explanations of automatic translations, but also leveraging them to suggest corrected translations.</p>`,
  "streams": [
  "interpretability",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2024.findings-emnlp.892/",
  "code": "https://huggingface.co/sardinelab/xTower13B",
  "bibtex": `@inproceedings{treviso-etal-2024-xtower,
    title = "x{T}ower: A Multilingual {LLM} for Explaining and Correcting Translation Errors",
    author = "Treviso, Marcos V  and
      Guerreiro, Nuno M  and
      Agrawal, Sweta  and
      Rei, Ricardo  and
      Pombal, Jos{\\'e}  and
      Vaz, Tania  and
      Wu, Helena  and
      Silva, Beatriz  and
      Stigt, Daan Van  and
      Martins, Andre",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.892/",
    doi = "10.18653/v1/2024.findings-emnlp.892",
    pages = "15222--15239",
}`
}
},
  {
  "id": 124,
  "title": "How Effective are State Space Models for Machine Translation?",
  "authors": "Hugo Pitorro, Pavlo Vasylenko, Marcos Treviso, Andr\u00e9 Martins",
  "venue": "WMT",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Transformers are the current architecture of choice for NLP, but their attention layers do not scale well to long contexts. Recent works propose to replace attention with linear recurrent layers -- this is the case for state space models, which enjoy efficient training and inference. However, it remains unclear whether these models are competitive with transformers in machine translation (MT). In this paper, we provide a rigorous and comprehensive experimental comparison between transformers and linear recurrent models for MT. Concretely, we experiment with RetNet, Mamba, and hybrid versions of Mamba which incorporate attention mechanisms. Our findings demonstrate that Mamba is highly competitive with transformers on sentence and paragraph-level datasets, where in the latter both models benefit from shifting the training distribution towards longer sequences. Further analysis show that integrating attention into Mamba improves translation quality, robustness to sequence length extrapolation, and the ability to recall named entities.</p>`,
  "streams": [
  "memory",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2024.wmt-1.111/",
  "bibtex": `@inproceedings{pitorro-etal-2024-effective,
    title = "How Effective Are State Space Models for Machine Translation?",
    author = "Pitorro, Hugo  and
      Vasylenko, Pavlo  and
      Treviso, Marcos  and
      Martins, Andr{\\'e}",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Ninth Conference on Machine Translation",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wmt-1.111/",
    doi = "10.18653/v1/2024.wmt-1.111",
    pages = "1107--1124",
    abstract = "Transformers are the current architecture of choice for NLP, but their attention layers do not scale well to long contexts. Recent works propose to replace attention with linear recurrent layers - this is the case for state space models, which enjoy efficient training and inference. However, it remains unclear whether these models are competitive with transformers in machine translation (MT). In this paper, we provide a rigorous and comprehensive experimental comparison between transformers and linear recurrent models for MT. Concretely, we experiment with RetNet, Mamba, and hybrid versions of Mamba which incorporate attention mechanisms. Our findings demonstrate that Mamba is highly competitive with transformers on sentence and paragraph-level datasets, where in the latter both models benefit from shifting the training distribution towards longer sequences. Further analysis show that integrating attention into Mamba improves translation quality, robustness to sequence length extrapolation, and the ability to recall named entities."
}`
}
},
  {
  "id": 123,
  "title": "Reranking laws for language generation: A communication-theoretic perspective",
  "authors": "Ant\u00f3nio Farinhas, Haau-Sing Li, Andr\u00e9 F. T. Martins",
  "venue": "NeurIPS",
  "year": 2024,
  "type": "conference",
  "award": "Spotlight",
  "abstract": `<p>To ensure large language models (LLMs) are used safely, one must reduce their propensity to hallucinate or to generate unacceptable answers. A simple and often used strategy is to first let the LLM generate multiple hypotheses and then employ a reranker to choose the best one. In this paper, we draw a parallel between this strategy and the use of redundancy to decrease the error rate in noisy communication channels. We conceptualize the generator as a sender transmitting multiple descriptions of a message through parallel noisy channels. The receiver decodes the message by ranking the (potentially corrupted) descriptions and selecting the one found to be most reliable. We provide conditions under which this protocol is asymptotically error-free (i.e., yields an acceptable answer almost surely) even in scenarios where the reranker is imperfect (governed by Mallows or Zipf-Mandelbrot models) and the channel distributions are statistically dependent. We use our framework to obtain reranking laws which we validate empirically on two real-world tasks using LLMs: text-to-code generation with DeepSeek-Coder 7B and machine translation of medical data with TowerInstruct 13B.</p>`,
  "streams": [
  "multilingual-translation",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2409.07131",
  "bibtex": `@inproceedings{Farinhas2024reranking,
  author={António Farinhas and Haau-Sing Li and André Martins},
  title={Reranking Laws for Language Generation: A Communication-Theoretic Perspective},
  year={2024},
  cdate={1704067200000},
  url={http://papers.nips.cc/paper_files/paper/2024/hash/c8b2f897e45770595656a79a9ad91e89-Abstract-Conference.html},
  booktitle={NeurIPS},
  crossref={conf/nips/2024}
}`
}
},
  {
  "id": 122,
  "title": "QUEST: Quality-aware metropolis-hastings sampling for machine translation",
  "authors": "Gon\u00e7alo R. A. Faria, Sweta Agrawal, Ant\u00f3nio Farinhas, Ricardo Rei, Jos\u00e9 G. C. de Souza, Andr\u00e9 F. T. Martins",
  "venue": "NeurIPS",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>An important challenge in machine translation (MT) is to generate high-quality and diverse translations. Prior work has shown that the estimated likelihood from the MT model correlates poorly with translation quality. In contrast, quality evaluation metrics (such as COMET or BLEURT) exhibit high correlations with human judgments, which has motivated their use as rerankers (such as quality-aware and minimum Bayes risk decoding). However, relying on a single translation with high estimated quality increases the chances of "gaming the metric''. In this paper, we address the problem of sampling a set of high-quality and diverse translations. We provide a simple and effective way to avoid over-reliance on noisy quality estimates by using them as the energy function of a Gibbs distribution. Instead of looking for a mode in the distribution, we generate multiple samples from high-density areas through the Metropolis-Hastings algorithm, a simple Markov chain Monte Carlo approach. The results show that our proposed method leads to high-quality and diverse outputs across multiple language pairs (English$łeftrightarrow\${German, Russian}) with two strong decoder-only LLMs (Alma-7b, Tower-7b).</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "theory"
],
  "links": {
  "paper": "https://openreview.net/forum?id=dLnduWGTB4",
  "code": "https://github.com/deep-spin/quest-decoding",
  "bibtex": `@inproceedings{
faria2024quest,
title={{QUEST}: Quality-Aware Metropolis-Hastings Sampling for Machine Translation},
author={Gon{\\c{c}}alo Faria and Sweta Agrawal and Ant{\\'o}nio Farinhas and Ricardo Rei and Jos{\\'e} G. C. de Souza and Andre Martins},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=dLnduWGTB4}
}`
}
},
  {
  "id": 121,
  "title": "A Context-aware Framework for Translation-mediated Conversations",
  "authors": "Jos\u00e9 P. Pombal, Sweta Agrawal, Patrick Fernandes, Emmanouil Zaranis, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2024,
  "type": "preprint",
  "abstract": `<p>Effective communication is fundamental to any interaction, yet challenges arise when participants do not share a common language. Automatic translation systems offer a powerful solution to bridge language barriers in such scenarios, but they introduce errors that can lead to misunderstandings and conversation breakdown. A key issue is that current systems fail to incorporate the rich contextual information necessary to resolve ambiguities and omitted details, resulting in literal, inappropriate, or misaligned translations. In this work, we present a framework to improve large language model-based translation systems by incorporating contextual information in bilingual conversational settings. During training, we leverage context-augmented parallel data, which allows the model to generate translations sensitive to conversational history. During inference, we perform quality-aware decoding with context-aware metrics to select the optimal translation from a pool of candidates. We validate both components of our framework on two task-oriented domains: customer chat and user-assistant interaction. Across both settings, our framework consistently results in better translations than state-of-the-art systems like GPT-4o and TowerInstruct, as measured by multiple automatic translation quality metrics on several language pairs. We also show that the resulting model leverages context in an intended and interpretable way, improving consistency between the conveyed message and the generated translations.</p>`,
  "streams": [
  "dialogue-context",
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://arxiv.org/abs/2412.04205",
  "bibtex": `@article{pombal2024context,
  title={A context-aware framework for translation-mediated conversations},
  author={Pombal, Jos{\\'e} and Agrawal, Sweta and Fernandes, Patrick and Zaranis, Emmanouil and Martins, Andr{\\'e} FT},
  journal={arXiv preprint arXiv:2412.04205},
  year={2024}
}`
}
},
  {
  "id": 120,
  "title": "Conformal Prediction for Natural Language Processing: A Survey",
  "authors": "Margarida Campos, Ant\u00f3nio Farinhas, Chrysoula Zerva, M\u00e1rio A. T. Figueiredo, Andr\u00e9 F. T. Martins",
  "venue": "TACL",
  "year": 2024,
  "type": "journal",
  "abstract": `<p>The rapid proliferation of large language models and natural language processing (NLP) applications creates a crucial need for uncertainty quantification to mitigate risks such as Hallucinations and to enhance decision-making reliability in critical applications. Conformal prediction is emerging as a theoretically sound and practically useful framework, combining flexibility with strong statistical guarantees. Its model-agnostic and distribution-free nature makes it particularly promising to address the current shortcomings of NLP systems that stem from the absence of uncertainty quantification. This paper provides a comprehensive survey of conformal prediction techniques, their guarantees, and existing applications in NLP, pointing to directions for future research and open challenges.</p>`,
  "streams": [
  "uncertainty"
],
  "links": {
  "paper": "https://aclanthology.org/2024.tacl-1.82/",
  "bibtex": `@article{campos-etal-2024-conformal,
    title = "Conformal Prediction for Natural Language Processing: A Survey",
    author = "Campos, Margarida  and
      Farinhas, Ant{\\'o}nio  and
      Zerva, Chrysoula  and
      Figueiredo, M{\\'a}rio A. T.  and
      Martins, Andr{\\'e} F. T.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "12",
    year = "2024",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2024.tacl-1.82/",
    doi = "10.1162/tacl_a_00715",
    pages = "1497--1516"
}`
}
},
  {
  "id": 119,
  "title": "Conformalizing Machine Translation Evaluation",
  "authors": "Chrysoula Zerva, Andr\u00e9 F. T. Martins",
  "venue": "TACL",
  "year": 2024,
  "type": "journal",
  "abstract": `<p>Several uncertainty estimation methods have been recently proposed for machine translation evaluation. While these methods can provide a useful indication of when not to trust model predictions, we show in this paper that the majority of them tend to underestimate model uncertainty, and as a result, they often produce misleading confidence intervals that do not cover the ground truth. We propose as an alternative the use of conformal prediction, a distribution-free method to obtain confidence intervals with a theoretically established guarantee on coverage. First, we demonstrate that split conformal prediction can “correct” the confidence intervals of previous methods to yield a desired coverage level, and we demonstrate these findings across multiple machine translation evaluation metrics and uncertainty quantification methods. Further, we highlight biases in estimated confidence intervals, reflected in imbalanced coverage for different attributes, such as the language and the quality of translations. We address this by applying conditional conformal prediction techniques to obtain calibration subsets for each data subgroup, leading to equalized coverage. Overall, we show that, provided access to a calibration set, conformal prediction can help identify the most suitable uncertainty quantification methods and adapt the predicted confidence intervals to ensure fairness with respect to different attributes.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "uncertainty"
],
  "links": {
  "paper": "https://aclanthology.org/2024.tacl-1.80/",
  "bibtex": `@article{zerva-martins-2024-conformalizing,
    title = "Conformalizing Machine Translation Evaluation",
    author = "Zerva, Chrysoula  and
      Martins, Andr{\\'e} F. T.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "12",
    year = "2024",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2024.tacl-1.80/",
    doi = "10.1162/tacl_a_00711",
    pages = "1460--1478"
}`
}
},
  {
  "id": 118,
  "title": "Hopfield-Fenchel-Young Networks: A Unified Framework for Associative Memory Retrieval",
  "authors": "Saul Santos, Vlad Niculae, Daniel McNamee, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2024,
  "type": "preprint",
  "abstract": `<p>Associative memory models, such as Hopfield networks and their modern variants, have garnered renewed interest due to advancements in memory capacity and connections with self-attention in transformers. In this work, we introduce a unified framework-Hopfield-Fenchel-Young networks-which generalizes these models to a broader family of energy functions. Our energies are formulated as the difference between two Fenchel-Young losses: one, parameterized by a generalized entropy, defines the Hopfield scoring mechanism, while the other applies a post-transformation to the Hopfield output. By utilizing Tsallis and norm entropies, we derive end-to-end differentiable update rules that enable sparse transformations, uncovering new connections between loss margins, sparsity, and exact retrieval of single memory patterns. We further extend this framework to structured Hopfield networks using the SparseMAP transformation, allowing the retrieval of pattern associations rather than a single pattern. Our framework unifies and extends traditional and modern Hopfield networks and provides an energy minimization perspective for widely used post-transformations like ℓ2-normalization and layer normalization-all through suitable choices of Fenchel-Young losses and by using convex analysis as a building block. Finally, we validate our Hopfield-Fenchel-Young networks on diverse memory recall tasks, including free and sequential recall. Experiments on simulated data, image retrieval, multiple instance learning, and text rationalization demonstrate the effectiveness of our approach.</p>`,
  "streams": [
  "memory",
  "resources",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2411.08590",
  "bibtex": `@article{santos2024hopfield,
  title={Hopfield-fenchel-young networks: A unified framework for associative memory retrieval},
  author={Santos, Saul and Niculae, Vlad and McNamee, Daniel and Martins, Andr{\\'e} FT},
  journal={arXiv preprint arXiv:2411.08590},
  year={2024}
}`
}
},
  {
  "id": 117,
  "title": "Fine-Grained Reward Optimization for Machine Translation using Error Severity Mappings",
  "authors": "Miguel Moura Ramos, T.C. Almeida, Daniel Vareta, Filipe Azevedo, Sweta Agrawal, Patrick Fernandes, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2024,
  "type": "preprint",
  "abstract": `<p>Reinforcement learning (RL) has been proven to be an effective and robust method for training neural machine translation systems, especially when paired with powerful reward models that accurately assess translation quality. However, most research has focused on RL methods that use sentence-level feedback, leading to inefficient learning signals due to the reward sparsity problem -- the model receives a single score for the entire sentence. To address this, we propose a novel approach that leverages fine-grained, token-level quality assessments along with error severity levels using RL methods. Specifically, we use xCOMET, a state-of-the-art quality estimation system, as our token-level reward model. We conduct experiments on small and large translation datasets with standard encoder-decoder and large language models-based machine translation systems, comparing the impact of sentence-level versus fine-grained reward signals on translation quality. Our results show that training with token-level rewards improves translation quality across language pairs over baselines according to both automatic and human evaluation. Furthermore, token-level reward optimization improves training stability, evidenced by a steady increase in mean rewards over training epochs.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://arxiv.org/abs/2411.05986",
  "bibtex": `@article{ramos2024fine,
  title={Fine-grained reward optimization for machine translation using error severity mappings},
  author={Ramos, Miguel Moura and Almeida, Tom{\\'a}s and Vareta, Daniel and Azevedo, Filipe and Agrawal, Sweta and Fernandes, Patrick and Martins, Andr{\\'e} FT},
  journal={arXiv preprint arXiv:2411.05986},
  year={2024}
}`
}
},
  {
  "id": 116,
  "title": "Tower v2: Unbabel-IST 2024 Submission for the General MT Shared Task",
  "authors": "Ricardo Rei, Jose Pombal, Nuno M. Guerreiro, Jo\u00e3o Alves, Pedro Henrique Martins, Patrick Fernandes, Helena Wu, Tania Vaz, Duarte Alves, Amin Farajian, Sweta Agrawal, Antonio Farinhas, Jos\u00e9 G. C. De Souza, Andr\u00e9 Martins",
  "venue": "WMT",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>In this work, we present Tower v2, an improved iteration of the state-of-the-art open-weight Tower models, and the backbone of our submission to the WMT24 General Translation shared task. Tower v2 introduces key improvements including expanded language coverage, enhanced data quality, and increased model capacity up to 70B parameters. Our final submission combines these advancements with quality-aware decoding strategies, selecting translations based on multiple translation quality signals. The resulting system demonstrates significant improvement over previous versions, outperforming closed commercial systems like GPT-4o, Claude 3.5, and DeepL even at a smaller 7B scale.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2024.wmt-1.12/",
  "bibtex": `@inproceedings{rei-etal-2024-tower,
    title = "Tower v2: Unbabel-{IST} 2024 Submission for the General {MT} Shared Task",
    author = "Rei, Ricardo  and
      Pombal, Jose  and
      Guerreiro, Nuno M.  and
      Alves, Jo{\\~a}o  and
      Martins, Pedro Henrique  and
      Fernandes, Patrick  and
      Wu, Helena  and
      Vaz, Tania  and
      Alves, Duarte  and
      Farajian, Amin  and
      Agrawal, Sweta  and
      Farinhas, Antonio  and
      C. De Souza, Jos{\\'e} G.  and
      Martins, Andr{\\'e}",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Ninth Conference on Machine Translation",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wmt-1.12/",
    doi = "10.18653/v1/2024.wmt-1.12",
    pages = "185--204"
}`
}
},
  {
  "id": 115,
  "title": "Improving Context Usage for Translating Bilingual Customer Support Chat with Large Language Models",
  "authors": "Jose Pombal, Sweta Agrawal, Andr\u00e9 Martins",
  "venue": "WMT",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>This paper describes Unbabel+IT’s submission to the Chat Shared Task held at the Workshop of Machine Translation 2024. The task focuses on translating customer support chats between agents and customers communicating in different languages. We present two strategies for adapting state-of-the-art language models to better utilize contextual information when translating such conversations. Our training strategy involves finetuning the model on chat datasets with context-augmented instructions, resulting in a specialized model, TOWERCHAT. For inference, we propose a novel quality-aware decoding approach that leverages a context-aware metric, CONTEXTCOMET, to select the optimal translation from a pool of candidates. We evaluate our proposed approach on the official shared task datasets for ten language pairs, showing that our submission consistently outperforms baselines on all and competing systems on 8 out of 10 language pairs across multiple automated metrics. Remarkably, TOWERCHAT outperforms our contrastive submission based on the much larger TOWER-V2-70B model while being 10× smaller. According to human evaluation, our system outperforms all other systems and baselines across all language pairs. These results underscore the importance of context-aware training and inference in handling complex bilingual dialogues.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2024.wmt-1.100/",
  "bibtex": `@inproceedings{pombal-etal-2024-improving,
    title = "Improving Context Usage for Translating Bilingual Customer Support Chat with Large Language Models",
    author = "Pombal, Jose  and
      Agrawal, Sweta  and
      Martins, Andr{\\'e}",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Ninth Conference on Machine Translation",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wmt-1.100/",
    doi = "10.18653/v1/2024.wmt-1.100",
    pages = "993--1003"
}`
}
},
  {
  "id": 114,
  "title": "Analyzing Context Contributions in LLM-based Machine Translation",
  "authors": "Emmanouil Zaranis, Nuno M Guerreiro, Andre Martins",
  "venue": "ACL Findings",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Large language models (LLMs) have achieved state-of-the-art performance in machine translation (MT) and demonstrated the ability to leverage in-context learning through few-shot examples. However, the mechanisms by which LLMs use different parts of the input context remain largely unexplored. In this work, we provide a comprehensive analysis of context utilization in MT, studying how LLMs use various context parts, such as few-shot examples and the source text, when generating translations. We highlight several key findings: (1) the source part of few-shot examples appears to contribute more than its corresponding targets, irrespective of translation direction; (2) finetuning LLMs with parallel data alters the contribution patterns of different context parts; and (3) there is a positional bias where earlier few-shot examples have higher contributions to the translated sequence. Finally, we demonstrate that inspecting anomalous context contributions can potentially uncover pathological translations, such as hallucinations. Our findings shed light on the internal workings of LLM-based MT which go beyond those known for standard encoder-decoder MT models.</p>`,
  "streams": [
  "dialogue-context",
  "interpretability",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2024.findings-emnlp.876/",
  "bibtex": `@inproceedings{zaranis-etal-2024-analyzing,
    title = "Analyzing Context Contributions in {LLM}-based Machine Translation",
    author = "Zaranis, Emmanouil  and
      Guerreiro, Nuno M  and
      Martins, Andre",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.876/",
    doi = "10.18653/v1/2024.findings-emnlp.876",
    pages = "14899--14924",
    abstract = "Large language models (LLMs) have achieved state-of-the-art performance in machine translation (MT) and demonstrated the ability to leverage in-context learning through few-shot examples. However, the mechanisms by which LLMs use different parts of the input context remain largely unexplored. In this work, we provide a comprehensive analysis of context utilization in MT, studying how LLMs use various context parts, such as few-shot examples and the source text, when generating translations. We highlight several key findings: (1) the source part of few-shot examples appears to contribute more than its corresponding targets, irrespective of translation direction; (2) finetuning LLMs with parallel data alters the contribution patterns of different context parts; and (3) there is a positional bias where earlier few-shot examples have higher contributions to the translated sequence. Finally, we demonstrate that inspecting anomalous context contributions can potentially uncover pathological translations, such as hallucinations. Our findings shed light on the internal workings of LLM-based MT which go beyond those known for standard encoder-decoder MT models."
}`
}
},
  {
  "id": 113,
  "title": "Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation",
  "authors": "Emmanouil Zaranis, Giuseppe Attanasio, Sweta Agrawal, Andre Martins",
  "venue": "ACL",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Quality estimation (QE)—the automatic assessment of translation quality—has recently become crucial across several stages of the translation pipeline, from data curation to training and decoding. While QE metrics have been optimized to align with human judgments, whether they encode social biases has been largely overlooked. Biased QE risks favoring certain demographic groups over others, e.g., by exacerbating gaps in visibility and usability. This paper defines and investigates gender bias of QE metrics and discusses its downstream implications for machine translation (MT). Experiments with state-of-the-art QE metrics across multiple domains, datasets, and languages reveal significant bias. When a human entity’s gender in the source is undisclosed, masculine-inflected translations score higher than feminine-inflected ones, and gender-neutral translations are penalized. Even when contextual cues disambiguate gender, using context-aware QE metrics leads to more errors in selecting the correct translation inflection for feminine referents than for masculine ones. Moreover, a biased QE metric affects data filtering and quality-aware decoding. Our findings underscore the need for a renewed focus on developing and evaluating QE metrics centered on gender.</p>`,
  "streams": [
  "evaluation-metrics",
  "fairness",
  "interpretability",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2025.acl-long.1228/",
  "bibtex": `@inproceedings{zaranis-etal-2025-watching,
    title = "Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation",
    author = "Zaranis, Emmanouil  and
      Attanasio, Giuseppe  and
      Agrawal, Sweta  and
      Martins, Andre",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1228/",
    doi = "10.18653/v1/2025.acl-long.1228",
    pages = "25261--25284",
    ISBN = "979-8-89176-251-0"
}`
}
},
  {
  "id": 112,
  "title": "Modeling User Preferences with Automatic Metrics: Creating a High-Quality Preference Dataset for Machine Translation",
  "authors": "Sweta Agrawal, Jos\u00e9 G. C. De Souza, Ricardo Rei, Ant\u00f3nio Farinhas, Gon\u00e7alo Faria, Patrick Fernandes, Nuno M Guerreiro, Andre Martins",
  "venue": "EMNLP",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Alignment with human preferences is an important step in developing accurate and safe large language models. This is no exception in machine translation (MT), where better handling of language nuances and context-specific variations leads to improved quality. However, preference data based on human feedback can be very expensive to obtain and curate at a large scale. Automatic metrics, on the other hand, can induce preferences, but they might not match human expectations perfectly. In this paper, we propose an approach that leverages the best of both worlds. We first collect sentence-level quality assessments from professional linguists on translations generated by multiple high-quality MT systems and evaluate the ability of current automatic metrics to recover these preferences. We then use this analysis to curate a new dataset, MT-Pref (metric induced translation preference) dataset, which comprises 18k instances covering 18 language directions, using texts sourced from multiple domains post-2022. We show that aligning TOWER models on MT-Pref significantly improves translation quality on WMT23 and FLORES benchmarks.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2024.emnlp-main.803/",
  "bibtex": `@inproceedings{agrawal-etal-2024-modeling,
    title = "Modeling User Preferences with Automatic Metrics: Creating a High-Quality Preference Dataset for Machine Translation",
    author = "Agrawal, Sweta  and
      De Souza, Jos{\\'e} G. C.  and
      Rei, Ricardo  and
      Farinhas, Ant{\\'o}nio  and
      Faria, Gon{\\c{c}}alo  and
      Fernandes, Patrick  and
      Guerreiro, Nuno M  and
      Martins, Andre",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.803/",
    doi = "10.18653/v1/2024.emnlp-main.803",
    pages = "14503--14519",
}`
}
},
  {
  "id": 111,
  "title": "Assessing the Role of Context in Chat Translation Evaluation: Is Context Helpful and Under What Conditions?",
  "authors": "Sweta Agrawal, Amin Farajian, Patrick Fernandes, Ricardo Rei, Andr\u00e9 F. T. Martins",
  "venue": "TACL",
  "year": 2024,
  "type": "journal",
  "abstract": `<p>Despite the recent success of automatic metrics for assessing translation quality, their application in evaluating the quality of machine-translated chats has been limited. Unlike more structured texts like news, chat conversations are often unstructured, short, and heavily reliant on contextual information. This poses questions about the reliability of existing sentence-level metrics in this domain as well as the role of context in assessing the translation quality. Motivated by this, we conduct a meta-evaluation of existing automatic metrics, primarily designed for structured domains such as news, to assess the quality of machine-translated chats. We find that reference-free metrics lag behind reference-based ones, especially when evaluating translation quality in out-of-English settings. We then investigate how incorporating conversational contextual information in these metrics for sentence-level evaluation affects their performance. Our findings show that augmenting neural learned metrics with contextual information helps improve correlation with human judgments in the reference-free scenario and when evaluating translations in out-of-English settings. Finally, we propose a new evaluation metric, Context-MQM, that utilizes bilingual context with a large language model (LLM) and further validate that adding context helps even for LLM-based evaluation metrics.</p>`,
  "streams": [
  "dialogue-context",
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2024.tacl-1.69/",
  "bibtex": `@article{agrawal-etal-2024-assessing,
    title = "Assessing the Role of Context in Chat Translation Evaluation: Is Context Helpful and Under What Conditions?",
    author = "Agrawal, Sweta  and
      Farajian, Amin  and
      Fernandes, Patrick  and
      Rei, Ricardo  and
      Martins, Andr{\\'e} F. T.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "12",
    year = "2024",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2024.tacl-1.69/",
    doi = "10.1162/tacl_a_00700",
    pages = "1250--1267"
}`
}
},
  {
  "id": 110,
  "title": "xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection",
  "authors": "Ricardo Rei, Ricardo Rei, Daan van Stigt, Lu\u00edsa Coheur, Pierre Colombo, Andr\u00e9 F. T. Martins",
  "venue": "TACL",
  "year": 2024,
  "type": "journal",
  "abstract": `<p>Widely used learned metrics for machine translation evaluation, such as Comet and Bleurt, estimate the quality of a translation hypothesis by providing a single sentence-level score. As such, they offer little insight into translation errors (e.g., what are the errors and what is their severity). On the other hand, generative large language models (LLMs) are amplifying the adoption of more granular strategies to evaluation, attempting to detail and categorize translation errors. In this work, we introduce xcomet, an open-source learned metric designed to bridge the gap between these approaches. xcomet integrates both sentence-level evaluation and error span detection capabilities, exhibiting state-of-the-art performance across all types of evaluation (sentence-level, system-level, and error span detection). Moreover, it does so while highlighting and categorizing error spans, thus enriching the quality assessment. We also provide a robustness analysis with stress tests, and show that xcomet is largely capable of identifying localized critical errors and hallucinations.</p>`,
  "streams": [
  "evaluation-metrics",
  "interpretability",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2024.tacl-1.54/",
  "bibtex": `@article{guerreiro-etal-2024-xcomet,
    title = "xcomet: Transparent Machine Translation Evaluation through Fine-grained Error Detection",
    author = "Guerreiro, Nuno M.  and
      Rei, Ricardo  and
      Stigt, Daan van  and
      Coheur, Luisa  and
      Colombo, Pierre  and
      Martins, Andr{\\'e} F. T.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "12",
    year = "2024",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2024.tacl-1.54/",
    doi = "10.1162/tacl_a_00683",
    pages = "979--995"
}`
}
},
  {
  "id": 109,
  "title": "DOCE: Finding the Sweet Spot for Execution-Based Code Generation",
  "authors": "Haau-Sing Li, Patrick Fernandes, Iryna Gurevych, Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2024,
  "type": "preprint",
  "abstract": `<p>Recently, a diverse set of decoding and reranking procedures have been shown effective for LLM-based code generation. However, a comprehensive framework that links and experimentally compares these methods is missing. We address this by proposing Decoding Objectives for Code Execution, a comprehensive framework that includes candidate generation, n-best reranking, minimum Bayes risk (MBR) decoding, and self-debugging as the core components. We then study the contributions of these components through execution-based evaluation metrics. Our findings highlight the importance of execution-based methods and the difference gap between execution-based and execution-free methods. Furthermore, we assess the impact of filtering based on trial unit tests, a simple and effective strategy that has been often overlooked in prior works. We also propose self-debugging on multiple candidates, obtaining state-of-the-art performance on reranking for code generation. We expect our framework to provide a solid guideline for future research on code generation.</p>`,
  "streams": [
  "code-generation"
],
  "links": {
  "paper": "https://arxiv.org/abs/2408.13745",
  "code": "https://github.com/deep-spin/doce",
  "bibtex": `@article{li2024doce,
  title={Doce: Finding the sweet spot for execution-based code generation},
  author={Li, Haau-Sing and Fernandes, Patrick and Gurevych, Iryna and Martins, Andr{\\'e} FT},
  journal={arXiv preprint arXiv:2408.13745},
  year={2024}
}`
}
},
  {
  "id": 108,
  "title": "A Recipe of Parallel Corpora Exploitation for Multilingual Large Language Models",
  "authors": "Peiqin Lin, Andre Martins, Hinrich Schuetze",
  "venue": "NAACL Findings",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Recent studies have highlighted the potential of exploiting parallel corpora to enhance multilingual large language models, improving performance in both bilingual tasks, e.g., machine translation, and general-purpose tasks, e.g., text classification. Building upon these findings, our comprehensive study aims to identify the most effective strategies for leveraging parallel corpora. We investigate the impact of parallel corpora quality and quantity, training objectives, and model size on the performance of multilingual large language models enhanced with parallel corpora across diverse languages and tasks. Our analysis reveals several key insights: (i) filtering noisy translations is essential for effectively exploiting parallel corpora, while language identification and short sentence filtering have little effect; (ii) even a corpus with just 10K parallel sentences can yield results comparable to those obtained from much larger datasets; (iii) employing only the machine translation objective yields the best results among various training objectives and their combinations; (iv) larger multilingual language models benefit more from parallel corpora than smaller models. Our study offers valuable insights into the optimal utilization of parallel corpora to enhance multilingual large language models, extending the generalizability of previous findings from limited languages and tasks to a broader range of scenarios.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2025.findings-naacl.225/",
  "bibtex": `@inproceedings{lin-etal-2025-recipe,
    title = "A Recipe of Parallel Corpora Exploitation for Multilingual Large Language Models",
    author = "Lin, Peiqin  and
      Martins, Andre  and
      Schuetze, Hinrich",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.225/",
    doi = "10.18653/v1/2025.findings-naacl.225",
    pages = "4038--4050",
    ISBN = "979-8-89176-195-7",
}`
}
},
  {
  "id": 107,
  "title": "LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks",
  "authors": "Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fern\u00e1ndez, Albert Gatt, Esam Ghaleb, Mario Giulianelli, Michael Hanna, Alexander Koller, Andre Martins, Philipp Mondorf, Vera Neplenbroek, Sandro Pezzelle, Barbara Plank, David Schlangen, Alessandro Suglia, Aditya K Surikuchi, Ece Takmaz, Alberto Testoni",
  "venue": "ACL",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>There is an increasing trend towards evaluating NLP models with LLMs instead of human judgments, raising questions about the validity of these evaluations, as well as their reproducibility in the case of proprietary models. We provide JUDGE-BENCH, an extensible collection of 20 NLP datasets with human annotations covering a broad range of evaluated properties and types of data, and comprehensively evaluate 11 current LLMs, covering both open-weight and proprietary models, for their ability to replicate the annotations. Our evaluations show substantial variance across models and datasets. Models are reliable evaluators on some tasks, but overall display substantial variability depending on the property being evaluated, the expertise level of the human judges, and whether the language is human or model-generated. We conclude that LLMs should be carefully validated against human judgments before being used as evaluators.</p>`,
  "streams": [
  "evaluation-metrics",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2025.acl-short.20/",
  "bibtex": `@inproceedings{bavaresco-etal-2025-llms,
    title = "{LLM}s instead of Human Judges? A Large Scale Empirical Study across 20 {NLP} Evaluation Tasks",
    author = "Bavaresco, Anna  and
      Bernardi, Raffaella  and
      Bertolazzi, Leonardo  and
      Elliott, Desmond  and
      Fern{\\'a}ndez, Raquel  and
      Gatt, Albert  and
      Ghaleb, Esam  and
      Giulianelli, Mario  and
      Hanna, Michael  and
      Koller, Alexander  and
      Martins, Andre  and
      Mondorf, Philipp  and
      Neplenbroek, Vera  and
      Pezzelle, Sandro  and
      Plank, Barbara  and
      Schlangen, David  and
      Suglia, Alessandro  and
      Surikuchi, Aditya K  and
      Takmaz, Ece  and
      Testoni, Alberto",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-short.20/",
    doi = "10.18653/v1/2025.acl-short.20",
    pages = "238--255",
    ISBN = "979-8-89176-252-7",
}`
}
},
  {
  "id": 106,
  "title": "Can Automatic Metrics Assess High-Quality Translations?",
  "authors": "Sweta Agrawal, Ant\u00f3nio Farinhas, Ricardo Rei, Andre Martins",
  "venue": "EMNLP",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Automatic metrics for evaluating translation quality are typically validated by measuring how well they correlate with human assessments. However, correlation methods tend to capture only the ability of metrics to differentiate between good and bad source-translation pairs, overlooking their reliability in distinguishing alternative translations for the same source. In this paper, we confirm that this is indeed the case by showing that current metrics are insensitive to nuanced differences in translation quality. This effect is most pronounced when the quality is high and the variance among alternatives is low. Given this finding, we shift towards detecting high-quality correct translations, an important problem in practical decision-making scenarios where a binary check of correctness is prioritized over a nuanced evaluation of quality. Using the MQM framework as the gold standard, we systematically stress-test the ability of current metrics to identify translations with no errors as marked by humans. Our findings reveal that current metrics often over or underestimate translation quality, indicating significant room for improvement in machine translation evaluation.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2024.emnlp-main.802/",
  "bibtex": `@inproceedings{agrawal-etal-2024-automatic-metrics,
    title = "Can Automatic Metrics Assess High-Quality Translations?",
    author = "Agrawal, Sweta  and
      Farinhas, Ant{\\'o}nio  and
      Rei, Ricardo  and
      Martins, Andre",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.802/",
    doi = "10.18653/v1/2024.emnlp-main.802",
    pages = "14491--14502",
}`
}
},
  {
  "id": 105,
  "title": "CroissantLLM: A Truly Bilingual French-English Language Model",
  "authors": "Manuel Faysse, Patrick Fernandes, Nuno Guerreiro, Ant\u00f3nio Loison, Duarte Alves, Caio Corro, Nicolas Boizard, Janaina Alves, Ricardo Rei, Pedro Rapha\u00ebl Martins, Antoni Bigata Casademunt, Fran\u00e7ois Yvon, Andr\u00e9 F. T. Martins, Gautier Viaud, C\u00e9line Hudelot, Pierre Colombo",
  "venue": "TMLR",
  "year": 2024,
  "type": "journal",
  "abstract": `<p>We introduce CroissantLLM, a 1.3B language model pretrained on a set of 3T English and French tokens, to bring to the research and industrial community a high-performance, fully open-sourced bilingual model that runs swiftly on consumer-grade local hardware. To that end, we pioneer the approach of training an intrinsically bilingual model with a 1:1 English-to-French pretraining data ratio, a custom tokenizer, and bilingual finetuning datasets. We release the training dataset, notably containing a French split with manually curated, high-quality, and varied data sources. To assess performance outside of English, we craft a novel benchmark, FrenchBench, consisting of an array of classification and generation tasks, covering various orthogonal aspects of model performance in the French Language. Additionally, rooted in transparency and to foster further Large Language Model research, we release codebases, and dozens of checkpoints across various model sizes, training data distributions, and training steps, as well as fine-tuned Chat models, and strong translation models. We evaluate our model through the FMTI framework, and validate 81 % of the transparency criteria, far beyond the scores of even most open initiatives. This work enriches the NLP landscape, breaking away from previous English-centric work in order to strengthen our understanding of multilinguality in language models.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://openreview.net/forum?id=uA19Xo1o31",
  "bibtex": `@article{
faysse2025croissantllm,
title={Croissant{LLM}: A Truly Bilingual French-English Language Model},
author={Manuel Faysse and Patrick Fernandes and Nuno M Guerreiro and Ant{\\'o}nio Loison and Duarte Miguel Alves and Caio Corro and Nicolas Boizard and Jo{\\~a}o Alves and Ricardo Rei and Pedro Henrique Martins and Antoni Bigata Casademunt and Fran{\\c{c}}ois Yvon and Andre Martins and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=uA19Xo1o31},
note={}
}`
}
},
  {
  "id": 104,
  "title": "XAMPLER: Learning to Retrieve Cross-Lingual In-Context Examples",
  "authors": "Peiqin Lin, Andre Martins, Hinrich Schuetze",
  "venue": "NAACL Findings",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Recent studies indicate that leveraging off-the-shelf or fine-tuned retrievers, capable of retrieving relevant in-context examples tailored to the input query, enhances few-shot in-context learning of English. However, adapting these methods to other languages, especially low-resource ones, poses challenges due to the scarcity of cross-lingual retrievers and annotated data. Thus, we introduce XAMPLER: Cross-Lingual Example Retrieval, a method tailored to tackle the challenge of cross-lingual in-context learning using only annotated English data. XAMPLER first trains a retriever based on Glot500, a multilingual small language model, using positive and negative English examples constructed from the predictions of a multilingual large language model, i.e., MaLA500. Leveraging the cross-lingual capacity of the retriever, it can directly retrieve English examples as few-shot examples for in-context learning of target languages. Experiments on two multilingual text classification benchmarks, namely SIB200 with 176 languages and MasakhaNEWS with 16 languages, demonstrate that XAMPLER substantially improves the in-context learning performance across languages.</p>`,
  "streams": [
  "multilingual-translation",
  "retrieval"
],
  "links": {
  "paper": "https://aclanthology.org/2025.findings-naacl.221/",
  "bibtex": `@inproceedings{lin-etal-2025-xampler,
    title = "{XAMPLER}: Learning to Retrieve Cross-Lingual In-Context Examples",
    author = "Lin, Peiqin  and
      Martins, Andre  and
      Schuetze, Hinrich",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.221/",
    doi = "10.18653/v1/2025.findings-naacl.221",
    pages = "3968--3977",
    ISBN = "979-8-89176-195-7",
}`
}
},
  {
  "id": 103,
  "title": "Did Translation Models Get More Robust Without Anyone Even Noticing?",
  "authors": "Ben Peters, Andre Martins",
  "venue": "ACL",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Neural machine translation (MT) models achieve strong results across a variety of settings, but it is widely believed that they are highly sensitive to "noisy" inputs, such as spelling errors, abbreviations, and other formatting issues. In this paper, we revisit this insight in light of recent multilingual MT models and large language models (LLMs) applied to machine translation. Somewhat surprisingly, we show through controlled experiments that these models are far more robust to many kinds of noise than previous models, even when they perform similarly on clean data. This is notable because, even though LLMs have more parameters and more complex training processes than past models, none of the open ones we consider use any techniques specifically designed to encourage robustness. Next, we show that similar trends hold for social media translation experiments -- LLMs are more robust to social media text. We include an analysis of the circumstances in which source correction techniques can be used to mitigate the effects of noise. Altogether, we show that robustness to many types of noise has increased.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2025.acl-long.122/",
  "bibtex": `@inproceedings{peters-martins-2025-translation,
    title = "Did Translation Models Get More Robust Without Anyone {E}ven Noticing?",
    author = "Peters, Ben  and
      Martins, Andre",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.122/",
    doi = "10.18653/v1/2025.acl-long.122",
    pages = "2445--2458",
    ISBN = "979-8-89176-251-0",
}`
}
},
  {
  "id": 102,
  "title": "SaulLM-7B: A pioneering Large Language Model for Law",
  "authors": "Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver, Rui Melo, Caio Corro, Andre F. T. Martins, Fabrizio Esposito, Vera L\u00facia Raposo, Sofia Morgado, Michael Desa",
  "venue": "arXiv",
  "year": 2024,
  "type": "preprint",
  "abstract": `<p>In this paper, we introduce SaulLM-7B, a large language model (LLM) tailored for the legal domain. With 7 billion parameters, SaulLM-7B is the first LLM designed explicitly for legal text comprehension and generation. Leveraging the Mistral 7B architecture as its foundation, SaulLM-7B is trained on an English legal corpus of over 30 billion tokens. SaulLM-7B exhibits state-of-the-art proficiency in understanding and processing legal documents. Additionally, we present a novel instructional fine-tuning method that leverages legal datasets to further enhance SaulLM-7B's performance in legal tasks. SaulLM-7B is released under the MIT License.</p>`,
  "streams": [
  "resources",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2403.03883",
  "bibtex": `@article{colombo2024saullm,
  title={Saullm-7b: A pioneering large language model for law},
  author={Colombo, Pierre and Pires, Telmo Pessoa and Boudiaf, Malik and Culver, Dominic and Melo, Rui and Corro, Caio and Martins, Andre FT and Esposito, Fabrizio and Raposo, Vera L{\\'u}cia and Morgado, Sofia and others},
  journal={arXiv preprint arXiv:2403.03883},
  year={2024}
}`
}
},
  {
  "id": 101,
  "title": "Tower: An Open Multilingual Large Language Model for Translation-Related Tasks",
  "authors": "Duarte M. Alves, Jos\u00e9 Pombal, Nuno M. Guerreiro, Pedro H. Martins, Jo\u00e3o Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, Pierre Colombo, Jos\u00e9 G.C. de Souza, Andr\u00e9 F.T. Martins",
  "venue": "COLM",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>While general-purpose large language models (LLMs) demonstrate proficiency on multiple tasks within the domain of translation, approaches based on open LLMs are competitive only when specializing on a single task. In this paper, we propose a recipe for tailoring LLMs to multiple tasks present in translation workflows. We perform continued pretraining on a multilingual mixture of monolingual and parallel data, creating TowerBase, followed by finetuning on instructions relevant for translation processes, creating TowerInstruct. Our final model surpasses open alternatives on several tasks relevant to translation workflows and is competitive with general-purpose closed LLMs. To facilitate future research, we release the Tower models, our specialization dataset, an evaluation framework for LLMs focusing on the translation ecosystem, and a collection of model generations, including ours, on our benchmark.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://openreview.net/forum?id=EHPns3hVkj",
  "bibtex": `@inproceedings{
alves2024tower,
title={Tower: An Open Multilingual Large Language Model for Translation-Related Tasks},
author={Duarte Miguel Alves and Jos{\\'e} Pombal and Nuno M Guerreiro and Pedro Henrique Martins and Jo{\\~a}o Alves and Amin Farajian and Ben Peters and Ricardo Rei and Patrick Fernandes and Sweta Agrawal and Pierre Colombo and Jos{\\'e} G. C. de Souza and Andre Martins},
booktitle={First Conference on Language Modeling},
year={2024},
url={https://openreview.net/forum?id=EHPns3hVkj}
}`
}
},
  {
  "id": 100,
  "title": "Sparse and Structured Hopfield Networks",
  "authors": "Saul Santos, Vlad Niculae, Daniel McNamee, Andr\u00e9 F. T. Martins",
  "venue": "ICML",
  "year": 2024,
  "type": "conference",
  "award": "Spotlight",
  "abstract": `<p>Modern Hopfield networks have enjoyed recent interest due to their connection to attention in transformers. Our paper provides a unified framework for sparse Hopfield networks by establishing a link with Fenchel-Young losses. The result is a new family of Hopfield-Fenchel-Young energies whose update rules are end-to-end differentiable sparse transformations. We reveal a connection between loss margins, sparsity, and exact memory retrieval. We further extend this framework to structured Hopfield networks via the SparseMAP transformation, which can retrieve pattern associations instead of a single pattern. Experiments on multiple instance learning and text rationalization demonstrate the usefulness of our approach.</p>`,
  "streams": [
  "memory",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2402.13725",
  "bibtex": `@InProceedings{pmlr-v235-santos24a,
  title = 	 {Sparse and Structured Hopfield Networks},
  author =       {Santos, Saul Jos\\'{e} Rodrigues Dos and Niculae, Vlad and Mcnamee, Daniel C and Martins, Andre},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {43368--43388},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/santos24a/santos24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/santos24a.html},
}`
}
},
  {
  "id": 99,
  "title": "Non-Exchangeable Conformal Language Generation with Nearest Neighbors",
  "authors": "Dennis Ulmer, Chrysoula Zerva, Andr\u00e9 F. T. Martins, Yvette Graham, Matthew Purver",
  "venue": "EACL Findings",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Quantifying uncertainty in automatically generated text is important for letting humans check potential hallucinations and making systems more reliable. Conformal prediction is an attractive framework to provide predictions imbued with statistical guarantees, however, its application to text generation is challenging since any i.i.d. assumptions are not realistic. In this paper, we bridge this gap by leveraging recent results on <em>non-exchangeable</em> conformal prediction, which still ensures bounds on coverage. The result, <em>non-exchangeable conformal nucleus sampling</em>, is a novel extension of the conformal prediction framework to generation based on nearest neighbors. Our method can be used post-hoc for an arbitrary model without extra training and supplies token-level, calibrated prediction sets equipped with statistical guarantees. Experiments in machine translation and language modeling show encouraging results in generation quality. By also producing tighter prediction sets with good coverage, we thus give a more theoretically principled way to perform sampling with conformal guarantees.</p>`,
  "streams": [
  "multilingual-translation",
  "retrieval",
  "uncertainty"
],
  "links": {
  "paper": "https://aclanthology.org/2024.findings-eacl.129/",
  "bibtex": `@inproceedings{ulmer-etal-2024-non,
    title = "Non-Exchangeable Conformal Language Generation with Nearest Neighbors",
    author = "Ulmer, Dennis  and
      Zerva, Chrysoula  and
      Martins, Andre",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.129/",
    pages = "1909--1929",
}`
}
},
  {
  "id": 98,
  "title": "MaLA-500: Massive Language Adaptation of Large Language Models",
  "authors": "Peiqin Lin, Shaoxiong Ji, J\u00f6rg Tiedemann, Andr\u00e9 F. T. Martins, Hinrich Sch\u00fctze",
  "venue": "arXiv",
  "year": 2024,
  "type": "preprint",
  "abstract": `<p>Large language models (LLMs) have advanced the state of the art in natural language processing. However, their predominant design for English or a limited set of languages creates a substantial gap in their effectiveness for low-resource languages. To bridge this gap, we introduce MaLA-500, a novel large language model designed to cover an extensive range of 534 languages. To train MaLA-500, we employ vocabulary extension and continued pretraining on LLaMA 2 with Glot500-c. Our intrinsic evaluation demonstrates that MaLA-500 is better at predicting the given texts of low-resource languages than existing multilingual LLMs. Moreover, the extrinsic evaluation of in-context learning shows that MaLA-500 outperforms previous LLMs on SIB200 and Taxi1500 by a significant margin, i.e., 11.68% and 4.82% marco-average accuracy across languages. We release MaLA-500 at <a href="https://huggingface.co/MaLA-LM">[https://huggingface.co/MaLA-LM](https://huggingface.co/MaLA-LM)</a></p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://arxiv.org/abs/2401.13303",
  "bibtex": `@article{lin2024mala,
  title={Mala-500: Massive language adaptation of large language models},
  author={Lin, Peiqin and Ji, Shaoxiong and Tiedemann, J{\\"o}rg and Martins, Andr{\\'e} FT and Sch{\\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2401.13303},
  year={2024}
}`
}
},
  {
  "id": 97,
  "title": "Findings of the Quality Estimation Shared Task at WMT 2024 Are LLMs Closing the Gap in QE?",
  "authors": "Chrysoula Zerva, Frederic Blain, Jos\u00e9 G. C. De Souza, Diptesh Kanojia, Sourabh Deoghare, Nuno M. Guerreiro, Giuseppe Attanasio, Ricardo Rei, Constantin Orasan, Matteo Negri, Marco Turchi, Rajen Chatterjee, Pushpak Bhattacharyya, Markus Freitag, Andr\u00e9 Martins",
  "venue": "WMT",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>We report the results of the WMT 2024 shared task on Quality Estimation, in which the challenge is to predict the quality of the output of neural machine translation systems at the word and sentence levels, without access to reference translations. In this edition, we expanded our scope to assess the potential for quality estimates to help in the correction of translated outputs, hence including an automated post-editing (APE) direction. We publish new test sets with human annotations that target two directions: providing new Multidimensional Quality Metrics (MQM) annotations for three multi-domain language pairs (English to German, Spanish and Hindi) and extending the annotations on Indic languages providing direct assessments and post edits for translation from English into Hindi, Gujarati, Tamil and Telugu. We also perform a detailed analysis of the behaviour of different models with respect to different phenomena including gender bias, idiomatic language, and numerical and entity perturbations. We received submissions based both on traditional, encoder-based approaches as well as large language model (LLM) based ones.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2024.wmt-1.3/",
  "bibtex": `@inproceedings{zerva-etal-2024-findings,
    title = "Findings of the Quality Estimation Shared Task at {WMT} 2024: Are {LLM}s Closing the Gap in {QE}?",
    author = "Zerva, Chrysoula  and
      Blain, Frederic  and
      C. De Souza, Jos{\\'e} G.  and
      Kanojia, Diptesh  and
      Deoghare, Sourabh  and
      Guerreiro, Nuno M.  and
      Attanasio, Giuseppe  and
      Rei, Ricardo  and
      Orasan, Constantin  and
      Negri, Matteo  and
      Turchi, Marco  and
      Chatterjee, Rajen  and
      Bhattacharyya, Pushpak  and
      Freitag, Markus  and
      Martins, Andr{\\'e}",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Ninth Conference on Machine Translation",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wmt-1.3/",
    doi = "10.18653/v1/2024.wmt-1.3",
    pages = "82--109",
}`
}
},
  {
  "id": 96,
  "title": "Non-Exchangeable Conformal Risk Control",
  "authors": "Ant\u00f3nio Farinhas, Chrysoula Zerva, Dennis Ulmer, Andr\u00e9 F. T. Martins",
  "venue": "ICLR",
  "year": 2024,
  "type": "conference",
  "abstract": `<p>Split conformal prediction has recently sparked great interest due to its ability to provide formally guaranteed uncertainty sets or intervals for predictions made by black-box neural models, ensuring a predefined probability of containing the actual ground truth. While the original formulation assumes data exchangeability, some extensions handle non-exchangeable data, which is often the case in many real-world scenarios. In parallel, some progress has been made in conformal methods that provide statistical guarantees for a broader range of objectives, such as bounding the best F1-score or minimizing the false negative rate in expectation. In this paper, we leverage and extend these two lines of work by proposing non-exchangeable conformal risk control, which allows controlling the expected value of any monotone loss function when the data is not exchangeable. Our framework is flexible, makes very few assumptions, and allows weighting the data based on its relevance for a given test example; a careful choice of weights may result on tighter bounds, making our framework useful in the presence of change points, time series, or other forms of distribution drift. Experiments with both synthetic and real world data show the usefulness of our method.</p>`,
  "streams": [
  "theory",
  "uncertainty"
],
  "links": {
  "paper": "https://openreview.net/forum?id=j511LaqEeP",
  "bibtex": `@inproceedings{
farinhas2024nonexchangeable,
title={Non-Exchangeable Conformal Risk Control},
author={Ant{\\'o}nio Farinhas and Chrysoula Zerva and Dennis Thomas Ulmer and Andre Martins},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=j511LaqEeP}
}`
}
},
  {
  "id": 95,
  "title": "Scaling up CometKiwi: Unbabel-IST 2023 Submission for the Quality Estimation Shared Task",
  "authors": "Ricardo Rei, Nuno M. Guerreiro, Jos\u00e9 Pombal, Daan van Stigt, Marcos Treviso, Luisa Coheur, Jos\u00e9 G. C. de Souza, Andr\u00e9 Martins",
  "venue": "WMT",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>We present the joint contribution of Unbabel and Instituto Superior Técnico to the WMT 2023 Shared Task on Quality Estimation (QE). Our team participated on all tasks: Sentence- and Word-level Quality Prediction and Fine-grained error span detection. For all tasks we build on the CometKiwi model (rei et al. 2022). Our multilingual approaches are ranked first for all tasks, reaching state-of-the-art performance for quality estimation at word-, span- and sentence-level granularity. Compared to the previous state-of-the-art, CometKiwi, we show large improvements in correlation with human judgements (up to 10 Spearman points) and surpassing the second-best multilingual submission with up to 3.8 absolute points.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2023.wmt-1.73/",
  "bibtex": `@inproceedings{rei-etal-2023-scaling,
    title = "Scaling up {C}omet{K}iwi: Unbabel-{IST} 2023 Submission for the Quality Estimation Shared Task",
    author = "Rei, Ricardo  and
      Guerreiro, Nuno M.  and
      Pombal, Jos{\\~A}{\\textcopyright}  and
      van Stigt, Daan  and
      Treviso, Marcos  and
      Coheur, Luisa  and
      C. de Souza, Jos{\\'e} G.  and
      Martins, Andr{\\'e}",
    editor = "Koehn, Philipp  and
      Haddow, Barry  and
      Kocmi, Tom  and
      Monz, Christof",
    booktitle = "Proceedings of the Eighth Conference on Machine Translation",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wmt-1.73/",
    doi = "10.18653/v1/2023.wmt-1.73",
    pages = "841--848",
}`
}
},
  {
  "id": 94,
  "title": "CREST: A Joint Framework for Rationalization and Counterfactual Text Generation",
  "authors": "Marcos Treviso, Alexis Ross, Nuno M. Guerreiro, Andr\u00e9 Martins",
  "venue": "ACL",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Selective rationales and counterfactual examples have emerged as two effective, complementary classes of interpretability methods for analyzing and training NLP models. However, prior work has not explored how these methods can be integrated to combine their complementary advantages. We overcome this limitation by introducing CREST (ContRastive Edits with Sparse raTionalization), a joint framework for selective rationalization and counterfactual text generation, and show that this framework leads to improvements in counterfactual quality, model robustness, and interpretability. First, CREST generates valid counterfactuals that are more natural than those produced by previous methods, and subsequently can be used for data augmentation at scale, reducing the need for human-generated examples. Second, we introduce a new loss function that leverages CREST counterfactuals to regularize selective rationales and show that this regularization improves both model robustness and rationale quality, compared to methods that do not leverage CREST counterfactuals. Our results demonstrate that CREST successfully bridges the gap between selective rationales and counterfactual examples, addressing the limitations of existing methods and providing a more comprehensive view of a model's predictions.</p>`,
  "streams": [
  "interpretability",
  "resources",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2023.acl-long.842/",
  "bibtex": `@inproceedings{treviso-etal-2023-crest,
    title = "{CREST}: A Joint Framework for Rationalization and Counterfactual Text Generation",
    author = "Treviso, Marcos  and
      Ross, Alexis  and
      Guerreiro, Nuno M.  and
      Martins, Andr{\\'e}",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.842/",
    doi = "10.18653/v1/2023.acl-long.842",
    pages = "15109--15126",
}`
}
},
  {
  "id": 93,
  "title": "The Inside Story: Towards Better Understanding of Machine Translation Neural Evaluation Metrics",
  "authors": "Ricardo Rei, Nuno M. Guerreiro, Marcos Treviso, Luisa Coheur, Alon Lavie, Andr\u00e9 Martins",
  "venue": "ACL",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Neural metrics for machine translation evaluation, such as COMET, exhibit significant improvements in their correlation with human judgments, as compared to traditional metrics based on lexical overlap, such as BLEU. Yet, neural metrics are, to a great extent, "black boxes" returning a single sentence-level score without transparency about the decision-making process. In this work, we develop and compare several neural explainability methods and demonstrate their effectiveness for interpreting state-of-the-art fine-tuned neural metrics. Our study reveals that these metrics leverage token-level information that can be directly attributed to translation errors, as assessed through comparison of token-level neural saliency maps with Multidimensional Quality Metrics (MQM) annotations and with synthetically-generated critical translation errors. To ease future research, we release our code at: <a href="https://github.com/Unbabel/COMET/tree/explainable-metrics">[https://github.com/Unbabel/COMET/tree/explainable-metrics](https://github.com/Unbabel/COMET/tree/explainable-metrics)</a></p>`,
  "streams": [
  "evaluation-metrics",
  "interpretability"
],
  "links": {
  "paper": "https://aclanthology.org/2023.acl-short.94/",
  "bibtex": `@inproceedings{rei-etal-2023-inside,
    title = "The Inside Story: Towards Better Understanding of Machine Translation Neural Evaluation Metrics",
    author = "Rei, Ricardo  and
      Guerreiro, Nuno M.  and
      Treviso, Marcos  and
      Coheur, Luisa  and
      Lavie, Alon  and
      Martins, Andr{\\'e}",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.94/",
    doi = "10.18653/v1/2023.acl-short.94",
    pages = "1089--1105",
}`
}
},
  {
  "id": 92,
  "title": "Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation",
  "authors": "Patrick Fernandes, Aman Madaan, Emmy Liu, Ant\u00f3nio Farinhas, Pedro Henrique Martins, Amanda Bertsch, Jos\u00e9 G. C. de Souza, Shuyan Zhou, Tongshuang Wu, Graham Neubig, Andr\u00e9 F. T. Martins",
  "venue": "TACL",
  "year": 2023,
  "type": "journal",
  "abstract": `<p>Natural language generation has witnessed significant advancements due to the training of large language models on vast internet-scale datasets. Despite these advancements, there exists a critical challenge: These models can inadvertently generate content that is toxic, inaccurate, and unhelpful, and existing automatic evaluation metrics often fall short of identifying these shortcomings. As models become more capable, human feedback is an invaluable signal for evaluating and improving models. This survey aims to provide an overview of recent research that has leveraged human feedback to improve natural language generation. First, we introduce a taxonomy distilled from existing research to categorize and organize the varied forms of feedback. Next, we discuss how feedback can be described by its format and objective, and cover the two approaches proposed to use feedback (either for training or decoding): directly using feedback or training feedback models. We also discuss existing datasets for human-feedback data collection, and concerns surrounding feedback collection. Finally, we provide an overview of the nascent field of AI feedback, which uses large language models to make judgments based on a set of principles and minimize the need for human intervention. We also release a website of this survey at feedback-gap-survey.info.</p>`,
  "streams": [
  "evaluation-metrics"
],
  "links": {
  "paper": "https://aclanthology.org/2023.tacl-1.92/",
  "bibtex": `@article{fernandes-etal-2023-bridging,
    title = "Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation",
    author = "Fernandes, Patrick  and
      Madaan, Aman  and
      Liu, Emmy  and
      Farinhas, Ant{\\'o}nio  and
      Martins, Pedro Henrique  and
      Bertsch, Amanda  and
      de Souza, Jos{\\'e} G. C.  and
      Zhou, Shuyan  and
      Wu, Tongshuang  and
      Neubig, Graham  and
      Martins, Andr{\\'e} F. T.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "11",
    year = "2023",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2023.tacl-1.92/",
    doi = "10.1162/tacl_a_00626",
    pages = "1643--1668",
}`
}
},
  {
  "id": 91,
  "title": "Hallucinations in Large Multilingual Translation Models",
  "authors": "Nuno M. Guerreiro, Duarte M. Alves, Jonas Waldendorf, Barry Haddow, Alexandra Birch, Pierre Colombo, Andr\u00e9 F. T. Martins",
  "venue": "TACL",
  "year": 2023,
  "type": "journal",
  "abstract": `<p>Hallucinated translations can severely undermine and raise safety issues when machine translation systems are deployed in the wild. Previous research on the topic focused on small bilingual models trained on high-resource languages, leaving a gap in our understanding of hallucinations in multilingual models across diverse translation scenarios. In this work, we fill this gap by conducting a comprehensive analysis—over 100 language pairs across various resource levels and going beyond English-centric directions—on both the M2M neural machine translation (NMT) models and GPT large language models (LLMs). Among several insights, we highlight that models struggle with hallucinations primarily in low-resource directions and when translating out of English, where, critically, they may reveal toxic patterns that can be traced back to the training data. We also find that LLMs produce qualitatively different hallucinations to those of NMT models. Finally, we show that hallucinations are hard to reverse by merely scaling models trained with the same data. However, employing more diverse models, trained on different data or with different procedures, as fallback systems can improve translation quality and virtually eliminate certain pathologies.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2023.tacl-1.85/",
  "code": "https://github.com/deep-spin/hallucinations-in-nmt",
  "bibtex": `@article{guerreiro-etal-2023-hallucinations,
    title = "Hallucinations in Large Multilingual Translation Models",
    author = "Guerreiro, Nuno M.  and
      Alves, Duarte M.  and
      Waldendorf, Jonas  and
      Haddow, Barry  and
      Birch, Alexandra  and
      Colombo, Pierre  and
      Martins, Andr{\\'e} F. T.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "11",
    year = "2023",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2023.tacl-1.85/",
    doi = "10.1162/tacl_a_00615",
    pages = "1500--1517",
}`
}
},
  {
  "id": 90,
  "title": "Aligning Neural Machine Translation Models: Human Feedback in Training and Inference",
  "authors": "Miguel Moura Ramos, Patrick Fernandes, Ant\u00f3nio Farinhas, Andr\u00e9 F. T. Martins",
  "venue": "EAMT",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Reinforcement learning from human feedback (RLHF) is a recent technique to improve the quality of the text generated by a language model, making it closer to what humans would generate. A core ingredient in RLHF's success in aligning and improving large language models (LLMs) is its reward model, trained using human feedback on model outputs. In machine translation (MT), where metrics trained from human annotations can readily be used as reward models, recent methods using minimum Bayes risk decoding and reranking have succeeded in improving the final quality of translation. In this study, we comprehensively explore and compare techniques for integrating quality metrics as reward models into the MT pipeline. This includes using the reward model for data filtering, during the training phase through RL, and at inference time by employing reranking techniques, and we assess the effects of combining these in a unified approach. Our experimental results, conducted across multiple translation tasks, underscore the crucial role of effective data filtering, based on estimated quality, in harnessing the full potential of RL in enhancing MT quality. Furthermore, our findings demonstrate the effectiveness of combining RL training with reranking techniques, showcasing substantial improvements in translation quality.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2024.eamt-1.22/",
  "bibtex": `@inproceedings{ramos-etal-2024-aligning,
    title = "Aligning Neural Machine Translation Models: Human Feedback in Training and Inference",
    author = "Ramos, Miguel  and
      Fernandes, Patrick  and
      Farinhas, Ant{\\'o}nio  and
      Martins, Andre",
    editor = "Scarton, Carolina  and
      Prescott, Charlotte  and
      Bayliss, Chris  and
      Oakley, Chris  and
      Wright, Joanna  and
      Wrigley, Stuart  and
      Song, Xingyi  and
      Gow-Smith, Edward  and
      Bawden, Rachel  and
      S{\\'a}nchez-Cartagena, V{\\'i}ctor M  and
      Cadwell, Patrick  and
      Lapshinova-Koltunski, Ekaterina  and
      Cabarr{\\~a}o, Vera  and
      Chatzitheodorou, Konstantinos  and
      Nurminen, Mary  and
      Kanojia, Diptesh  and
      Moniz, Helena",
    booktitle = "Proceedings of the 25th Annual Conference of the European Association for Machine Translation (Volume 1)",
    month = jun,
    year = "2024",
    address = "Sheffield, UK",
    publisher = "European Association for Machine Translation (EAMT)",
    url = "https://aclanthology.org/2024.eamt-1.22/",
    pages = "258--274",
}`
}
},
  {
  "id": 89,
  "title": "Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning",
  "authors": "Duarte Alves, Nuno Guerreiro, Jo\u00e3o Alves, Jos\u00e9 Pombal, Ricardo Rei, Jos\u00e9 de Souza, Pierre Colombo, Andre Martins",
  "venue": "EMNLP Findings",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Large language models (LLMs) are a promising avenue for machine translation (MT). However, current LLM-based MT systems are brittle: their effectiveness highly depends on the choice of few-shot examples and they often require extra post-processing due to overgeneration. Alternatives such as finetuning on translation instructions are computationally expensive and may weaken in-context learning capabilities, due to overspecialization. In this paper, we provide a closer look at this problem. We start by showing that adapter-based finetuning with LoRA matches the performance of traditional finetuning while reducing the number of training parameters by a factor of 50. This method also outperforms few-shot prompting and eliminates the need for post-processing or in-context examples. However, we show that finetuning generally degrades few-shot performance, hindering adaptation capabilities. Finally, to obtain the best of both worlds, we propose a simple approach that incorporates few-shot examples during finetuning. Experiments on 10 language pairs show that our proposed approach recovers the original few-shot capabilities while keeping the added benefits of finetuning.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2023.findings-emnlp.744/",
  "bibtex": `@inproceedings{alves-etal-2023-steering,
    title = "Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning",
    author = "Alves, Duarte  and
      Guerreiro, Nuno  and
      Alves, Jo{\\~a}o  and
      Pombal, Jos{\\'e}  and
      Rei, Ricardo  and
      de Souza, Jos{\\'e}  and
      Colombo, Pierre  and
      Martins, Andre",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.744/",
    doi = "10.18653/v1/2023.findings-emnlp.744",
    pages = "11127--11148",
}`
}
},
  {
  "id": 88,
  "title": "An Empirical Study of Translation Hypothesis Ensembling with Large Language Models",
  "authors": "Ant\u00f3nio Farinhas, Jos\u00e9 de Souza, Andre Martins",
  "venue": "EMNLP",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Large language models (LLMs) are becoming a one-fits-many solution, but they sometimes hallucinate or produce unreliable output. In this paper, we investigate how hypothesis ensembling can improve the quality of the generated text for the specific problem of LLM-based machine translation. We experiment with several techniques for ensembling hypotheses produced by LLMs such as ChatGPT, LLaMA, and Alpaca. We provide a comprehensive study along multiple dimensions, including the method to generate hypotheses (multiple prompts, temperature-based sampling, and beam search) and the strategy to produce the final translation (instruction-based, quality-based reranking, and minimum Bayes risk (MBR) decoding). Our results show that MBR decoding is a very effective method, that translation quality can be improved using a small number of samples, and that instruction tuning has a strong impact on the relation between the diversity of the hypotheses and the sampling temperature.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2023.emnlp-main.733/",
  "bibtex": `@inproceedings{farinhas-etal-2023-empirical,
    title = "An Empirical Study of Translation Hypothesis Ensembling with Large Language Models",
    author = "Farinhas, Ant{\\'o}nio  and
      de Souza, Jos{\\'e}  and
      Martins, Andre",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.733/",
    doi = "10.18653/v1/2023.emnlp-main.733",
    pages = "11956--11970",
}`
}
},
  {
  "id": 87,
  "title": "Efficient Methods for Natural Language Processing: A Survey",
  "authors": "Marcos Treviso, Ji-Ung Lee, Tianchu Ji, Betty van Aken, Qingqing Cao, Manuel R. Ciosici, Michael Hassid, Kenneth Heafield, Sara Hooker, Colin Raffel, Pedro H. Martins, Andr\u00e9 F. T. Martins, Jessica Zosa Forde, Peter Milder, Edwin Simpson, Noam Slonim, Jesse Dodge, Emma Strubell, Niranjan Balasubramanian, Leon Derczynski, Iryna Gurevych, Roy Schwartz",
  "venue": "TACL",
  "year": 2023,
  "type": "journal",
  "abstract": `<p>Recent work in natural language processing (NLP) has yielded appealing results from scaling model parameters and training data; however, using only scale to improve performance means that resource consumption also grows. Such resources include data, time, storage, or energy, all of which are naturally limited and unevenly distributed. This motivates research into efficient methods that require fewer resources to achieve similar results. This survey synthesizes and relates current methods and findings in efficient NLP. We aim to provide both guidance for conducting NLP under limited resources, and point towards promising research directions for developing more efficient methods.</p>`,
  "streams": [
  "efficiency",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2023.tacl-1.48/",
  "bibtex": `@article{treviso-etal-2023-efficient,
    title = "Efficient Methods for Natural Language Processing: A Survey",
    author = "Treviso, Marcos  and
      Lee, Ji-Ung  and
      Ji, Tianchu  and
      van Aken, Betty  and
      Cao, Qingqing  and
      Ciosici, Manuel R.  and
      Hassid, Michael  and
      Heafield, Kenneth  and
      Hooker, Sara  and
      Raffel, Colin  and
      Martins, Pedro H.  and
      Martins, Andr{\\'e} F. T.  and
      Forde, Jessica Zosa  and
      Milder, Peter  and
      Simpson, Edwin  and
      Slonim, Noam  and
      Dodge, Jesse  and
      Strubell, Emma  and
      Balasubramanian, Niranjan  and
      Derczynski, Leon  and
      Gurevych, Iryna  and
      Schwartz, Roy",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "11",
    year = "2023",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2023.tacl-1.48/",
    doi = "10.1162/tacl_a_00577",
    pages = "826--860",
}`
}
},
  {
  "id": 86,
  "title": "The Devil Is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation",
  "authors": "Patrick Fernandes, Daniel Deutsch, Mara Finkelstein, Parker Riley, Andr\u00e9 Martins, Graham Neubig, Ankush Garg, Jonathan Clark, Markus Freitag, Orhan Firat",
  "venue": "WMT",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Automatic evaluation of machine translation (MT) is a critical tool driving the rapid iterative development of MT systems. While considerable progress has been made on estimating a single scalar quality score, current metrics lack the informativeness of more detailed schemes that annotate individual errors, such as Multidimensional Quality Metrics (MQM). In this paper, we help fill this gap by proposing AutoMQM, a prompting technique which leverages the reasoning and in-context learning capabilities of large language models (LLMs) and asks them to identify and categorize errors in translations. We start by evaluating recent LLMs, such as PaLM and PaLM-2, through simple score prediction prompting, and we study the impact of labeled data through in-context learning and finetuning. We then evaluate AutoMQM with PaLM-2 models, and we find that it improves performance compared to just prompting for scores (with particularly large gains for larger models) while providing interpretability through error spans that align with human annotations.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2023.wmt-1.100/",
  "bibtex": `@inproceedings{fernandes-etal-2023-devil,
    title = "The Devil Is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation",
    author = "Fernandes, Patrick  and
      Deutsch, Daniel  and
      Finkelstein, Mara  and
      Riley, Parker  and
      Martins, Andr{\\'e}  and
      Neubig, Graham  and
      Garg, Ankush  and
      Clark, Jonathan  and
      Freitag, Markus  and
      Firat, Orhan",
    editor = "Koehn, Philipp  and
      Haddow, Barry  and
      Kocmi, Tom  and
      Monz, Christof",
    booktitle = "Proceedings of the Eighth Conference on Machine Translation",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wmt-1.100/",
    doi = "10.18653/v1/2023.wmt-1.100",
    pages = "1066--1083",
}`
}
},
  {
  "id": 85,
  "title": "Empirical Assessment of kNN-MT for Real-World Translation Scenarios",
  "authors": "Pedro Henrique Martins, Jo\u00e3o Alves, T\u00e2nia Vaz, Madalena Gon\u00e7alves, Beatriz Silva, Marianna Buchicchio, Jos\u00e9 G. C. de Souza, Andr\u00e9 F. T. Martins",
  "venue": "EAMT",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>This paper aims to investigate the effectiveness of the k-Nearest Neighbor Machine Translation model (kNN-MT) in real-world scenarios. kNN-MT is a retrieval-augmented framework that combines the advantages of parametric models with non-parametric datastores built using a set of parallel sentences. Previous studies have primarily focused on evaluating the model using only the BLEU metric and have not tested kNN-MT in real world scenarios. Our study aims to fill this gap by conducting a comprehensive analysis on various datasets comprising different language pairs and different domains, using multiple automatic metrics and expert evaluated Multidimensional Quality Metrics (MQM). We compare kNN-MT with two alternate strategies: fine-tuning all the model parameters and adapter-based finetuning. Finally, we analyze the effect of the datastore size on translation quality, and we examine the number of entries necessary to bootstrap and configure the index.</p>`,
  "streams": [
  "multilingual-translation",
  "retrieval"
],
  "links": {
  "paper": "https://aclanthology.org/2023.eamt-1.12/",
  "bibtex": `@inproceedings{martins-etal-2023-empirical,
    title = "Empirical Assessment of k{NN}-{MT} for Real-World Translation Scenarios",
    author = "Martins, Pedro Henrique  and
      Alves, Jo{\\~a}o  and
      Vaz, T{\\^a}nia  and
      Gon{\\c{c}}alves, Madalena  and
      Silva, Beatriz  and
      Buchicchio, Marianna  and
      de Souza, Jos{\\'e} G. C.  and
      Martins, Andr{\\'e} F. T.",
    editor = "Nurminen, Mary  and
      Brenner, Judith  and
      Koponen, Maarit  and
      Latomaa, Sirkku  and
      Mikhailov, Mikhail  and
      Schierl, Frederike  and
      Ranasinghe, Tharindu  and
      Vanmassenhove, Eva  and
      Vidal, Sergi Alvarez  and
      Aranberri, Nora  and
      Nunziatini, Mara  and
      Escart{\\'i}n, Carla Parra  and
      Forcada, Mikel  and
      Popovic, Maja  and
      Scarton, Carolina  and
      Moniz, Helena",
    booktitle = "Proceedings of the 24th Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2023",
    address = "Tampere, Finland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2023.eamt-1.12/",
    pages = "115--124",
    abstract = "This paper aims to investigate the effectiveness of the k-Nearest Neighbor Machine Translation model (kNN-MT) in real-world scenarios. kNN-MT is a retrieval-augmented framework that combines the advantages of parametric models with non-parametric datastores built using a set of parallel sentences. Previous studies have primarily focused on evaluating the model using only the BLEU metric and have not tested kNN-MT in real world scenarios. Our study aims to fill this gap by conducting a comprehensive analysis on various datasets comprising different language pairs and different domains, using multiple automatic metrics and expert evaluated Multidimensional Quality Metrics (MQM). We compare kNN-MT with two alternate strategies: fine-tuning all the model parameters and adapter-based finetuning. Finally, we analyze the effect of the datastore size on translation quality, and we examine the number of entries necessary to bootstrap and configure the index."
}`
}
},
  {
  "id": 84,
  "title": "BLEU Meets COMET: Combining Lexical and Neural Metrics Towards Robust Machine Translation Evaluation",
  "authors": "Taisiya Glushkova, Chrysoula Zerva, Andr\u00e9 F. T. Martins",
  "venue": "EAMT",
  "year": 2023,
  "type": "preprint",
  "abstract": `<p>Although neural-based machine translation evaluation metrics, such as COMET or BLEURT, have achieved strong correlations with human judgements, they are sometimes unreliable in detecting certain phenomena that can be considered as critical errors, such as deviations in entities and numbers. In contrast, traditional evaluation metrics such as BLEU or chrF, which measure lexical or character overlap between translation hypotheses and human references, have lower correlations with human judgements but are sensitive to such deviations. In this paper, we investigate several ways of combining the two approaches in order to increase robustness of state-of-the-art evaluation methods to translations with critical errors. We show that by using additional information during training, such as sentence-level features and word-level tags, the trained metrics improve their capability to penalize translations with specific troublesome phenomena, which leads to gains in correlations with humans and on the recent DEMETR benchmark on several language pairs.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2023.eamt-1.6/",
  "bibtex": `@inproceedings{glushkova-etal-2023-bleu,
    title = "{BLEU} Meets {COMET}: Combining Lexical and Neural Metrics Towards Robust Machine Translation Evaluation",
    author = "Glushkova, Taisiya  and
      Zerva, Chrysoula  and
      Martins, Andr{\\'e} F. T.",
    editor = "Nurminen, Mary  and
      Brenner, Judith  and
      Koponen, Maarit  and
      Latomaa, Sirkku  and
      Mikhailov, Mikhail  and
      Schierl, Frederike  and
      Ranasinghe, Tharindu  and
      Vanmassenhove, Eva  and
      Vidal, Sergi Alvarez  and
      Aranberri, Nora  and
      Nunziatini, Mara  and
      Escart{\\'i}n, Carla Parra  and
      Forcada, Mikel  and
      Popovic, Maja  and
      Scarton, Carolina  and
      Moniz, Helena",
    booktitle = "Proceedings of the 24th Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2023",
    address = "Tampere, Finland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2023.eamt-1.6/",
    pages = "47--58",
    abstract = "Although neural-based machine translation evaluation metrics, such as COMET or BLEURT, have achieved strong correlations with human judgements, they are sometimes unreliable in detecting certain phenomena that can be considered as critical errors, such as deviations in entities and numbers. In contrast, traditional evaluation metrics such as BLEU or chrF, which measure lexical or character overlap between translation hypotheses and human references, have lower correlations with human judgements but are sensitive to such deviations. In this paper, we investigate several ways of combining the two approaches in order to increase robustness of state-of-the-art evaluation methods to translations with critical errors. We show that by using additional information during training, such as sentence-level features and word-level tags, the trained metrics improve their capability to penalize translations with specific troublesome phenomena, which leads to gains in correlations with humans and on the recent DEMETR benchmark on several language pairs."
}`
}
},
  {
  "id": 83,
  "title": "mPLM-Sim: Better Cross-Lingual Similarity and Transfer in Multilingual Pretrained Language Models",
  "authors": "Peiqin Lin, Chengzhi Hu, Zheyu Zhang, Andr\u00e9 F. T. Martins, Hinrich Sch\u00fctze",
  "venue": "EACL Findings",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Recent multilingual pretrained language models (mPLMs) have been shown to encode strong language-specific signals, which are not explicitly provided during pretraining. It remains an open question whether it is feasible to employ mPLMs to measure language similarity, and subsequently use the similarity results to select source languages for boosting cross-lingual transfer. To investigate this, we propose mPLM-Sim, a language similarity measure that induces the similarities across languages from mPLMs using multi-parallel corpora. Our study shows that mPLM-Sim exhibits moderately high correlations with linguistic similarity measures, such as lexicostatistics, genealogical language family, and geographical sprachbund. We also conduct a case study on languages with low correlation and observe that mPLM-Sim yields more accurate similarity results. Additionally, we find that similarity results vary across different mPLMs and different layers within an mPLM. We further investigate whether mPLM-Sim is effective for zero-shot cross-lingual transfer by conducting experiments on both low-level syntactic tasks and high-level semantic tasks. The experimental results demonstrate that mPLM-Sim is capable of selecting better source languages than linguistic measures, resulting in a 1%-2% improvement in zero-shot cross-lingual transfer performance.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2024.findings-eacl.20/",
  "bibtex": `@inproceedings{lin-etal-2024-mplm,
    title = "m{PLM}-Sim: Better Cross-Lingual Similarity and Transfer in Multilingual Pretrained Language Models",
    author = "Lin, Peiqin  and
      Hu, Chengzhi  and
      Zhang, Zheyu  and
      Martins, Andre  and
      Schuetze, Hinrich",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.20/",
    pages = "276--310",
    abstract = "Recent multilingual pretrained language models (mPLMs) have been shown to encode strong language-specific signals, which are not explicitly provided during pretraining. It remains an open question whether it is feasible to employ mPLMs to measure language similarity, and subsequently use the similarity results to select source languages for boosting cross-lingual transfer. To investigate this, we propose mPLM-Sim, a language similarity measure that induces the similarities across languages from mPLMs using multi-parallel corpora. Our study shows that mPLM-Sim exhibits moderately high correlations with linguistic similarity measures, such as lexicostatistics, genealogical language family, and geographical sprachbund. We also conduct a case study on languages with low correlation and observe that mPLM-Sim yields more accurate similarity results. Additionally, we find that similarity results vary across different mPLMs and different layers within an mPLM. We further investigate whether mPLM-Sim is effective for zero-shot cross-lingual transfer by conducting experiments on both low-level syntactic tasks and high-level semantic tasks. The experimental results demonstrate that mPLM-Sim is capable of selecting better source languages than linguistic measures, resulting in a 1{\\%}-2{\\%} improvement in zero-shot cross-lingual transfer performance."
}`
}
},
  {
  "id": 82,
  "title": "Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages",
  "authors": "Ayyoob Imani, Peiqin Lin, Amir Hossein Kargaran, Silvia Severini, Masoud Jalili Sabet, Nora Kassner, Chunlan Ma, Helmut Schmid, Andr\u00e9 Martins, Fran\u00e7ois Yvon, Hinrich Sch\u00fctze",
  "venue": "ACL",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>The NLP community has mainly focused on scaling Large Language Models (LLMs) vertically, i.e., making them better for about 100 languages. We instead scale LLMs horizontally: we create, through continued pretraining, Glot500-m, an LLM that covers 511 predominantly low-resource languages. An important part of this effort is to collect and clean Glot500-c, a corpus that covers these 511 languages and allows us to train Glot500-m. We evaluate Glot500-m on five diverse tasks across these languages. We observe large improvements for both high-resource and low-resource languages compared to an XLM-R baseline. Our analysis shows that no single factor explains the quality of multilingual LLM representations. Rather, a combination of factors determines quality including corpus size, script, “help” from related languages and the total capacity of the model. Our work addresses an important goal of NLP research: we should notlimit NLP to a small fraction of the world’s languages and instead strive to support as many languages as possible to bring the benefits of NLP technology to all languages and cultures. Code, data and models are available at https://github.com/cisnlp/Glot500.</p>`,
  "streams": [
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2023.acl-long.61/",
  "bibtex": `@inproceedings{imanigooghari-etal-2023-glot500,
    title = "Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages",
    author = {Imani, Ayyoob  and
      Lin, Peiqin  and
      Kargaran, Amir Hossein  and
      Severini, Silvia  and
      Jalili Sabet, Masoud  and
      Kassner, Nora  and
      Ma, Chunlan  and
      Schmid, Helmut  and
      Martins, Andr{\\'e}  and
      Yvon, Fran{\\c{c}}ois  and
      Sch{\\"u}tze, Hinrich},
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.61/",
    doi = "10.18653/v1/2023.acl-long.61",
    pages = "1082--1117",
    abstract = "The NLP community has mainly focused on scaling Large Language Models (LLMs) vertically, i.e., making them better for about 100 languages. We instead scale LLMs horizontally: we create, through continued pretraining, Glot500-m, an LLM that covers 511 predominantly low-resource languages. An important part of this effort is to collect and clean Glot500-c, a corpus that covers these 511 languages and allows us to train Glot500-m. We evaluate Glot500-m on five diverse tasks across these languages. We observe large improvements for both high-resource and low-resource languages compared to an XLM-R baseline. Our analysis shows that no single factor explains the quality of multilingual LLM representations. Rather, a combination of factors determines quality including corpus size, script, \`\`help'' from related languages and the total capacity of the model. Our work addresses an important goal of NLP research: we should notlimit NLP to a small fraction of the world{'}s languages and instead strive to support as many languages as possible to bring the benefits of NLP technology to all languages and cultures. Code, data and models are available at \\url{https://github.com/cisnlp/Glot500}."
}`
}
},
  {
  "id": 81,
  "title": "Sparse modern hopfield networks",
  "authors": "Andre Martins, Vlad Niculae, Daniel C McNamee",
  "venue": "AMHN",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Ramsauer et al. (2021) recently pointed out a connection between modern Hopfield networks and attention heads in transformers. In this paper, we extend their framework to a broader family of energy functions which can be written as a difference of a quadratic regularizer and a Fenchel-Young loss (Blondel et al., 2020), parametrized by a generalized negentropy function. By working with Tsallis negentropies, the resulting update rules become end-to-end differentiable sparse transformations, establishing a new link to adaptively sparse transformers (Correia et al., 2019) and allowing for exact convergence to single memory patterns. Experiments on simulated data show a higher tendency to avoid metastable states.</p>`,
  "streams": [
  "memory",
  "theory"
],
  "links": {
  "paper": "https://openreview.net/forum?id=zwqlV7HoaT",
  "bibtex": `@inproceedings{
martins2023sparse,
title={Sparse Modern Hopfield Networks},
author={Andre Martins and Vlad Niculae and Daniel C McNamee},
booktitle={Associative Memory {\\&} Hopfield Networks in 2023},
year={2023},
url={https://openreview.net/forum?id=zwqlV7HoaT}
}`
}
},
  {
  "id": 80,
  "title": "Findings of the WMT 2023 Shared Task on Quality Estimation",
  "authors": "Frederic Blain, Chrysoula Zerva, Ricardo Ribeiro, Nuno M. Guerreiro, Diptesh Kanojia, Jos\u00e9 G. C. de Souza, Beatriz Silva, T\u00e2nia Vaz, Yan Jingxuan, Fatemeh Azadi, Constantin Orasan, Andr\u00e9 Martins",
  "venue": "WMT",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>We report the results of the WMT 2023 shared task on Quality Estimation, in which the challenge is to predict the quality of the output of neural machine translation systems at the word and sentence levels, without access to reference translations. This edition introduces a few novel aspects and extensions that aim to enable more fine-grained, and explainable quality estimation approaches. We introduce an updated quality annotation scheme using Multidimensional Quality Metrics to obtain sentence- and word-level quality scores for three language pairs. We also extend the provided data to new language pairs: we specifically target low-resource languages and provide training, development and test data for English-Hindi, English-Tamil, English-Telegu and English-Gujarati as well as a zero-shot test-set for English-Farsi. Further, we introduce a novel fine-grained error prediction task aspiring to motivate research towards more detailed quality predictions.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2023.wmt-1.52/",
  "bibtex": `@inproceedings{blain-etal-2023-findings,
    title = "Findings of the {WMT} 2023 Shared Task on Quality Estimation",
    author = "Blain, Frederic  and
      Zerva, Chrysoula  and
      Rei, Ricardo  and
      Guerreiro, Nuno M.  and
      Kanojia, Diptesh  and
      C. de Souza, Jos{\\'e} G.  and
      Silva, Beatriz  and
      Vaz, T{\\^a}nia  and
      Jingxuan, Yan  and
      Azadi, Fatemeh  and
      Orasan, Constantin  and
      Martins, Andr{\\'e}",
    editor = "Koehn, Philipp  and
      Haddow, Barry  and
      Kocmi, Tom  and
      Monz, Christof",
    booktitle = "Proceedings of the Eighth Conference on Machine Translation",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wmt-1.52/",
    doi = "10.18653/v1/2023.wmt-1.52",
    pages = "629--653",
    abstract = "We report the results of the WMT 2023 shared task on Quality Estimation, in which the challenge is to predict the quality of the output of neural machine translation systems at the word and sentence levels, without access to reference translations. This edition introduces a few novel aspects and extensions that aim to enable more fine-grained, and explainable quality estimation approaches. We introduce an updated quality annotation scheme using Multidimensional Quality Metrics to obtain sentence- and word-level quality scores for three language pairs. We also extend the provided data to new language pairs: we specifically target low-resource languages and provide training, development and test data for English-Hindi, English-Tamil, English-Telegu and English-Gujarati as well as a zero-shot test-set for English-Farsi. Further, we introduce a novel fine-grained error prediction task aspiring to motivate research towards more detailed quality predictions."
}`
}
},
  {
  "id": 79,
  "title": "When Does Translation Require Context? A Data-driven, Multilingual Exploration",
  "authors": "Patrick Fernandes, Kayo Yin, Emmy Liu, Andr\u00e9 Martins, Graham Neubig",
  "venue": "ACL",
  "year": 2023,
  "type": "conference",
  "abstract": `<p>Although proper handling of discourse significantly contributes to the quality of machine translation (MT), these improvements are not adequately measured in common translation quality metrics. Recent works in context-aware MT attempt to target a small set of discourse phenomena during evaluation, however not in a fully systematic way. In this paper, we develop the Multilingual Discourse-Aware (MuDA) benchmark, a series of taggers that identify and evaluate model performance on discourse phenomena in any given dataset. The choice of phenomena is inspired by a novel methodology to systematically identify translations that require context. This methodology confirms the difficulty of previously studied phenomena while uncovering others which were not previously addressed. We find that commonly studied context-aware MT models make only marginal improvements over context-agnostic models, which suggests these models do not handle these ambiguities effectively. We release code and data for 14 language pairs to encourage the MT community to focus on accurately capturing discourse phenomena. Code available at https://github.com/neulab/contextual-mt</p>`,
  "streams": [
  "dialogue-context",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2023.acl-long.36/",
  "bibtex": `@inproceedings{fernandes-etal-2023-translation,
    title = "When Does Translation Require Context? A Data-driven, Multilingual Exploration",
    author = "Fernandes, Patrick  and
      Yin, Kayo  and
      Liu, Emmy  and
      Martins, Andr{\\'e}  and
      Neubig, Graham",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.36/",
    doi = "10.18653/v1/2023.acl-long.36",
    pages = "606--626",
    abstract = "Although proper handling of discourse significantly contributes to the quality of machine translation (MT), these improvements are not adequately measured in common translation quality metrics. Recent works in context-aware MT attempt to target a small set of discourse phenomena during evaluation, however not in a fully systematic way. In this paper, we develop the Multilingual Discourse-Aware (MuDA) benchmark, a series of taggers that identify and evaluate model performance on discourse phenomena in any given dataset. The choice of phenomena is inspired by a novel methodology to systematically identify translations that require context. This methodology confirms the difficulty of previously studied phenomena while uncovering others which were not previously addressed. We find that commonly studied context-aware MT models make only marginal improvements over context-agnostic models, which suggests these models do not handle these ambiguities effectively. We release code and data for 14 language pairs to encourage the MT community to focus on accurately capturing discourse phenomena. Code available at \\url{https://github.com/neulab/contextual-mt}"
}`
}
},
  {
  "id": 78,
  "title": "CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task",
  "authors": "Ricardo Rei, Marcos Treviso, Ricardo Rei, Chrysoula Zerva, Ana C Farinha, Christine Maroti, Jos\u00e9 G. C. de Souza, Taisiya Glushkova, Duarte M. Alves, Alon Lavie, Lu\u00edsa Coheur, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>We present the joint contribution of IST and Unbabel to the WMT 2022 Shared Task on Quality Estimation (QE). Our team participated in all three subtasks: (i) Sentence and Word-level Quality Prediction; (ii) Explainable QE; and (iii) Critical Error Detection. For all tasks we build on top of the COMET framework, connecting it with the predictor-estimator architecture of OpenKiwi, and equipping it with a word-level sequence tagger and an explanation extractor. Our results suggest that incorporating references during pretraining improves performance across several language pairs on downstream tasks, and that jointly training with sentence and word-level objectives yields a further boost. Furthermore, combining attention and gradient information proved to be the top strategy for extracting good explanations of sentence-level QE models. Overall, our submissions achieved the best results for all three tasks for almost all language pairs by a considerable margin.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2022.wmt-1.60/",
  "bibtex": `@inproceedings{rei-etal-2022-cometkiwi,
    title = "{C}omet{K}iwi: {IST}-Unbabel 2022 Submission for the Quality Estimation Shared Task",
    author = "Rei, Ricardo  and
      Treviso, Marcos  and
      Guerreiro, Nuno M.  and
      Zerva, Chrysoula  and
      Farinha, Ana C  and
      Maroti, Christine  and
      C. de Souza, Jos{\\'e} G.  and
      Glushkova, Taisiya  and
      Alves, Duarte  and
      Coheur, Luisa  and
      Lavie, Alon  and
      Martins, Andr{\\'e} F. T.",
    editor = {Koehn, Philipp  and
      Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Jimeno Yepes, Antonio  and
      Kocmi, Tom  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Popel, Martin  and
      Turchi, Marco  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.60/",
    pages = "634--645",
    abstract = "We present the joint contribution of IST and Unbabel to the WMT 2022 Shared Task on Quality Estimation (QE). Our team participated in all three subtasks: (i) Sentence and Word-level Quality Prediction; (ii) Explainable QE; and (iii) Critical Error Detection. For all tasks we build on top of the COMET framework, connecting it with the predictor-estimator architecture of OpenKiwi, and equipping it with a word-level sequence tagger and an explanation extractor. Our results suggest that incorporating references during pretraining improves performance across several language pairs on downstream tasks, and that jointly training with sentence and word-level objectives yields a further boost. Furthermore, combining attention and gradient information proved to be the top strategy for extracting good explanations of sentence-level QE models. Overall, our submissions achieved the best results for all three tasks for almost all language pairs by a considerable margin."
}`
}
},
  {
  "id": 77,
  "title": "Learning to Scaffold: Optimizing Model Explanations for Teaching",
  "authors": "Patrick Fernandes, Marcos Treviso, Danish Pruthi, Andr\u00e9 F. T. Martins, Graham Neubig",
  "venue": "NeurIPS",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Modern machine learning models are opaque, and as a result there is a burgeoning academic subfield on methods that explain these models' behavior. However, what is the precise goal of providing such explanations, and how can we demonstrate that explanations achieve this goal? Some research argues that explanations should help teach a student (either human or machine) to simulate the model being explained, and that the quality of explanations can be measured by the simulation accuracy of students on unexplained examples. In this work, leveraging meta-learning techniques, we extend this idea to improve the quality of the explanations themselves, specifically by optimizing explanations such that student models more effectively learn to simulate the original model. We train models on three natural language processing and computer vision tasks, and find that students trained with explanations extracted with our framework are able to simulate the teacher significantly more effectively than ones produced with previous methods. Through human annotations and a user study, we further find that these learned explanations more closely align with how humans would explain the required decisions in these tasks. Our code is available at <a href="https://github.com/coderpat/learning-scaffold">[https://github.com/coderpat/learning-scaffold](https://github.com/coderpat/learning-scaffold)</a></p>`,
  "streams": [
  "interpretability",
  "resources",
  "theory"
],
  "links": {
  "paper": "https://openreview.net/forum?id=V5rlSPsHpkf",
  "code": "https://github.com/CoderPat/learning-scaffold",
  "bibtex": `@inproceedings{
fernandes2022learning,
title={Learning to Scaffold: Optimizing Model Explanations for Teaching},
author={Patrick Fernandes and Marcos Vinicius Treviso and Danish Pruthi and Andre Martins and Graham Neubig},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=V5rlSPsHpkf}
}`
}
},
  {
  "id": 76,
  "title": "Sparse continuous distributions and Fenchel-Young losses",
  "authors": "Andr\u00e9 F. T. Martins, Marcos Treviso, Ant\u00f3nio Farinhas, Pedro M. Q. Aguiar, M\u00e1rio A. T. Figueiredo, Mathieu Blondel, Vlad Niculae",
  "venue": "JMLR",
  "year": 2022,
  "type": "journal",
  "abstract": `<p>Exponential families are widely used in machine learning, including many distributions in continuous and discrete domains (e.g., Gaussian, Dirichlet, Poisson, and categorical distributions via the softmax transformation). Distributions in each of these families have fixed support. In contrast, for finite domains, recent work on sparse alternatives to softmax (e.g., sparsemax, $\\alpha$-entmax, and fusedmax), has led to distributions with varying support. This paper develops sparse alternatives to continuous distributions, based on several technical contributions: First, we define $Ømega$-regularized prediction maps and Fenchel-Young losses for arbitrary domains (possibly countably infinite or continuous). For linearly parametrized families, we show that minimization of Fenchel-Young losses is equivalent to moment matching of the statistics, generalizing a fundamental property of exponential families. When $Ømega$ is a Tsallis negentropy with parameter $\\alpha$, we obtain \`\`deformed exponential families,'' which include $\\alpha$-entmax and sparsemax ($\\alpha=2$) as particular cases. For quadratic energy functions, the resulting densities are $\\beta$-Gaussians, an instance of elliptical distributions that contain as particular cases the Gaussian, biweight, triweight, and Epanechnikov densities, and for which we derive closed-form expressions for the variance, Tsallis entropy, and Fenchel-Young loss. When $Ømega$ is a total variation or Sobolev regularizer, we obtain a continuous version of the fusedmax. Finally, we introduce continuous-domain attention mechanisms, deriving efficient gradient backpropagation algorithms for $\\alpha ın {1, 4/3, 3/2, 2}$. Using these algorithms, we demonstrate our sparse continuous distributions for attention-based audio classification and visual question answering, showing that they allow attending to time intervals and compact regions.</p>`,
  "streams": [
  "attention",
  "theory"
],
  "links": {
  "paper": "https://www.jmlr.org/papers/v23/21-0879.html",
  "code": "https://github.com/deep-spin/sparse_continuous_distributions",
  "bibtex": `@article{martins2022sparse,
  title={Sparse continuous distributions and Fenchel-Young losses},
  author={Martins, Andr{\\'e} FT and Treviso, Marcos and Farinhas, Ant{\\'o}nio and Aguiar, Pedro MQ and Figueiredo, M{\\'a}rio AT and Blondel, Mathieu and Niculae, Vlad},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={257},
  pages={1--74},
  year={2022}
}`
}
},
  {
  "id": 75,
  "title": "A Framework to Semi-automated Usability Evaluations Processing Considering Users\u2019 Emotional Aspects",
  "authors": "Fl\u00e1via de Souza Santos, Marcos Vin\u00edcius Treviso, Sandra Pereira Gama, Renata Pontin de Mattos Fortes",
  "venue": "HCII",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>The concern with providing a good experience for users increases simultaneously with technological evolution and dissemination. Different evaluation methods are presented in the literature to help developers evaluate and verify the interfaces they develop. However, for the application of evaluation methods, it is often necessary to have an expert, or users, which can make the assessment costly, both time-consumingly and monetarily. Consequently, developers may launch their products without carefully checking some critical aspects beforehand, causing anything from the non-acceptance of the technology to even its abandonment. Carrying out part of the evaluations with users, or totally with them, in an automated way, considering the diversity of data obtained and the support of analyzes processed by a computer, presents itself as a possible alternative to be investigated to support developers of interactive systems. At the same time, considering the emotional aspects during the interaction can provide the developer with valuable information, including the acceptance of the technology. In this way, considering users’ emotions can improve the automatic evaluation results, capturing and processing the user experience in the system. Thus, we present a framework (EmotiUsing) composed of semi-automated usability evaluations, considering the emotional aspects of users. The framework aims to make the analytical evaluations less subjective and streamline the evaluation process, reducing costs and time for the evaluators.</p>`,
  "streams": [
  "interpretability",
  "resources"
],
  "links": {
  "paper": "https://link.springer.com/chapter/10.1007/978-3-031-05311-5_29",
  "bibtex": `@inproceedings{de2022framework,
  title={A framework to semi-automated usability evaluations processing considering users’ emotional aspects},
  author={de Souza Santos, Fl{\\'a}via and Vin{\\'\\i}cius Treviso, Marcos and Gama, Sandra Pereira and de Mattos Fortes, Renata Pontin},
  booktitle={International conference on human-computer interaction},
  pages={419--438},
  year={2022},
  organization={Springer}
}`
}
},
  {
  "id": 74,
  "title": "DeepSPIN: Deep Structured Prediction for Natural Language Processing",
  "authors": "Andr\u00e9 F. T. Martins, Ben Peters, Chrysoula Zerva, Chunchuan Lyu, Gon\u00e7alo Correia, Marcos Treviso, Pedro Martins, Tsvetomila Mihaylova",
  "venue": "EAMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>DeepSPIN is a research project funded by the European Research Council (ERC) whose goal is to develop new neural structured prediction methods, models, and algorithms for improving the quality, interpretability, and data-efficiency of natural language processing (NLP) systems, with special emphasis on machine translation and quality estimation applications.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2022.eamt-1.53/",
  "bibtex": `@inproceedings{martins-etal-2022-deepspin,
    title = "{D}eep{SPIN}: Deep Structured Prediction for Natural Language Processing",
    author = "Martins, Andr{\\'e} F. T.  and
      Peters, Ben  and
      Zerva, Chrysoula  and
      Lyu, Chunchuan  and
      Correia, Gon{\\c{c}}alo  and
      Treviso, Marcos  and
      Martins, Pedro  and
      Mihaylova, Tsvetomila",
    editor = {Moniz, Helena  and
      Macken, Lieve  and
      Rufener, Andrew  and
      Barrault, Lo{\\"i}c  and
      Costa-juss{\\\`a}, Marta R.  and
      Declercq, Christophe  and
      Koponen, Maarit  and
      Kemp, Ellie  and
      Pilos, Spyridon  and
      Forcada, Mikel L.  and
      Scarton, Carolina  and
      Van den Bogaert, Joachim  and
      Daems, Joke  and
      Tezcan, Arda  and
      Vanroy, Bram  and
      Fonteyne, Margot},
    booktitle = "Proceedings of the 23rd Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2022",
    address = "Ghent, Belgium",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2022.eamt-1.53/",
    pages = "327--328",
    abstract = "DeepSPIN is a research project funded by the European Research Council (ERC) whose goal is to develop new neural structured prediction methods, models, and algorithms for improving the quality, interpretability, and data-efficiency of natural language processing (NLP) systems, with special emphasis on machine translation and quality estimation. We describe in this paper the latest findings from this project."
}`
}
},
  {
  "id": 73,
  "title": "Haau-Sing (Xiaocheng) Li, Mohsen Mesgar, Andr\u00e9 Martins, Iryna Gurevych",
  "authors": "Haau-Sing (Xiaocheng) Li, Mohsen Mesgar, Andr\u00e9 Martins, Iryna Gurevych",
  "venue": "ACL",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Code generation from text requires understanding the user’s intent from a natural languagedescription and generating an executable code snippet that satisfies this intent. While recent pretrained language models demonstrate remarkable performance for this task, these models fail when the given natural language description is under-specified. In this work, we introduce a novel and more realistic setup for this task. We hypothesize that the under-specification of a natural language description can be resolved by asking clarification questions. Therefore, we collect and introduce a new dataset named CodeClarQA containing pairs of natural language descriptions and code with created synthetic clarification questions and answers. The empirical results of our evaluation of pretrained language model performance on code generation show that clarifications result in more precisely generated code, as shown by the substantial improvement of model performance in all evaluation metrics. Alongside this, our task and dataset introduce new challenges to the community, including when and what clarification questions should be asked. Our code and dataset are available on GitHub.</p>`,
  "streams": [
  "code-generation"
],
  "links": {
  "paper": "https://aclanthology.org/2023.acl-long.799/",
  "bibtex": `@inproceedings{li-etal-2023-python,
    title = "Python Code Generation by Asking Clarification Questions",
    author = "Li, Haau-Sing (Xiaocheng)  and
      Mesgar, Mohsen  and
      Martins, Andr{\\'e}  and
      Gurevych, Iryna",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.799/",
    doi = "10.18653/v1/2023.acl-long.799",
    pages = "14287--14306",
    abstract = "Code generation from text requires understanding the user{'}s intent from a natural languagedescription and generating an executable code snippet that satisfies this intent. While recent pretrained language models demonstrate remarkable performance for this task, these models fail when the given natural language description is under-specified. In this work, we introduce a novel and more realistic setup for this task. We hypothesize that the under-specification of a natural language description can be resolved by asking clarification questions. Therefore, we collect and introduce a new dataset named CodeClarQA containing pairs of natural language descriptions and code with created synthetic clarification questions and answers. The empirical results of our evaluation of pretrained language model performance on code generation show that clarifications result in more precisely generated code, as shown by the substantial improvement of model performance in all evaluation metrics. Alongside this, our task and dataset introduce new challenges to the community, including when and what clarification questions should be asked. Our code and dataset are available on GitHub."
}`
}
},
  {
  "id": 72,
  "title": "Optimal Transport for Unsupervised Hallucination Detection in Neural Machine Translation",
  "authors": "Nuno M. Guerreiro, Pierre Colombo, Pablo Piantanida, Andr\u00e9 Martins",
  "venue": "ACL",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Neural machine translation (NMT) has become the de-facto standard in real-world machine translation applications. However, NMT models can unpredictably produce severely pathological translations, known as hallucinations, that seriously undermine user trust. It becomes thus crucial to implement effective preventive strategies to guarantee their proper functioning. In this paper, we address the problem of hallucination detection in NMT by following a simple intuition: as hallucinations are detached from the source content, they exhibit encoder-decoder attention patterns that are statistically different from those of good quality translations. We frame this problem with an optimal transport formulation and propose a fully unsupervised, plug-in detector that can be used with any attention-based NMT model. Experimental results show that our detector not only outperforms all previous model-based detectors, but is also competitive with detectors that employ external models trained on millions of samples for related tasks such as quality estimation and cross-lingual sentence similarity.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2023.acl-long.770/",
  "bibtex": `@inproceedings{guerreiro-etal-2023-optimal,
    title = "Optimal Transport for Unsupervised Hallucination Detection in Neural Machine Translation",
    author = "Guerreiro, Nuno M.  and
      Colombo, Pierre  and
      Piantanida, Pablo  and
      Martins, Andr{\\'e}",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.770/",
    doi = "10.18653/v1/2023.acl-long.770",
    pages = "13766--13784",
    abstract = "Neural machine translation (NMT) has become the de-facto standard in real-world machine translation applications. However, NMT models can unpredictably produce severely pathological translations, known as hallucinations, that seriously undermine user trust. It becomes thus crucial to implement effective preventive strategies to guarantee their proper functioning. In this paper, we address the problem of hallucination detection in NMT by following a simple intuition: as hallucinations are detached from the source content, they exhibit encoder-decoder attention patterns that are statistically different from those of good quality translations. We frame this problem with an optimal transport formulation and propose a fully unsupervised, plug-in detector that can be used with any attention-based NMT model. Experimental results show that our detector not only outperforms all previous model-based detectors, but is also competitive with detectors that employ external models trained on millions of samples for related tasks such as quality estimation and cross-lingual sentence similarity."
}`
}
},
  {
  "id": 71,
  "title": "COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task",
  "authors": "Ricardo Rei, Jos\u00e9 G. C. de Souza, Duarte Alves, Chrysoula Zerva, Ana C Farinha, Taisiya Glushkova, Alon Lavie, Luisa Coheur, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>In this paper, we present the joint contribution of Unbabel and IST to the WMT 2022 Metrics Shared Task. Our primary submission – dubbed COMET-22 – is an ensemble between a COMET estimator model trained with Direct Assessments and a newly proposed multitask model trained to predict sentence-level scores along with OK/BAD word-level tags derived from Multidimensional Quality Metrics error annotations. These models are ensembled together using a hyper-parameter search that weights different features extracted from both evaluation models and combines them into a single score. For the reference-free evaluation, we present CometKiwi. Similarly to our primary submission, CometKiwi is an ensemble between two models. A traditional predictor-estimator model inspired by OpenKiwi and our new multitask model trained on Multidimensional Quality Metrics which can also be used without references. Both our submissions show improved correlations compared to state-of-the-art metrics from last year as well as increased robustness to critical errors.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2022.wmt-1.52/",
  "bibtex": `@inproceedings{rei-etal-2022-comet,
    title = "{COMET}-22: Unbabel-{IST} 2022 Submission for the Metrics Shared Task",
    author = "Rei, Ricardo  and
      C. de Souza, Jos{\\'e} G.  and
      Alves, Duarte  and
      Zerva, Chrysoula  and
      Farinha, Ana C  and
      Glushkova, Taisiya  and
      Lavie, Alon  and
      Coheur, Luisa  and
      Martins, Andr{\\'e} F. T.",
    editor = {Koehn, Philipp  and
      Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Jimeno Yepes, Antonio  and
      Kocmi, Tom  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Popel, Martin  and
      Turchi, Marco  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.52/",
    pages = "578--585",
    abstract = "In this paper, we present the joint contribution of Unbabel and IST to the WMT 2022 Metrics Shared Task. Our primary submission {--} dubbed COMET-22 {--} is an ensemble between a COMET estimator model trained with Direct Assessments and a newly proposed multitask model trained to predict sentence-level scores along with OK/BAD word-level tags derived from Multidimensional Quality Metrics error annotations. These models are ensembled together using a hyper-parameter search that weights different features extracted from both evaluation models and combines them into a single score. For the reference-free evaluation, we present CometKiwi. Similarly to our primary submission, CometKiwi is an ensemble between two models. A traditional predictor-estimator model inspired by OpenKiwi and our new multitask model trained on Multidimensional Quality Metrics which can also be used without references. Both our submissions show improved correlations compared to state-of-the-art metrics from last year as well as increased robustness to critical errors."
}`
}
},
  {
  "id": 70,
  "title": "Findings of the WMT 2022 Shared Task on Chat Translation",
  "authors": "Ana C Farinha, M. Amin Farajian, Marianna Buchicchio, Patrick Fernandes, Jos\u00e9 G. C. de Souza, Helena Moniz, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>This paper reports the findings of the second edition of the Chat Translation Shared Task. Similarly to the previous WMT 2020 edition, the task consisted of translating bilingual customer support conversational text. However, unlike the previous edition, in which the bilingual data was created from a synthetic monolingual English corpus, this year we used a portion of the newly released Unbabel’s MAIA corpus, which contains genuine bilingual conversations between agents and customers. We also expanded the language pairs to English↔German (en↔de), English↔French (en↔fr), and English↔Brazilian Portuguese (en↔pt-br).Given that the main goal of the shared task is to translate bilingual conversations, participants were encouraged to train and test their models specifically for this environment. In total, we received 18 submissions from 4 different teams. All teams participated in both directions of en↔de. One of the teams also participated in en↔fr and en↔pt-br. We evaluated the submissions with automatic metrics as well as human judgments via Multidimensional Quality Metrics (MQM) on both directions. The official ranking of the systems is based on the overall MQM scores of the participating systems on both directions, i.e. agent and customer.</p>`,
  "streams": [
  "dialogue-context",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2022.wmt-1.70/",
  "bibtex": `@inproceedings{farinha-etal-2022-findings,
    title = "Findings of the {WMT} 2022 Shared Task on Chat Translation",
    author = "Farinha, Ana C  and
      Farajian, M. Amin  and
      Buchicchio, Marianna  and
      Fernandes, Patrick  and
      C. de Souza, Jos{\\'e} G.  and
      Moniz, Helena  and
      Martins, Andr{\\'e} F. T.",
    editor = {Koehn, Philipp  and
      Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Jimeno Yepes, Antonio  and
      Kocmi, Tom  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Popel, Martin  and
      Turchi, Marco  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.70/",
    pages = "724--743",
    abstract = "This paper reports the findings of the second edition of the Chat Translation Shared Task. Similarly to the previous WMT 2020 edition, the task consisted of translating bilingual customer support conversational text. However, unlike the previous edition, in which the bilingual data was created from a synthetic monolingual English corpus, this year we used a portion of the newly released Unbabel{'}s MAIA corpus, which contains genuine bilingual conversations between agents and customers. We also expanded the language pairs to English{\\ensuremath{\\leftrightarrow}}German (en{\\ensuremath{\\leftrightarrow}}de), English{\\ensuremath{\\leftrightarrow}}French (en{\\ensuremath{\\leftrightarrow}}fr), and English{\\ensuremath{\\leftrightarrow}}Brazilian Portuguese (en{\\ensuremath{\\leftrightarrow}}pt-br).Given that the main goal of the shared task is to translate bilingual conversations, participants were encouraged to train and test their models specifically for this environment. In total, we received 18 submissions from 4 different teams. All teams participated in both directions of en{\\ensuremath{\\leftrightarrow}}de. One of the teams also participated in en{\\ensuremath{\\leftrightarrow}}fr and en{\\ensuremath{\\leftrightarrow}}pt-br. We evaluated the submissions with automatic metrics as well as human judgments via Multidimensional Quality Metrics (MQM) on both directions. The official ranking of the systems is based on the overall MQM scores of the participating systems on both directions, i.e. agent and customer."
}`
}
},
  {
  "id": 69,
  "title": "Unbabel-IST at the WMT Chat Translation Shared Task",
  "authors": "Jo\u00e3o Alves, Pedro Henrique Martins, Jos\u00e9 G. C. de Souza, M. Amin Farajian, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>We present the joint contribution of IST and Unbabel to the WMT 2022 Chat Translation Shared Task. We participated in all six language directions (English ↔ German, English ↔ French, English ↔ Brazilian Portuguese). Due to the lack of domain-specific data, we use mBART50, a large pretrained language model trained on millions of sentence-pairs, as our base model. We fine-tune it using a two step fine-tuning process. In the first step, we fine-tune the model on publicly available data. In the second step, we use the validation set. After having a domain specific model, we explore the use of kNN-MT as a way of incorporating domain-specific data at decoding time.</p>`,
  "streams": [
  "dialogue-context",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2022.wmt-1.89/",
  "bibtex": `@inproceedings{alves-etal-2022-unbabel,
    title = "Unbabel-{IST} at the {WMT} Chat Translation Shared Task",
    author = "Alves, Jo{\\~a}o  and
      Martins, Pedro Henrique  and
      C. de Souza, Jos{\\'e} G.  and
      Farajian, M. Amin  and
      Martins, Andr{\\'e} F. T.",
    editor = {Koehn, Philipp  and
      Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Jimeno Yepes, Antonio  and
      Kocmi, Tom  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Popel, Martin  and
      Turchi, Marco  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.89/",
    pages = "943--948",
    abstract = "We present the joint contribution of IST and Unbabel to the WMT 2022 Chat Translation Shared Task. We participated in all six language directions (English {\\ensuremath{\\leftrightarrow}} German, English {\\ensuremath{\\leftrightarrow}} French, English {\\ensuremath{\\leftrightarrow}} Brazilian Portuguese). Due to the lack of domain-specific data, we use mBART50, a large pretrained language model trained on millions of sentence-pairs, as our base model. We fine-tune it using a two step fine-tuning process. In the first step, we fine-tune the model on publicly available data. In the second step, we use the validation set. After having a domain specific model, we explore the use of kNN-MT as a way of incorporating domain-specific data at decoding time."
}`
}
},
  {
  "id": 68,
  "title": "Results of WMT22 Metrics Shared Task: Stop Using BLEU \u2013 Neural Metrics Are Better and More Robust",
  "authors": "Markus Freitag, Ricardo Rei, Nitika Mathur, Chi-kiu Lo, Craig Stewart, Eleftherios Avramidis, Tom Kocmi, George Foster, Alon Lavie, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>This paper presents the results of the WMT22 Metrics Shared Task. Participants submitting automatic MT evaluation metrics were asked to score the outputs of the translation systems competing in the WMT22 News Translation Task on four different domains: news, social, ecommerce, and chat. All metrics were evaluated on how well they correlate with human ratings at the system and segment level. Similar to last year, we acquired our own human ratings based on expert-based human evaluation via Multidimensional Quality Metrics (MQM). This setup had several advantages, among other things: (i) expert-based evaluation is more reliable, (ii) we extended the pool of translations by 5 additional translations based on MBR decoding or rescoring which are challenging for current metrics. In addition, we initiated a challenge set subtask, where participants had to create contrastive test suites for evaluating metrics’ ability to capture and penalise specific types of translation errors. Finally, we present an extensive analysis on how well metrics perform on three language pairs: English to German, English to Russian and Chinese to English. The results demonstrate the superiority of neural-based learned metrics and demonstrate again that overlap metrics like Bleu, spBleu or chrf correlate poorly with human ratings. The results also reveal that neural-based metrics are remarkably robust across different domains and challenges.</p>`,
  "streams": [
  "evaluation-metrics",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2022.wmt-1.2/",
  "bibtex": `@inproceedings{freitag-etal-2022-results,
    title = "Results of {WMT}22 Metrics Shared Task: Stop Using {BLEU} {--} Neural Metrics Are Better and More Robust",
    author = "Freitag, Markus  and
      Rei, Ricardo  and
      Mathur, Nitika  and
      Lo, Chi-kiu  and
      Stewart, Craig  and
      Avramidis, Eleftherios  and
      Kocmi, Tom  and
      Foster, George  and
      Lavie, Alon  and
      Martins, Andr{\\'e} F. T.",
    editor = {Koehn, Philipp  and
      Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Jimeno Yepes, Antonio  and
      Kocmi, Tom  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Popel, Martin  and
      Turchi, Marco  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.2/",
    pages = "46--68",
    abstract = "This paper presents the results of the WMT22 Metrics Shared Task. Participants submitting automatic MT evaluation metrics were asked to score the outputs of the translation systems competing in the WMT22 News Translation Task on four different domains: news, social, ecommerce, and chat. All metrics were evaluated on how well they correlate with human ratings at the system and segment level. Similar to last year, we acquired our own human ratings based on expert-based human evaluation via Multidimensional Quality Metrics (MQM). This setup had several advantages, among other things: (i) expert-based evaluation is more reliable, (ii) we extended the pool of translations by 5 additional translations based on MBR decoding or rescoring which are challenging for current metrics. In addition, we initiated a challenge set subtask, where participants had to create contrastive test suites for evaluating metrics' ability to capture and penalise specific types of translation errors. Finally, we present an extensive analysis on how well metrics perform on three language pairs: English to German, English to Russian and Chinese to English. The results demonstrate the superiority of neural-based learned metrics and demonstrate again that overlap metrics like Bleu, spBleu or chrf correlate poorly with human ratings. The results also reveal that neural-based metrics are remarkably robust across different domains and challenges."
}`
}
},
  {
  "id": 67,
  "title": "Robust MT Evaluation with Sentence-level Multilingual Augmentation",
  "authors": "Duarte Alves, Ricardo Rei, Ana C Farinha, Jos\u00e9 G. C. de Souza, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Automatic translations with critical errors may lead to misinterpretations and pose several risks for the user. As such, it is important that Machine Translation (MT) Evaluation systems are robust to these errors in order to increase the reliability and safety of Machine Translation systems. Here we introduce SMAUG a novel Sentence-level Multilingual AUGmentation approach for generating translations with critical errors and apply this approach to create a test set to evaluate the robustness of MT metrics to these errors. We show that current State-of-the-Art metrics are improving their capability to distinguish translations with and without critical errors and to penalize the first accordingly. We also show that metrics tend to struggle with errors related to named entities and numbers and that there is a high variance in the robustness of current methods to translations with critical errors.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2022.wmt-1.43/",
  "bibtex": `@inproceedings{alves-etal-2022-robust,
    title = "Robust {MT} Evaluation with Sentence-level Multilingual Augmentation",
    author = "Alves, Duarte  and
      Rei, Ricardo  and
      Farinha, Ana C  and
      C. de Souza, Jos{\\'e} G.  and
      Martins, Andr{\\'e} F. T.",
    editor = {Koehn, Philipp  and
      Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Jimeno Yepes, Antonio  and
      Kocmi, Tom  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Popel, Martin  and
      Turchi, Marco  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.43/",
    pages = "469--478",
    abstract = "Automatic translations with critical errors may lead to misinterpretations and pose several risks for the user. As such, it is important that Machine Translation (MT) Evaluation systems are robust to these errors in order to increase the reliability and safety of Machine Translation systems. Here we introduce SMAUG a novel Sentence-level Multilingual AUGmentation approach for generating translations with critical errors and apply this approach to create a test set to evaluate the robustness of MT metrics to these errors. We show that current State-of-the-Art metrics are improving their capability to distinguish translations with and without critical errors and to penalize the first accordingly. We also show that metrics tend to struggle with errors related to named entities and numbers and that there is a high variance in the robustness of current methods to translations with critical errors."
}`
}
},
  {
  "id": 66,
  "title": "Findings of the WMT 2022 Shared Task on Quality Estimation",
  "authors": "Chrysoula Zerva, Fr\u00e9d\u00e9ric Blain, Ricardo Rei, Piyawat Lertvittayakumjorn, Jos\u00e9 G. C. de Souza, Steffen Eger, Diptesh Kanojia, Duarte Alves, Constantin Or\u0103san, Marina Fomicheva, Andr\u00e9 F. T. Martins, Lucia Specia",
  "venue": "WMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>We report the results of the WMT 2022 shared task on Quality Estimation, in which the challenge is to predict the quality of the output of neural machine translation systems at the word and sentence levels, without access to reference translations. This edition introduces a few novel aspects and extensions that aim to enable more fine-grained, and explainable quality estimation approaches. We introduce an updated quality annotation scheme using Multidimensional Quality Metrics to obtain sentence- and word-level quality scores for three language pairs. We also extend the Direct Assessments and post-edit data (MLQE-PE) to new language pairs: we present a novel and large dataset on English-Marathi, as well as a zero-shot test set on English-Yoruba. Further, we include an explainability sub-task for all language pairs and present a new format of a critical error detection task for two new language pairs. Participants from 11 different teams submitted altogether 991 systems to different task variants and language pairs.</p>`,
  "streams": [
  "evaluation-metrics",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2022.wmt-1.3/",
  "bibtex": `@inproceedings{zerva-etal-2022-findings,
    title = "Findings of the {WMT} 2022 Shared Task on Quality Estimation",
    author = "Zerva, Chrysoula  and
      Blain, Fr{\\'e}d{\\'e}ric  and
      Rei, Ricardo  and
      Lertvittayakumjorn, Piyawat  and
      C. de Souza, Jos{\\'e} G.  and
      Eger, Steffen  and
      Kanojia, Diptesh  and
      Alves, Duarte  and
      Or{\\u{a}}san, Constantin  and
      Fomicheva, Marina  and
      Martins, Andr{\\'e} F. T.  and
      Specia, Lucia",
    editor = {Koehn, Philipp  and
      Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Jimeno Yepes, Antonio  and
      Kocmi, Tom  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Popel, Martin  and
      Turchi, Marco  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.3/",
    pages = "69--99",
    abstract = "We report the results of the WMT 2022 shared task on Quality Estimation, in which the challenge is to predict the quality of the output of neural machine translation systems at the word and sentence levels, without access to reference translations. This edition introduces a few novel aspects and extensions that aim to enable more fine-grained, and explainable quality estimation approaches. We introduce an updated quality annotation scheme using Multidimensional Quality Metrics to obtain sentence- and word-level quality scores for three language pairs. We also extend the Direct Assessments and post-edit data (MLQE-PE) to new language pairs: we present a novel and large dataset on English-Marathi, as well as a zero-shot test set on English-Yoruba. Further, we include an explainability sub-task for all language pairs and present a new format of a critical error detection task for two new language pairs. Participants from 11 different teams submitted altogether 991 systems to different task variants and language pairs."
}`
}
},
  {
  "id": 65,
  "title": "Improving abstractive summarization with energy-based re-ranking",
  "authors": "Diogo Pernes, Afonso Mendes, Andr\u00e9 F. T. Martins",
  "venue": "GEM",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Current abstractive summarization systems present important weaknesses which prevent their deployment in real-world applications, such as the omission of relevant information and the generation of factual inconsistencies (also known as hallucinations). At the same time, automatic evaluation metrics such as CTC scores (Deng et al., 2021) have been recently proposed that exhibit a higher correlation with human judgments than traditional lexical-overlap metrics such as ROUGE. In this work, we intend to close the loop by leveraging the recent advances in summarization metrics to create quality-aware abstractive summarizers. Namely, we propose an energy-based model that learns to re-rank summaries according to one or a combination of these metrics. We experiment using several metrics to train our energy-based re-ranker and show that it consistently improves the scores achieved by the predicted summaries. Nonetheless, human evaluation results show that the re-ranking approach should be used with care for highly abstractive summaries, as the available metrics are not yet sufficiently reliable for this purpose.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2022.gem-1.1/",
  "bibtex": `@inproceedings{pernes-etal-2022-improving,
    title = "Improving abstractive summarization with energy-based re-ranking",
    author = "Pernes, Diogo  and
      Mendes, Afonso  and
      Martins, Andr{\\'e} F. T.",
    editor = "Bosselut, Antoine  and
      Chandu, Khyathi  and
      Dhole, Kaustubh  and
      Gangal, Varun  and
      Gehrmann, Sebastian  and
      Jernite, Yacine  and
      Novikova, Jekaterina  and
      Perez-Beltrachini, Laura",
    booktitle = "Proceedings of the Second Workshop on Natural Language Generation, Evaluation, and Metrics (GEM)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.gem-1.1/",
    doi = "10.18653/v1/2022.gem-1.1",
    pages = "1--17",
    abstract = "Current abstractive summarization systems present important weaknesses which prevent their deployment in real-world applications, such as the omission of relevant information and the generation of factual inconsistencies (also known as hallucinations). At the same time, automatic evaluation metrics such as CTC scores (Deng et al., 2021) have been recently proposed that exhibit a higher correlation with human judgments than traditional lexical-overlap metrics such as ROUGE. In this work, we intend to close the loop by leveraging the recent advances in summarization metrics to create quality-aware abstractive summarizers. Namely, we propose an energy-based model that learns to re-rank summaries according to one or a combination of these metrics. We experiment using several metrics to train our energy-based re-ranker and show that it consistently improves the scores achieved by the predicted summaries. Nonetheless, human evaluation results show that the re-ranking approach should be used with care for highly abstractive summaries, as the available metrics are not yet sufficiently reliable for this purpose."
}`
}
},
  {
  "id": 64,
  "title": "Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation",
  "authors": "Nuno M. Guerreiro, Elena Voita, Andr\u00e9 Martins",
  "venue": "EACL",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Although the problem of hallucinations in neural machine translation (NMT) has received some attention, research on this highly pathological phenomenon lacks solid ground. Previous work has been limited in several ways: it often resorts to artificial settings where the problem is amplified, it disregards some (common) types of hallucinations, and it does not validate adequacy of detection heuristics. In this paper, we set foundations for the study of NMT hallucinations. First, we work in a natural setting, i.e., in-domain data without artificial noise neither in training nor in inference. Next, we annotate a dataset of over 3.4k sentences indicating different kinds of critical errors and hallucinations. Then, we turn to detection methods and both revisit methods used previously and propose using glass-box uncertainty-based detectors. Overall, we show that for preventive settings, (i) previously used methods are largely inadequate, (ii) sequence log-probability works best and performs on par with reference-based methods. Finally, we propose DeHallucinator, a simple method for alleviating hallucinations at test time that significantly reduces the hallucinatory rate. To ease future research, we release our annotated dataset for WMT18 German-English data, along with the model, training data, and code.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2023.eacl-main.75/",
  "bibtex": `@inproceedings{guerreiro-etal-2023-looking,
    title = "Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation",
    author = "Guerreiro, Nuno M.  and
      Voita, Elena  and
      Martins, Andr{\\'e}",
    editor = "Vlachos, Andreas  and
      Augenstein, Isabelle",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.75/",
    doi = "10.18653/v1/2023.eacl-main.75",
    pages = "1059--1075",
    abstract = "Although the problem of hallucinations in neural machine translation (NMT) has received some attention, research on this highly pathological phenomenon lacks solid ground. Previous work has been limited in several ways: it often resorts to artificial settings where the problem is amplified, it disregards some (common) types of hallucinations, and it does not validate adequacy of detection heuristics. In this paper, we set foundations for the study of NMT hallucinations. First, we work in a natural setting, i.e., in-domain data without artificial noise neither in training nor in inference. Next, we annotate a dataset of over 3.4k sentences indicating different kinds of critical errors and hallucinations. Then, we turn to detection methods and both revisit methods used previously and propose using glass-box uncertainty-based detectors. Overall, we show that for preventive settings, (i) previously used methods are largely inadequate, (ii) sequence log-probability works best and performs on par with reference-based methods. Finally, we propose DeHallucinator, a simple method for alleviating hallucinations at test time that significantly reduces the hallucinatory rate."
}`
}
},
  {
  "id": 63,
  "title": "Beyond Characters: Subword-level Morpheme Segmentation",
  "authors": "Ben Peters, Andre F. T. Martins",
  "venue": "SIGMORPHON",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>This paper presents DeepSPIN’s submissions to the SIGMORPHON 2022 Shared Task on Morpheme Segmentation. We make three submissions, all to the word-level subtask. First, we show that entmax-based sparse sequence-tosequence models deliver large improvements over conventional softmax-based models, echoing results from other tasks. Then, we challenge the assumption that models for morphological tasks should be trained at the character level by building a transformer that generates morphemes as sequences of unigram language model-induced subwords. This subword transformer outperforms all of our character-level models and wins the word-level subtask. Although we do not submit an official submission to the sentence-level subtask, we show that this subword-based approach is highly effective there as well.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2022.sigmorphon-1.14/",
  "bibtex": `@inproceedings{peters-martins-2022-beyond,
    title = "Beyond Characters: Subword-level Morpheme Segmentation",
    author = "Peters, Ben  and
      Martins, Andre F. T.",
    editor = "Nicolai, Garrett  and
      Chodroff, Eleanor",
    booktitle = "Proceedings of the 19th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sigmorphon-1.14/",
    doi = "10.18653/v1/2022.sigmorphon-1.14",
    pages = "131--138",
    abstract = "This paper presents DeepSPIN{'}s submissions to the SIGMORPHON 2022 Shared Task on Morpheme Segmentation. We make three submissions, all to the word-level subtask. First, we show that entmax-based sparse sequence-tosequence models deliver large improvements over conventional softmax-based models, echoing results from other tasks. Then, we challenge the assumption that models for morphological tasks should be trained at the character level by building a transformer that generates morphemes as sequences of unigram language model-induced subwords. This subword transformer outperforms all of our character-level models and wins the word-level subtask. Although we do not submit an official submission to the sentence-level subtask, we show that this subword-based approach is highly effective there as well."
}`
}
},
  {
  "id": 62,
  "title": "Differentiable Causal Discovery Under Latent Interventions",
  "authors": "Gon\u00e7alo R. A. Faria, Andr\u00e9 F. T. Martins, M\u00e1rio A. T. Figueiredo",
  "venue": "CLeaR",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Recent work has shown promising results in causal discovery by leveraging interventional data with gradient-based methods, even when the intervened variables are unknown. However, previous work assumes that the correspondence between samples and interventions is known, which is often unrealistic. We envision a scenario with an extensive dataset sampled from multiple intervention distributions and one observation distribution, but where we do not know which distribution originated each sample and how the intervention affected the system, \\textit{i.e.}, interventions are entirely latent. We propose a method based on neural networks and variational inference that addresses this scenario by framing it as learning a shared causal graph among an infinite mixture (under a Dirichlet process prior) of intervention structural causal models. Experiments with synthetic and real data show that our approach and its semi-supervised variant are able to discover causal relations in this challenging scenario.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://proceedings.mlr.press/v177/faria22a/faria22a.pdf",
  "bibtex": `@InProceedings{pmlr-v177-faria22a,
  title = 	 {Differentiable Causal Discovery Under Latent Interventions},
  author =       {Faria, Gon{\\c{c}}alo Rui Alves and Martins, Andre and Figueiredo, Mario A. T.},
  booktitle = 	 {Proceedings of the First Conference on Causal Learning and Reasoning},
  pages = 	 {253--274},
  year = 	 {2022},
  editor = 	 {Schölkopf, Bernhard and Uhler, Caroline and Zhang, Kun},
  volume = 	 {177},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {11--13 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v177/faria22a/faria22a.pdf},
  url = 	 {https://proceedings.mlr.press/v177/faria22a.html},
  abstract = 	 {Recent work has shown promising results in causal discovery by leveraging interventional data with gradient-based methods, even when the intervened variables are unknown. However, previous work assumes that the correspondence between samples and interventions is known, which is often unrealistic. We envision a scenario with an extensive dataset sampled from multiple intervention distributions and one observation distribution, but where we do not know which distribution originated each sample and how the intervention affected the system, \\textit{i.e.}, interventions are entirely latent. We propose a method based on neural networks and variational inference that addresses this scenario by framing it as learning a shared causal graph among a infinite mixture (under a Dirichlet process prior) of intervention structural causal models . Experiments with synthetic and real data show that our approach and its semi-supervised variant are able to discover causal relations in this challenging scenario. }
}`
}
},
  {
  "id": 61,
  "title": "Modeling Structure with Undirected Neural Networks",
  "authors": "Tsvetomila Mihaylova, Vlad Niculae, Andr\u00e9 F. T. Martins",
  "venue": "ICML",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Neural networks are powerful function estimators, leading to their status as a paradigm of choice for modeling structured data. However, unlike other structured representations that emphasize the modularity of the problem -- e.g., factor graphs -- neural networks are usually monolithic mappings from inputs to outputs, with a fixed computation order. This limitation prevents them from capturing different directions of computation and interaction between the modeled variables. In this paper, we combine the representational strengths of factor graphs and of neural networks, proposing undirected neural networks (UNNs): a flexible framework for specifying computations that can be performed in any order. For particular choices, our proposed models subsume and extend many existing architectures: feed-forward, recurrent, self-attention networks, auto-encoders, and networks with implicit layers. We demonstrate the effectiveness of undirected neural architectures, both unstructured and structured, on a range of tasks: tree-constrained dependency parsing, convolutional image classification, and sequence completion with attention. By varying the computation order, we show how a single UNN can be used both as a classifier and a prototype generator, and how it can fill in missing parts of an input sequence, making them a promising field for further research.</p>`,
  "streams": [
  "resources",
  "theory"
],
  "links": {
  "paper": "https://proceedings.mlr.press/v162/mihaylova22a/mihaylova22a.pdf",
  "bibtex": `@InProceedings{pmlr-v162-mihaylova22a,
  title = 	 {Modeling Structure with Undirected Neural Networks},
  author =       {Mihaylova, Tsvetomila and Niculae, Vlad and Martins, Andre},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {15544--15560},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/mihaylova22a/mihaylova22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/mihaylova22a.html},
  abstract = 	 {Neural networks are powerful function estimators, leading to their status as a paradigm of choice for modeling structured data. However, unlike other structured representations that emphasize the modularity of the problem {–} e.g., factor graphs {–} neural networks are usually monolithic mappings from inputs to outputs, with a fixed computation order. This limitation prevents them from capturing different directions of computation and interaction between the modeled variables. In this paper, we combine the representational strengths of factor graphs and of neural networks, proposing undirected neural networks (UNNs): a flexible framework for specifying computations that can be performed in any order. For particular choices, our proposed models subsume and extend many existing architectures: feed-forward, recurrent, self-attention networks, auto-encoders, and networks with implicit layers. We demonstrate the effectiveness of undirected neural architectures, both unstructured and structured, on a range of tasks: tree-constrained dependency parsing, convolutional image classification, and sequence completion with attention. By varying the computation order, we show how a single UNN can be used both as a classifier and a prototype generator, and how it can fill in missing parts of an input sequence, making them a promising field for further research.}
}`
}
},
  {
  "id": 60,
  "title": "Searching for COMETINHO: The Little Metric That Could",
  "authors": "Ricardo Rei, Ana C Farinha, Jos\u00e9 G.C. de Souza, Pedro G. Ramos, Andr\u00e9 F.T. Martins, Luisa Coheur, Alon Lavie",
  "venue": "EAMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>In recent years, several neural fine-tuned machine translation evaluation metrics such as COMET and BLEURT have been proposed. These metrics achieve much higher correlations with human judgments than lexical overlap metrics at the cost of computational efficiency and simplicity, limiting their applications to scenarios in which one has to score thousands of translation hypothesis (e.g. scoring multiple systems or Minimum Bayes Risk decoding). In this paper, we explore optimization techniques, pruning, and knowledge distillation to create more compact and faster COMET versions. Our results show that just by optimizing the code through the use of caching and length batching we can reduce inference time between 39% and 65% when scoring multiple systems. Also, we show that pruning COMET can lead to a 21% model reduction without affecting the model’s accuracy beyond 0.01 Kendall tau correlation. Furthermore, we present DISTIL-COMET a lightweight distilled version that is 80% smaller and 2.128x faster while attaining a performance close to the original model and above strong baselines such as BERTSCORE and PRISM.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2022.eamt-1.9/",
  "bibtex": `@inproceedings{rei-etal-2022-searching,
    title = "Searching for {COMETINHO}: The Little Metric That Could",
    author = "Rei, Ricardo  and
      Farinha, Ana C  and
      de Souza, Jos{\\'e} G.C.  and
      Ramos, Pedro G.  and
      Martins, Andr{\\'e} F.T.  and
      Coheur, Luisa  and
      Lavie, Alon",
    editor = {Moniz, Helena  and
      Macken, Lieve  and
      Rufener, Andrew  and
      Barrault, Lo{\\"i}c  and
      Costa-juss{\\\`a}, Marta R.  and
      Declercq, Christophe  and
      Koponen, Maarit  and
      Kemp, Ellie  and
      Pilos, Spyridon  and
      Forcada, Mikel L.  and
      Scarton, Carolina  and
      Van den Bogaert, Joachim  and
      Daems, Joke  and
      Tezcan, Arda  and
      Vanroy, Bram  and
      Fonteyne, Margot},
    booktitle = "Proceedings of the 23rd Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2022",
    address = "Ghent, Belgium",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2022.eamt-1.9/",
    pages = "61--70",
    abstract = "In recent years, several neural fine-tuned machine translation evaluation metrics such as COMET and BLEURT have been proposed. These metrics achieve much higher correlations with human judgments than lexical overlap metrics at the cost of computational efficiency and simplicity, limiting their applications to scenarios in which one has to score thousands of translation hypothesis (e.g. scoring multiple systems or Minimum Bayes Risk decoding). In this paper, we explore optimization techniques, pruning, and knowledge distillation to create more compact and faster COMET versions. Our results show that just by optimizing the code through the use of caching and length batching we can reduce inference time between 39{\\%} and 65{\\%} when scoring multiple systems. Also, we show that pruning COMET can lead to a 21{\\%} model reduction without affecting the model{'}s accuracy beyond 0.01 Kendall tau correlation. Furthermore, we present DISTIL-COMET a lightweight distilled version that is 80{\\%} smaller and 2.128x faster while attaining a performance close to the original model and above strong baselines such as BERTSCORE and PRISM."
}`
}
},
  {
  "id": 59,
  "title": "QUARTZ: Quality-Aware Machine Translation",
  "authors": "Jos\u00e9 G.C. de Souza, Ricardo Rei, Ana C. Farinha, Helena Moniz, Andr\u00e9 F. T. Martins",
  "venue": "EAMT",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>This paper presents QUARTZ, QUality-AwaRe machine Translation, a project led by Unbabel which aims at developing machine translation systems that are more robust and produce fewer critical errors. With QUARTZ we want to enable machine translation for user-generated conversational content types that do not tolerate critical errors in automatic translations.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2022.eamt-1.47/",
  "bibtex": `@inproceedings{de-souza-etal-2022-quartz,
    title = "{QUARTZ}: Quality-Aware Machine Translation",
    author = "de Souza, Jos{\\'e} G.C.  and
      Rei, Ricardo  and
      Farinha, Ana C.  and
      Moniz, Helena  and
      Martins, Andr{\\'e} F. T.",
    editor = {Moniz, Helena  and
      Macken, Lieve  and
      Rufener, Andrew  and
      Barrault, Lo{\\"i}c  and
      Costa-juss{\\\`a}, Marta R.  and
      Declercq, Christophe  and
      Koponen, Maarit  and
      Kemp, Ellie  and
      Pilos, Spyridon  and
      Forcada, Mikel L.  and
      Scarton, Carolina  and
      Van den Bogaert, Joachim  and
      Daems, Joke  and
      Tezcan, Arda  and
      Vanroy, Bram  and
      Fonteyne, Margot},
    booktitle = "Proceedings of the 23rd Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2022",
    address = "Ghent, Belgium",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2022.eamt-1.47/",
    pages = "315--316",
    abstract = "This paper presents QUARTZ, QUality-AwaRe machine Translation, a project led by Unbabel which aims at developing machine translation systems that are more robust and produce fewer critical errors. With QUARTZ we want to enable machine translation for user-generated conversational content types that do not tolerate critical errors in automatic translations."
}`
}
},
  {
  "id": 58,
  "title": "Chunk-based Nearest Neighbor Machine Translation",
  "authors": "Pedro Henrique Martins, Zita Marinho, Andr\u00e9 F. T. Martins",
  "venue": "EMNLP",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Semi-parametric models, which augment generation with retrieval, have led to impressive results in language modeling and machine translation, due to their ability to retrieve fine-grained information from a datastore of examples. One of the most prominent approaches, kNN-MT, exhibits strong domain adaptation capabilities by retrieving tokens from domain-specific datastores (Khandelwal et al., 2021). However, kNN-MT requires an expensive retrieval operation for every single generated token, leading to a very low decoding speed (around 8 times slower than a parametric model). In this paper, we introduce a chunk-based kNN-MT model which retrieves chunks of tokens from the datastore, instead of a single token. We propose several strategies for incorporating the retrieved chunks into the generation process, and for selecting the steps at which the model needs to search for neighbors in the datastore. Experiments on machine translation in two settings, static and “on-the-fly” domain adaptation, show that the chunk-based kNN-MT model leads to significant speed-ups (up to 4 times) with only a small drop in translation quality.</p>`,
  "streams": [
  "multilingual-translation",
  "retrieval"
],
  "links": {
  "paper": "https://aclanthology.org/2022.emnlp-main.284/",
  "bibtex": `@inproceedings{martins-etal-2022-chunk,
    title = "Chunk-based Nearest Neighbor Machine Translation",
    author = "Martins, Pedro Henrique  and
      Marinho, Zita  and
      Martins, Andr{\\'e} F. T.",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.284/",
    doi = "10.18653/v1/2022.emnlp-main.284",
    pages = "4228--4245",
    abstract = "Semi-parametric models, which augment generation with retrieval, have led to impressive results in language modeling and machine translation, due to their ability to retrieve fine-grained information from a datastore of examples. One of the most prominent approaches, kNN-MT, exhibits strong domain adaptation capabilities by retrieving tokens from domain-specific datastores (Khandelwal et al., 2021). However, kNN-MT requires an expensive retrieval operation for every single generated token, leading to a very low decoding speed (around 8 times slower than a parametric model). In this paper, we introduce a chunk-based kNN-MT model which retrieves chunks of tokens from the datastore, instead of a single token. We propose several strategies for incorporating the retrieved chunks into the generation process, and for selecting the steps at which the model needs to search for neighbors in the datastore. Experiments on machine translation in two settings, static and \`\`on-the-fly'' domain adaptation, show that the chunk-based kNN-MT model leads to significant speed-ups (up to 4 times) with only a small drop in translation quality."
}`
}
},
  {
  "id": 57,
  "title": "Quality-Aware Decoding for Neural Machine Translation",
  "authors": "Patrick Fernandes, Ant\u00f3nio Farinhas, Ricardo Rei, Jos\u00e9 De Souza, Perez Ogayo, Graham Neubig, Andre Martins",
  "venue": "NAACL",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Despite the progress in machine translation quality estimation and evaluation in the last years, decoding in neural machine translation (NMT) is mostly oblivious to this and centers around finding the most probable translation according to the model (MAP decoding), approximated with beam search. In this paper, we bring together these two lines of research and propose quality-aware decoding for NMT, by leveraging recent breakthroughs in reference-free and reference-based MT evaluation through various inference methods like N-best reranking and minimum Bayes risk decoding. We perform an extensive comparison of various possible candidate generation and ranking methods across four datasets and two model classes and find that quality-aware decoding consistently outperforms MAP-based decoding according both to state-of-the-art automatic metrics (COMET and BLEURT) and to human assessments.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2022.naacl-main.100/",
  "bibtex": `@inproceedings{fernandes-etal-2022-quality,
    title = "Quality-Aware Decoding for Neural Machine Translation",
    author = "Fernandes, Patrick  and
      Farinhas, Ant{\\'o}nio  and
      Rei, Ricardo  and
      C. de Souza, Jos{\\'e} G.  and
      Ogayo, Perez  and
      Neubig, Graham  and
      Martins, Andre",
    editor = "Carpuat, Marine  and
      de Marneffe, Marie-Catherine  and
      Meza Ruiz, Ivan Vladimir",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.100/",
    doi = "10.18653/v1/2022.naacl-main.100",
    pages = "1396--1412",
    abstract = "Despite the progress in machine translation quality estimation and evaluation in the last years, decoding in neural machine translation (NMT) is mostly oblivious to this and centers around finding the most probable translation according to the model (MAP decoding), approximated with beam search. In this paper, we bring together these two lines of research and propose \\textit{quality-aware decoding} for NMT, by leveraging recent breakthroughs in reference-free and reference-based MT evaluation through various inference methods like $N$-best reranking and minimum Bayes risk decoding. We perform an extensive comparison of various possible candidate generation and ranking methods across four datasets and two model classes and find that quality-aware decoding consistently outperforms MAP-based decoding according both to state-of-the-art automatic metrics (COMET and BLEURT) and to human assessments."
}`
}
},
  {
  "id": 56,
  "title": "Efficient Machine Translation Domain Adaptation",
  "authors": "Pedro Martins, Zita Marinho, Andre Martins",
  "venue": "SPANLP",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Machine translation models struggle when translating out-of-domain text, which makes domain adaptation a topic of critical importance. However, most domain adaptation methods focus on fine-tuning or training the entire or part of the model on every new domain, which can be costly. On the other hand, semi-parametric models have been shown to successfully perform domain adaptation by retrieving examples from an in-domain datastore (Khandelwal et al., 2021). A drawback of these retrieval-augmented models, however, is that they tend to be substantially slower. In this paper, we explore several approaches to speed up nearest neighbor machine translation. We adapt the methods recently proposed by He et al. (2021) for language modeling, and introduce a simple but effective caching strategy that avoids performing retrieval when similar contexts have been seen before. Translation quality and runtimes for several domains show the effectiveness of the proposed solutions.</p>`,
  "streams": [
  "efficiency",
  "multilingual-translation",
  "retrieval"
],
  "links": {
  "paper": "https://aclanthology.org/2022.spanlp-1.3/",
  "code": "https://github.com/deep-spin/efficient_kNN_MT",
  "bibtex": `@inproceedings{martins-etal-2022-efficient,
    title = "Efficient Machine Translation Domain Adaptation",
    author = "Martins, Pedro  and
      Marinho, Zita  and
      Martins, Andre",
    editor = "Das, Rajarshi  and
      Lewis, Patrick  and
      Min, Sewon  and
      Thai, June  and
      Zaheer, Manzil",
    booktitle = "Proceedings of the 1st Workshop on Semiparametric Methods in NLP: Decoupling Logic from Knowledge",
    month = may,
    year = "2022",
    address = "Dublin, Ireland and Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.spanlp-1.3/",
    doi = "10.18653/v1/2022.spanlp-1.3",
    pages = "23--29",
    abstract = "Machine translation models struggle when translating out-of-domain text, which makes domain adaptation a topic of critical importance. However, most domain adaptation methods focus on fine-tuning or training the entire or part of the model on every new domain, which can be costly. On the other hand, semi-parametric models have been shown to successfully perform domain adaptation by retrieving examples from an in-domain datastore (Khandelwal et al., 2021). A drawback of these retrieval-augmented models, however, is that they tend to be substantially slower. In this paper, we explore several approaches to speed up nearest neighbors machine translation. We adapt the methods recently proposed by He et al. (2021) for language modeling, and introduce a simple but effective caching strategy that avoids performing retrieval when similar contexts have been seen before. Translation quality and runtimes for several domains show the effectiveness of the proposed solutions."
}`
}
},
  {
  "id": 55,
  "title": "Disentangling Uncertainty in Machine Translation Evaluation\n",
  "authors": "Chrysoula Zerva, Taisiya Glushkova, Ricardo Rei, Andr\u00e9 F. T. Martins",
  "venue": "EMNLP",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Trainable evaluation metrics for machine translation (MT) exhibit strong correlation with human judgements, but they are often hard to interpret and might produce unreliable scores under noisy or out-of-domain data. Recent work has attempted to mitigate this with simple uncertainty quantification techniques (Monte Carlo dropout and deep ensembles), however these techniques (as we show) are limited in several ways -- for example, they are unable to distinguish between different kinds of uncertainty, and they are time and memory consuming. In this paper, we propose more powerful and efficient uncertainty predictors for MT evaluation, and we assess their ability to target different sources of aleatoric and epistemic uncertainty. To this end, we develop and compare training objectives for the COMET metric to enhance it with an uncertainty prediction output, including heteroscedastic regression, divergence minimization, and direct uncertainty prediction. Our experiments show improved results on uncertainty prediction for the WMT metrics task datasets, with a substantial reduction in computational costs. Moreover, they demonstrate the ability of these predictors to address specific uncertainty causes in MT evaluation, such as low quality references and out-of-domain data.</p>`,
  "streams": [
  "evaluation-metrics",
  "uncertainty"
],
  "links": {
  "paper": "https://aclanthology.org/2022.emnlp-main.591/",
  "bibtex": `@inproceedings{zerva-etal-2022-disentangling,
    title = "Disentangling Uncertainty in Machine Translation Evaluation",
    author = "Zerva, Chrysoula  and
      Glushkova, Taisiya  and
      Rei, Ricardo  and
      Martins, Andr{\\'e} F. T.",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.591/",
    doi = "10.18653/v1/2022.emnlp-main.591",
    pages = "8622--8641",
    abstract = "Trainable evaluation metrics for machine translation (MT) exhibit strong correlation with human judgements, but they are often hard to interpret and might produce unreliable scores under noisy or out-of-domain data. Recent work has attempted to mitigate this with simple uncertainty quantification techniques (Monte Carlo dropout and deep ensembles), however these techniques (as we show) are limited in several ways {--} for example, they are unable to distinguish between different kinds of uncertainty, and they are time and memory consuming. In this paper, we propose more powerful and efficient uncertainty predictors for MT evaluation, and we assess their ability to target different sources of aleatoric and epistemic uncertainty. To this end, we develop and compare training objectives for the COMET metric to enhance it with an uncertainty prediction output, including heteroscedastic regression, divergence minimization, and direct uncertainty prediction.Our experiments show improved results on uncertainty prediction for the WMT metrics task datasets, with a substantial reduction in computational costs. Moreover, they demonstrate the ability of these predictors to address specific uncertainty causes in MT evaluation, such as low quality references and out-of-domain data."
}`
}
},
  {
  "id": 54,
  "title": "\u221e-former: Infinite Memory Transformer",
  "authors": "Pedro Henrique Martins, Zita Marinho, Andr\u00e9 F. T. Martins",
  "venue": "ACL",
  "year": 2022,
  "type": "conference",
  "abstract": `<p>Transformers are unable to model long-term memories effectively, since the amount of computation they need to perform grows with the context length. While variations of efficient transformers have been proposed, they all have a finite memory capacity and are forced to drop old information. In this paper, we propose the ∞-former, which extends the vanilla transformer with an unbounded long-term memory. By making use of a continuous-space attention mechanism to attend over the long-term memory, the ∞-former’s attention complexity becomes independent of the context length, trading off memory length with precision.In order to control where precision is more important, ∞-former maintains “sticky memories,” being able to model arbitrarily long contexts while keeping the computation budget fixed.Experiments on a synthetic sorting task, language modeling, and document grounded dialogue generation demonstrate the ∞-former’s ability to retain information from long sequences.</p>`,
  "streams": [
  "memory",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2022.acl-long.375/",
  "code": "https://github.com/deep-spin/infinite-former",
  "bibtex": `@inproceedings{martins-etal-2022-former,
    title = "$\\infty$-former: Infinite Memory Transformer",
    author = "Martins, Pedro Henrique  and
      Marinho, Zita  and
      Martins, Andre",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.375/",
    doi = "10.18653/v1/2022.acl-long.375",
    pages = "5468--5485",
    abstract = "Transformers are unable to model long-term memories effectively, since the amount of computation they need to perform grows with the context length. While variations of efficient transformers have been proposed, they all have a finite memory capacity and are forced to drop old information. In this paper, we propose the $\\infty$-former, which extends the vanilla transformer with an unbounded long-term memory. By making use of a continuous-space attention mechanism to attend over the long-term memory, the $\\infty$-former{'}s attention complexity becomes independent of the context length, trading off memory length with precision.In order to control where precision is more important, $\\infty$-former maintains \`\`sticky memories,'' being able to model arbitrarily long contexts while keeping the computation budget fixed.Experiments on a synthetic sorting task, language modeling, and document grounded dialogue generation demonstrate the $\\infty$-former{'}s ability to retain information from long sequences."
}`
}
},
  {
  "id": 53,
  "title": "Predicting Attention Sparsity in Transformers",
  "authors": "Marcos Treviso, Ant\u00f3nio G\u00f3is, Patrick Fernandes, Erick Fonseca, Andre Martins",
  "venue": "SPNLP",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>Transformers' quadratic complexity with respect to the input sequence length has motivated a body of work on efficient sparse approximations to softmax. An alternative path, used by entmax transformers, consists of having built-in exact sparse attention; however this approach still requires quadratic computation. In this paper, we propose Sparsefinder, a simple model trained to identify the sparsity pattern of entmax attention before computing it. We experiment with three variants of our method, based on distances, quantization, and clustering, on two tasks: machine translation (attention in the decoder) and masked language modeling (encoder-only). Our work provides a new angle to study model efficiency by doing extensive analysis of the tradeoff between the sparsity and recall of the predicted attention graph. This allows for detailed comparison between different models along their Pareto curves, important to guide future benchmarks for sparse attention models.</p>`,
  "streams": [
  "attention"
],
  "links": {
  "paper": "https://aclanthology.org/2022.spnlp-1.7/",
  "bibtex": `@inproceedings{treviso-etal-2022-predicting,
    title = "Predicting Attention Sparsity in Transformers",
    author = "Treviso, Marcos  and
      G{\\'o}is, Ant{\\'o}nio  and
      Fernandes, Patrick  and
      Fonseca, Erick  and
      Martins, Andre",
    editor = "Vlachos, Andreas  and
      Agrawal, Priyanka  and
      Martins, Andr{\\'e}  and
      Lampouras, Gerasimos  and
      Lyu, Chunchuan",
    booktitle = "Proceedings of the Sixth Workshop on Structured Prediction for NLP",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.spnlp-1.7/",
    doi = "10.18653/v1/2022.spnlp-1.7",
    pages = "67--81",
    abstract = "Transformers' quadratic complexity with respect to the input sequence length has motivated a body of work on efficient sparse approximations to softmax. An alternative path, used by entmax transformers, consists of having built-in exact sparse attention; however this approach still requires quadratic computation. In this paper, we propose Sparsefinder, a simple model trained to identify the sparsity pattern of entmax attention before computing it. We experiment with three variants of our method, based on distances, quantization, and clustering, on two tasks: machine translation (attention in the decoder) and masked language modeling (encoder-only). Our work provides a new angle to study model efficiency by doing extensive analysis of the tradeoff between the sparsity and recall of the predicted attention graph. This allows for detailed comparison between different models along their Pareto curves, important to guide future benchmarks for sparse attention models."
}`
}
},
  {
  "id": 52,
  "title": "IST-Unbabel 2021 Submission for the Explainable Quality Estimation Shared Task",
  "authors": "Marcos Treviso, Nuno M. Guerreiro, Ricardo Rei, Andr\u00e9 F. T. Martins",
  "venue": "Eval4NLP",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>We present the joint contribution of Instituto Superior Técnico (IST) and Unbabel to the Explainable Quality Estimation (QE) shared task, where systems were submitted to two tracks: constrained (without word-level supervision) and unconstrained (with word-level supervision). For the constrained track, we experimented with several explainability methods to extract the relevance of input tokens from sentence-level QE models built on top of multilingual pre-trained transformers. Among the different tested methods, composing explanations in the form of attention weights scaled by the norm of value vectors yielded the best results. When word-level labels are used during training, our best results were obtained by using word-level predicted probabilities. We further improve the performance of our methods on the two tracks by ensembling explanation scores extracted from models trained with different pre-trained transformers, achieving strong results for in-domain and zero-shot language pairs.</p>`,
  "streams": [
  "evaluation-metrics",
  "interpretability",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2021.eval4nlp-1.14/",
  "bibtex": `@inproceedings{treviso-etal-2021-ist,
    title = "{IST}-Unbabel 2021 Submission for the Explainable Quality Estimation Shared Task",
    author = "Treviso, Marcos  and
      Guerreiro, Nuno M.  and
      Rei, Ricardo  and
      Martins, Andr{\\'e} F. T.",
    editor = "Gao, Yang  and
      Eger, Steffen  and
      Zhao, Wei  and
      Lertvittayakumjorn, Piyawat  and
      Fomicheva, Marina",
    booktitle = "Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eval4nlp-1.14/",
    doi = "10.18653/v1/2021.eval4nlp-1.14",
    pages = "133--145",
    abstract = "We present the joint contribution of Instituto Superior T{\\'e}cnico (IST) and Unbabel to the Explainable Quality Estimation (QE) shared task, where systems were submitted to two tracks: constrained (without word-level supervision) and unconstrained (with word-level supervision). For the constrained track, we experimented with several explainability methods to extract the relevance of input tokens from sentence-level QE models built on top of multilingual pre-trained transformers. Among the different tested methods, composing explanations in the form of attention weights scaled by the norm of value vectors yielded the best results. When word-level labels are used during training, our best results were obtained by using word-level predicted probabilities. We further improve the performance of our methods on the two tracks by ensembling explanation scores extracted from models trained with different pre-trained transformers, achieving strong results for in-domain and zero-shot language pairs."
}`
}
},
  {
  "id": 51,
  "title": "Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task",
  "authors": "Ricardo Rei, Ana C Farinha, Chrysoula Zerva, Daan van Stigt, Craig Stewart, Pedro Ramos, Taisiya Glushkova, Andr\u00e9 F. T. Martins, Alon Lavie",
  "venue": "WMT",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>In this paper, we present the joint contribution of Unbabel and IST to the WMT 2021 Metrics Shared Task. With this year’s focus on Multidimensional Quality Metric (MQM) as the ground-truth human assessment, our aim was to steer COMET towards higher correlations with MQM. We do so by first pre-training on Direct Assessments and then fine-tuning on z-normalized MQM scores. In our experiments we also show that reference-free COMET models are becoming competitive with reference-based models, even outperforming the best COMET model from 2020 on this year’s development data. Additionally, we present COMETinho, a lightweight COMET model that is 19x faster on CPU than the original model, while also achieving state-of-the-art correlations with MQM. Finally, in the “QE as a metric” track, we also participated with a QE model trained using the OpenKiwi framework leveraging MQM scores and word-level annotations.</p>`,
  "streams": [
  "evaluation-metrics",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2021.wmt-1.111/",
  "bibtex": `@inproceedings{rei-etal-2021-references,
    title = "Are References Really Needed? Unbabel-{IST} 2021 Submission for the Metrics Shared Task",
    author = "Rei, Ricardo  and
      Farinha, Ana C  and
      Zerva, Chrysoula  and
      van Stigt, Daan  and
      Stewart, Craig  and
      Ramos, Pedro  and
      Glushkova, Taisiya  and
      Martins, Andr{\\'e} F. T.  and
      Lavie, Alon",
    editor = "Barrault, Loic  and
      Bojar, Ondrej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-jussa, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Kocmi, Tom  and
      Martins, Andre  and
      Morishita, Makoto  and
      Monz, Christof",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wmt-1.111/",
    pages = "1030--1040",
    abstract = "In this paper, we present the joint contribution of Unbabel and IST to the WMT 2021 Metrics Shared Task. With this year{'}s focus on Multidimensional Quality Metric (MQM) as the ground-truth human assessment, our aim was to steer COMET towards higher correlations with MQM. We do so by first pre-training on Direct Assessments and then fine-tuning on z-normalized MQM scores. In our experiments we also show that reference-free COMET models are becoming competitive with reference-based models, even outperforming the best COMET model from 2020 on this year{'}s development data. Additionally, we present COMETinho, a lightweight COMET model that is 19x faster on CPU than the original model, while also achieving state-of-the-art correlations with MQM. Finally, in the \`\`QE as a metric'' track, we also participated with a QE model trained using the OpenKiwi framework leveraging MQM scores and word-level annotations."
}`
}
},
  {
  "id": 50,
  "title": "Findings of the WMT 2021 Shared Task on Quality Estimation",
  "authors": "Lucia Specia, Fr\u00e9d\u00e9ric Blain, Marina Fomicheva, Chrysoula Zerva, Zhenhao Li, Vishrav Chaudhary, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>We report the results of the WMT 2021 shared task on Quality Estimation, where the challenge is to predict the quality of the output of neural machine translation systems at the word and sentence levels. This edition focused on two main novel additions: (i) prediction for unseen languages, i.e. zero-shot settings, and (ii) prediction of sentences with catastrophic errors. In addition, new data was released for a number of languages, especially post-edited data. Participating teams from 19 institutions submitted altogether 1263 systems to different task variants and language pairs.</p>`,
  "streams": [
  "evaluation-metrics",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2021.wmt-1.71/",
  "bibtex": `@inproceedings{specia-etal-2021-findings,
    title = "Findings of the {WMT} 2021 Shared Task on Quality Estimation",
    author = "Specia, Lucia  and
      Blain, Fr{\\'e}d{\\'e}ric  and
      Fomicheva, Marina  and
      Zerva, Chrysoula  and
      Li, Zhenhao  and
      Chaudhary, Vishrav  and
      Martins, Andr{\\'e} F. T.",
    editor = "Barrault, Loic  and
      Bojar, Ondrej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-jussa, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Freitag, Markus  and
      Graham, Yvette  and
      Grundkiewicz, Roman  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Kocmi, Tom  and
      Martins, Andre  and
      Morishita, Makoto  and
      Monz, Christof",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wmt-1.71/",
    pages = "684--725",
    abstract = "We report the results of the WMT 2021 shared task on Quality Estimation, where the challenge is to predict the quality of the output of neural machine translation systems at the word and sentence levels. This edition focused on two main novel additions: (i) prediction for unseen languages, i.e. zero-shot settings, and (ii) prediction of sentences with catastrophic errors. In addition, new data was released for a number of languages, especially post-edited data. Participating teams from 19 institutions submitted altogether 1263 systems to different task variants and language pairs."
}`
}
},
  {
  "id": 49,
  "title": "Sparse and Structured Visual Attention",
  "authors": "Pedro Henrique Martins, Vlad Niculae, Zita Marinho, Andre F. T. Martins",
  "venue": "ICIP",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>Visual attention mechanisms are widely used in multimodal tasks, as visual question answering (VQA). One drawback of softmax-based attention mechanisms is that they assign some probability mass to all image regions, regardless of their adjacency structure and of their relevance to the text. In this paper, to better link the image structure with the text, we replace the traditional softmax attention mechanism with two alternative sparsity-promoting transformations: sparsemax, which is able to select only the relevant regions (assigning zero weight to the rest), and a newly proposed Total-Variation Sparse Attention (TVmax), which further encourages the joint selection of adjacent spatial locations. Experiments in VQA show gains in accuracy as well as higher similarity to human attention, which suggests better interpretability.</p>`,
  "streams": [
  "attention",
  "multimodal"
],
  "links": {
  "paper": "https://arxiv.org/abs/2002.05556",
  "bibtex": `@inproceedings{martins2021sparse,
  title={Sparse and structured visual attention},
  author={Martins, Pedro Henrique and Niculae, Vlad and Marinho, Zita and Martins, Andr{\\'e} FT},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={379--383},
  year={2021},
  organization={IEEE}
}`
}
},
  {
  "id": 48,
  "title": "Uncertainty-Aware Machine Translation Evaluation",
  "authors": "Taisiya Glushkova, Chrysoula Zerva, Ricardo Rei, Andr\u00e9 F. T. Martins",
  "venue": "EMNLP Findings",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>Several neural-based metrics have been recently proposed to evaluate machine translation quality. However, all of them resort to point estimates, which provide limited information at segment level. This is made worse as they are trained on noisy, biased and scarce human judgements, often resulting in unreliable quality predictions. In this paper, we introduce uncertainty-aware MT evaluation and analyze the trustworthiness of the predicted quality. We combine the COMET framework with two uncertainty estimation methods, Monte Carlo dropout and deep ensembles, to obtain quality scores along with confidence intervals. We compare the performance of our uncertainty-aware MT evaluation methods across multiple language pairs from the QT21 dataset and the WMT20 metrics task, augmented with MQM annotations. We experiment with varying numbers of references and further discuss the usefulness of uncertainty-aware quality estimation (without references) to flag possibly critical translation mistakes.</p>`,
  "streams": [
  "evaluation-metrics",
  "uncertainty"
],
  "links": {
  "paper": "https://aclanthology.org/2021.findings-emnlp.330/",
  "bibtex": `@inproceedings{glushkova-etal-2021-uncertainty-aware,
    title = "Uncertainty-Aware Machine Translation Evaluation",
    author = "Glushkova, Taisiya  and
      Zerva, Chrysoula  and
      Rei, Ricardo  and
      Martins, Andr{\\'e} F. T.",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.330/",
    doi = "10.18653/v1/2021.findings-emnlp.330",
    pages = "3920--3938",
    abstract = "Several neural-based metrics have been recently proposed to evaluate machine translation quality. However, all of them resort to point estimates, which provide limited information at segment level. This is made worse as they are trained on noisy, biased and scarce human judgements, often resulting in unreliable quality predictions. In this paper, we introduce uncertainty-aware MT evaluation and analyze the trustworthiness of the predicted quality. We combine the COMET framework with two uncertainty estimation methods, Monte Carlo dropout and deep ensembles, to obtain quality scores along with confidence intervals. We compare the performance of our uncertainty-aware MT evaluation methods across multiple language pairs from the QT21 dataset and the WMT20 metrics task, augmented with MQM annotations. We experiment with varying numbers of references and further discuss the usefulness of uncertainty-aware quality estimation (without references) to flag possibly critical translation mistakes."
}`
}
},
  {
  "id": 47,
  "title": "SPECTRA: Sparse Structured Text Rationalization",
  "authors": "Nuno M. Guerreiro, Andr\u00e9 F. T. Martins",
  "venue": "EMNLP",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>Selective rationalization aims to produce decisions along with rationales (e.g., text highlights or word alignments between two sentences). Commonly, rationales are modeled as stochastic binary masks, requiring sampling-based gradient estimators, which complicates training and requires careful hyperparameter tuning. Sparse attention mechanisms are a deterministic alternative, but they lack a way to regularize the rationale extraction (e.g., to control the sparsity of a text highlight or the number of alignments). In this paper, we present a unified framework for deterministic extraction of structured explanations via constrained inference on a factor graph, forming a differentiable layer. Our approach greatly eases training and rationale regularization, generally outperforming previous work on what comes to performance and plausibility of the extracted rationales. We further provide a comparative study of stochastic and deterministic methods for rationale extraction for classification and natural language inference tasks, jointly assessing their predictive power, quality of the explanations, and model variability.</p>`,
  "streams": [
  "attention",
  "interpretability",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2021.emnlp-main.525/",
  "bibtex": `@inproceedings{guerreiro-martins-2021-spectra,
    title = "{SPECTRA}: Sparse Structured Text Rationalization",
    author = "Guerreiro, Nuno M.  and
      Martins, Andr{\\'e} F. T.",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.525/",
    doi = "10.18653/v1/2021.emnlp-main.525",
    pages = "6534--6550",
    abstract = "Selective rationalization aims to produce decisions along with rationales (e.g., text highlights or word alignments between two sentences). Commonly, rationales are modeled as stochastic binary masks, requiring sampling-based gradient estimators, which complicates training and requires careful hyperparameter tuning. Sparse attention mechanisms are a deterministic alternative, but they lack a way to regularize the rationale extraction (e.g., to control the sparsity of a text highlight or the number of alignments). In this paper, we present a unified framework for deterministic extraction of structured explanations via constrained inference on a factor graph, forming a differentiable layer. Our approach greatly eases training and rationale regularization, generally outperforming previous work on what comes to performance and plausibility of the extracted rationales. We further provide a comparative study of stochastic and deterministic methods for rationale extraction for classification and natural language inference tasks, jointly assessing their predictive power, quality of the explanations, and model variability."
}`
}
},
  {
  "id": 46,
  "title": "Sparse Communication via Mixed Distributions",
  "authors": "Ant\u00f3nio Farinhas, Wilker Aziz, Vlad Niculae, Andr\u00e9 F. T. Martins",
  "venue": "ICLR",
  "year": 2021,
  "type": "conference",
  "award": "Oral",
  "abstract": `<p>Neural networks and other machine learning models compute continuous representations, while humans communicate mostly through discrete symbols. Reconciling these two forms of communication is desirable for generating human-readable interpretations or learning discrete latent variable models, while maintaining end-to-end differentiability. Some existing approaches (such as the Gumbel-Softmax transformation) build continuous relaxations that are discrete approximations in the zero-temperature limit, while others (such as sparsemax transformations and the Hard Concrete distribution) produce discrete/continuous hybrids. In this paper, we build rigorous theoretical foundations for these hybrids, which we call "mixed random variables." Our starting point is a new "direct sum" base measure defined on the face lattice of the probability simplex. From this measure, we introduce new entropy and Kullback-Leibler divergence functions that subsume the discrete and differential cases and have interpretations in terms of code optimality. Our framework suggests two strategies for representing and sampling mixed random variables, an extrinsic ("sample-and-project") and an intrinsic one (based on face stratification). We experiment with both approaches on an emergent communication benchmark and on modeling MNIST and Fashion-MNIST data with variational auto-encoders with mixed latent variables.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://openreview.net/forum?id=WAid50QschI",
  "code": "https://github.com/deep-spin/sparse-communication",
  "bibtex": `@inproceedings{
farinhas2022sparse,
title={Sparse Communication via Mixed Distributions},
author={Ant{\\'o}nio Farinhas and Wilker Aziz and Vlad Niculae and Andre Martins},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=WAid50QschI}
}`
}
},
  {
  "id": 45,
  "title": "Do Context-Aware Translation Models Pay the Right Attention?",
  "authors": "Kayo Yin, Patrick Fernandes, Danish Pruthi, Aditi Chaudhary, Andr\u00e9 F. T. Martins, Graham Neubig",
  "venue": "ACL",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>Context-aware machine translation models are designed to leverage contextual information, but often fail to do so. As a result, they inaccurately disambiguate pronouns and polysemous words that require context for resolution. In this paper, we ask several questions: What contexts do human translators use to resolve ambiguous words? Are models paying large amounts of attention to the same context? What if we explicitly train them to do so? To answer these questions, we introduce SCAT (Supporting Context for Ambiguous Translations), a new English-French dataset comprising supporting context words for 14K translations that professional translators found useful for pronoun disambiguation. Using SCAT, we perform an in-depth analysis of the context used to disambiguate, examining positional and lexical characteristics of the supporting words. Furthermore, we measure the degree of alignment between the model’s attention scores and the supporting context from SCAT, and apply a guided attention strategy to encourage agreement between the two.</p>`,
  "streams": [
  "dialogue-context",
  "interpretability",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2021.acl-long.65/",
  "bibtex": `@inproceedings{yin-etal-2021-context,
    title = "Do Context-Aware Translation Models Pay the Right Attention?",
    author = "Yin, Kayo  and
      Fernandes, Patrick  and
      Pruthi, Danish  and
      Chaudhary, Aditi  and
      Martins, Andr{\\'e} F. T.  and
      Neubig, Graham",
    editor = "Zong, Chengqing  and
      Xia, Fei  and
      Li, Wenjie  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.65/",
    doi = "10.18653/v1/2021.acl-long.65",
    pages = "788--801",
    abstract = "Context-aware machine translation models are designed to leverage contextual information, but often fail to do so. As a result, they inaccurately disambiguate pronouns and polysemous words that require context for resolution. In this paper, we ask several questions: What contexts do human translators use to resolve ambiguous words? Are models paying large amounts of attention to the same context? What if we explicitly train them to do so? To answer these questions, we introduce SCAT (Supporting Context for Ambiguous Translations), a new English-French dataset comprising supporting context words for 14K translations that professional translators found useful for pronoun disambiguation. Using SCAT, we perform an in-depth analysis of the context used to disambiguate, examining positional and lexical characteristics of the supporting words. Furthermore, we measure the degree of alignment between the model{'}s attention scores and the supporting context from SCAT, and apply a guided attention strategy to encourage agreement between the two."
}`
}
},
  {
  "id": 44,
  "title": "Measuring and Increasing Context Usage in Context-Aware Machine Translation",
  "authors": "Patrick Fernandes, Kayo Yin, Graham Neubig, Andr\u00e9 F. T. Martins",
  "venue": "ACL",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>Recent work in neural machine translation has demonstrated both the necessity and feasibility of using inter-sentential context, context from sentences other than those currently being translated. However, while many current methods present model architectures that theoretically can use this extra context, it is often not clear how much they do actually utilize it at translation time. In this paper, we introduce a new metric, conditional cross-mutual information, to quantify usage of context by these models. Using this metric, we measure how much document-level machine translation systems use particular varieties of context. We find that target context is referenced more than source context, and that including more context has a diminishing affect on results. We then introduce a new, simple training method, context-aware word dropout, to increase the usage of context by context-aware models. Experiments show that our method not only increases context usage, but also improves the translation quality according to metrics such as BLEU and COMET, as well as performance on anaphoric pronoun resolution and lexical cohesion contrastive datasets.</p>`,
  "streams": [
  "dialogue-context",
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2021.acl-long.505/",
  "bibtex": `@inproceedings{fernandes-etal-2021-measuring,
    title = "Measuring and Increasing Context Usage in Context-Aware Machine Translation",
    author = "Fernandes, Patrick  and
      Yin, Kayo  and
      Neubig, Graham  and
      Martins, Andr{\\'e} F. T.",
    editor = "Zong, Chengqing  and
      Xia, Fei  and
      Li, Wenjie  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.505/",
    doi = "10.18653/v1/2021.acl-long.505",
    pages = "6467--6478",
    abstract = "Recent work in neural machine translation has demonstrated both the necessity and feasibility of using inter-sentential context, context from sentences other than those currently being translated. However, while many current methods present model architectures that theoretically can use this extra context, it is often not clear how much they do actually utilize it at translation time. In this paper, we introduce a new metric, conditional cross-mutual information, to quantify usage of context by these models. Using this metric, we measure how much document-level machine translation systems use particular varieties of context. We find that target context is referenced more than source context, and that including more context has a diminishing affect on results. We then introduce a new, simple training method, context-aware word dropout, to increase the usage of context by context-aware models. Experiments show that our method not only increases context usage, but also improves the translation quality according to metrics such as BLEU and COMET, as well as performance on anaphoric pronoun resolution and lexical cohesion contrastive datasets."
}`
}
},
  {
  "id": 43,
  "title": "Reconciling the Discrete-Continuous Divide: Towards a Mathematical Theory of Sparse Communication",
  "authors": "Andr\u00e9 F. T. Martins",
  "venue": "arXiv",
  "year": 2021,
  "type": "preprint",
  "abstract": `<p>Neural networks and other machine learning models compute continuous representations, while humans communicate with discrete symbols. Reconciling these two forms of communication is desirable to generate human-readable interpretations or to learn discrete latent variable models, while maintaining end-to-end differentiability. Some existing approaches (such as the Gumbel-softmax transformation) build continuous relaxations that are discrete approximations in the zero-temperature limit, while others (such as sparsemax transformations and the hard concrete distribution) produce discrete/continuous hybrids. In this paper, we build rigorous theoretical foundations for these hybrids. Our starting point is a new "direct sum" base measure defined on the face lattice of the probability simplex. From this measure, we introduce a new entropy function that includes the discrete and differential entropies as particular cases, and has an interpretation in terms of code optimality, as well as two other information-theoretic counterparts that generalize the mutual information and Kullback-Leibler divergences. Finally, we introduce "mixed languages" as strings of hybrid symbols and a new mixed weighted finite state automaton that recognizes a class of regular mixed languages, generalizing closure properties of regular languages.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2104.00755",
  "bibtex": `@article{martins2021reconciling,
  title={Reconciling the Discrete-Continuous Divide: Towards a Mathematical Theory of Sparse Communication},
  author={Martins, Andr{\\'e} FT},
  journal={arXiv preprint arXiv:2104.00755},
  year={2021}
}`
}
},
  {
  "id": 42,
  "title": "Smoothing and Shrinking the Sparse Seq2Seq Search Space",
  "authors": "Ben Peters, Andr\u00e9 F. T. Martins",
  "venue": "NAACL",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>Current sequence-to-sequence models are trained to minimize cross-entropy and use softmax to compute the locally normalized probabilities over target sequences. While this setup has led to strong results in a variety of tasks, one unsatisfying aspect is its length bias: models give high scores to short, inadequate hypotheses and often make the empty string the argmax -- the so-called cat got your tongue problem. Recently proposed entmax-based sparse sequence-to-sequence models present a possible solution, since they can shrink the search space by assigning zero probability to bad hypotheses, but their ability to handle word-level tasks with transformers has never been tested. In this work, we show that entmax-based models effectively solve the cat got your tongue problem, removing a major source of model error for neural machine translation. In addition, we generalize label smoothing, a critical regularization technique, to the broader family of Fenchel-Young losses, which includes both cross-entropy and the entmax losses. Our resulting label-smoothed entmax loss models set a new state of the art on multilingual grapheme-to-phoneme conversion and deliver improvements and better calibration properties on cross-lingual morphological inflection and machine translation for 6 language pairs.</p>`,
  "streams": [
  "attention",
  "multilingual-translation",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2021.naacl-main.210/",
  "code": "https://github.com/deep-spin/S7",
  "bibtex": `@inproceedings{peters-martins-2021-smoothing,
    title = "Smoothing and Shrinking the Sparse {S}eq2{S}eq Search Space",
    author = "Peters, Ben  and
      Martins, Andr{\\'e} F. T.",
    editor = "Toutanova, Kristina  and
      Rumshisky, Anna  and
      Zettlemoyer, Luke  and
      Hakkani-Tur, Dilek  and
      Beltagy, Iz  and
      Bethard, Steven  and
      Cotterell, Ryan  and
      Chakraborty, Tanmoy  and
      Zhou, Yichao",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.210/",
    doi = "10.18653/v1/2021.naacl-main.210",
    pages = "2642--2654",
    abstract = "Current sequence-to-sequence models are trained to minimize cross-entropy and use softmax to compute the locally normalized probabilities over target sequences. While this setup has led to strong results in a variety of tasks, one unsatisfying aspect is its length bias: models give high scores to short, inadequate hypotheses and often make the empty string the argmax{---}the so-called cat got your tongue problem. Recently proposed entmax-based sparse sequence-to-sequence models present a possible solution, since they can shrink the search space by assigning zero probability to bad hypotheses, but their ability to handle word-level tasks with transformers has never been tested. In this work, we show that entmax-based models effectively solve the cat got your tongue problem, removing a major source of model error for neural machine translation. In addition, we generalize label smoothing, a critical regularization technique, to the broader family of Fenchel-Young losses, which includes both cross-entropy and the entmax losses. Our resulting label-smoothed entmax loss models set a new state of the art on multilingual grapheme-to-phoneme conversion and deliver improvements and better calibration properties on cross-lingual morphological inflection and machine translation for 7 language pairs."
}`
}
},
  {
  "id": 41,
  "title": "Multimodal Continuous Visual Attention Mechanisms",
  "authors": "Antonio Farinhas, Andre F. T. Martins, Pedro M. Q. Aguiar",
  "venue": "ICCV",
  "year": 2021,
  "type": "conference",
  "abstract": `<p>Visual attention mechanisms are a key component of neural network models for computer vision. By focusing on a discrete set of objects or image regions, these mechanisms identify the most relevant features and use them to build more powerful representations. Recently, continuous-domain alternatives to discrete attention models have been proposed, which exploit the continuity of images. These approaches model attention as simple unimodal densities (e.g. a Gaussian), making them less suitable to deal with images whose region of interest has a complex shape or is composed of multiple non-contiguous patches. In this paper, we introduce a new continuous attention mechanism that produces multimodal densities, in the form of mixtures of Gaussians. We use the EM algorithm to obtain a clustering of relevant regions in the image, and a description length penalty to select the number of components in the mixture. Our densities decompose as a linear combination of unimodal attention mechanisms, enabling closed-form Jacobians for the backpropagation step. Experiments on visual question answering in the VQA-v2 dataset show competitive accuracies and a selection of regions that mimics human attention more closely in VQA-HAT. We present several examples that suggest how multimodal attention maps are naturally more interpretable than their unimodal counterparts, showing the ability of our model to automatically segregate objects from ground in complex scenes.</p>`,
  "streams": [
  "multimodal"
],
  "links": {
  "paper": "https://arxiv.org/abs/2104.03046",
  "bibtex": `@inproceedings{farinhas2021multimodal,
  title={Multimodal continuous visual attention mechanisms},
  author={Farinhas, Ant{\\'o}nio and Martins, Andr{\\'e} FT and Aguiar, Pedro MQ},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1047--1056},
  year={2021}
}`
}
},
  {
  "id": 40,
  "title": "Sparse and Continuous Attention Mechanisms",
  "authors": "Andr\u00e9 F. T. Martins, Ant\u00f3nio Farinhas, Marcos V. Treviso, Vlad Niculae, Pedro M. Q. Aguiar, M\u00e1rio A. T. Figueiredo",
  "venue": "NeurIPS",
  "year": 2020,
  "type": "conference",
  "award": "Spotlight",
  "abstract": `<p>Exponential families are widely used in machine learning; they include many distributions in continuous and discrete domains (e.g., Gaussian, Dirichlet, Poisson, and categorical distributions via the softmax transformation). Distributions in each of these families have fixed support. In contrast, for finite domains, there has been recent work on sparse alternatives to softmax (e.g., sparsemax and alpha-entmax), which have varying support, being able to assign zero probability to irrelevant categories. These discrete sparse mappings have been used for improving interpretability of neural attention mechanisms. This paper expands that work in two directions: first, we extend alpha-entmax to continuous domains, revealing a link with Tsallis statistics and deformed exponential families. Second, we introduce continuous-domain attention mechanisms, deriving efficient gradient backpropagation algorithms for alpha in {1,2}. Experiments on attention-based text classification, machine translation, and visual question answering illustrate the use of continuous attention in 1D and 2D, showing that it allows attending to time intervals and compact regions.</p>`,
  "streams": [
  "attention",
  "multimodal"
],
  "links": {
  "paper": "https://arxiv.org/abs/2006.07214",
  "bibtex": `@article{martins2020sparse,
  title={Sparse and continuous attention mechanisms},
  author={Martins, Andr{\\'e} and Farinhas, Ant{\\'o}nio and Treviso, Marcos and Niculae, Vlad and Aguiar, Pedro and Figueiredo, Mario},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={20989--21001},
  year={2020}
}`
}
},
  {
  "id": 39,
  "title": "The Explanation Game: Towards Prediction Explainability through Sparse Communication",
  "authors": "Marcos Treviso, Andr\u00e9 F. T. Martins",
  "venue": "BlackBoxNLP",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>Explainability is a topic of growing importance in NLP. In this work, we provide a unified perspective of explainability as a communication problem between an explainer and a layperson about a classifier’s decision. We use this framework to compare several explainers, including gradient methods, erasure, and attention mechanisms, in terms of their communication success. In addition, we reinterpret these methods in the light of classical feature selection, and use this as inspiration for new embedded explainers, through the use of selective, sparse attention. Experiments in text classification and natural language inference, using different configurations of explainers and laypeople (including both machines and humans), reveal an advantage of attention-based explainers over gradient and erasure methods, and show that selective attention is a simpler alternative to stochastic rationalizers. Human experiments show strong results on text classification with post-hoc explainers trained to optimize communication success.</p>`,
  "streams": [
  "attention",
  "interpretability"
],
  "links": {
  "paper": "https://aclanthology.org/2020.blackboxnlp-1.10/",
  "bibtex": `@inproceedings{treviso-martins-2020-explanation,
    title = "The Explanation Game: Towards Prediction Explainability through Sparse Communication",
    author = "Treviso, Marcos  and
      Martins, Andr{\\'e} F. T.",
    editor = "Alishahi, Afra  and
      Belinkov, Yonatan  and
      Chrupa{\\l}a, Grzegorz  and
      Hupkes, Dieuwke  and
      Pinter, Yuval  and
      Sajjad, Hassan",
    booktitle = "Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.blackboxnlp-1.10/",
    doi = "10.18653/v1/2020.blackboxnlp-1.10",
    pages = "107--118",
    abstract = "Explainability is a topic of growing importance in NLP. In this work, we provide a unified perspective of explainability as a communication problem between an explainer and a layperson about a classifier{'}s decision. We use this framework to compare several explainers, including gradient methods, erasure, and attention mechanisms, in terms of their communication success. In addition, we reinterpret these methods in the light of classical feature selection, and use this as inspiration for new embedded explainers, through the use of selective, sparse attention. Experiments in text classification and natural language inference, using different configurations of explainers and laypeople (including both machines and humans), reveal an advantage of attention-based explainers over gradient and erasure methods, and show that selective attention is a simpler alternative to stochastic rationalizers. Human experiments show strong results on text classification with post-hoc explainers trained to optimize communication success."
}`
}
},
  {
  "id": 38,
  "title": "LP-SparseMAP: Differentiable Relaxed Optimization for Sparse Structured Prediction\n",
  "authors": "Vlad Niculae, Andr\u00e9 F. T. Martins",
  "venue": "ICML",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>Structured predictors require solving a combinatorial optimization problem over a large number of structures, such as dependency trees or alignments. When embedded as structured hidden layers in a neural net, argmin differentiation and efficient gradient computation are further required. Recently, SparseMAP has been proposed as a differentiable, sparse alternative to maximum a posteriori (MAP) and marginal inference. SparseMAP returns an interpretable combination of a small number of structures; its sparsity being the key to efficient optimization. However, SparseMAP requires access to an exact MAP oracle in the structured model, excluding, e.g., loopy graphical models or logic constraints, which generally require approximate inference. In this paper, we introduce LP-SparseMAP, an extension of SparseMAP addressing this limitation via a local polytope relaxation. LP-SparseMAP uses the flexible and powerful language of factor graphs to define expressive hidden structures, supporting coarse decompositions, hard logic constraints, and higher-order correlations. We derive the forward and backward algorithms needed for using LP-SparseMAP as a structured hidden or output layer. Experiments in three structured tasks show benefits versus SparseMAP and Structured SVM.</p>`,
  "streams": [
  "interpretability",
  "theory"
],
  "links": {
  "paper": "http://proceedings.mlr.press/v119/niculae20a/niculae20a.pdf",
  "code": "https://github.com/deep-spin/lp-sparsemap",
  "bibtex": `@InProceedings{pmlr-v119-niculae20a,
  title =          {{LP}-{S}parse{MAP}: Differentiable Relaxed Optimization for Sparse Structured Prediction},
  author =       {Niculae, Vlad and Martins, Andre},
  booktitle =          {Proceedings of the 37th International Conference on Machine Learning},
  pages =          {7348--7359},
  year =          {2020},
  editor =          {III, Hal Daumé and Singh, Aarti},
  volume =          {119},
  series =          {Proceedings of Machine Learning Research},
  month =          {13--18 Jul},
  publisher =    {PMLR},
  pdf =          {http://proceedings.mlr.press/v119/niculae20a/niculae20a.pdf},
  url =          {https://proceedings.mlr.press/v119/niculae20a.html},
  abstract =          {Structured predictors require solving a combinatorial optimization problem over a large number of structures, such as dependency trees or alignments. When embedded as structured hidden layers in a neural net, argmin differentiation and efficient gradient computation are further required. Recently, SparseMAP has been proposed as a differentiable, sparse alternative to maximum a posteriori (MAP) and marginal inference. SparseMAP returns an interpretable combination of a small number of structures; its sparsity being the key to efficient optimization. However, SparseMAP requires access to an exact MAP oracle in the structured model, excluding, e.g., loopy graphical models or logic constraints, which generally require approximate inference. In this paper, we introduce LP-SparseMAP, an extension of SparseMAP addressing this limitation via a local polytope relaxation. LP-SparseMAP uses the flexible and powerful language of factor graphs to define expressive hidden structures, supporting coarse decompositions, hard logic constraints, and higher-order correlations. We derive the forward and backward algorithms needed for using LP-SparseMAP as a structured hidden or output layer. Experiments in three structured tasks show benefits versus SparseMAP and Structured SVM.}
}`
}
},
  {
  "id": 37,
  "title": "IST-Unbabel Participation in the WMT20 Quality Estimation Shared Task",
  "authors": "Jo\u00e3o Martinho Moura, Miguel Vera, Daan van Stigt, F\u00e1bio Kepler, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>We present the joint contribution of IST and Unbabel to the WMT 2020 Shared Task on Quality Estimation. Our team participated on all tracks (Direct Assessment, Post-Editing Effort, Document-Level), encompassing a total of 14 submissions. Our submitted systems were developed by extending the OpenKiwi framework to a transformer-based predictor-estimator architecture, and to cope with glass-box, uncertainty-based features coming from neural machine translation systems.</p>`,
  "streams": [
  "evaluation-metrics",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2020.wmt-1.119/",
  "bibtex": `@inproceedings{moura-etal-2020-ist,
    title = "{IST}-Unbabel Participation in the {WMT}20 Quality Estimation Shared Task",
    author = "Moura, Jo{\\~a}o  and
      Vera, Miguel  and
      van Stigt, Daan  and
      Kepler, Fabio  and
      Martins, Andr{\\'e} F. T.",
    editor = {Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Graham, Yvette  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo},
    booktitle = "Proceedings of the Fifth Conference on Machine Translation",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.wmt-1.119/",
    pages = "1029--1036",
    abstract = "We present the joint contribution of IST and Unbabel to the WMT 2020 Shared Task on Quality Estimation. Our team participated on all tracks (Direct Assessment, Post-Editing Effort, Document-Level), encompassing a total of 14 submissions. Our submitted systems were developed by extending the OpenKiwi framework to a transformer-based predictor-estimator architecture, and to cope with glass-box, uncertainty-based features coming from neural machine translation systems."
}`
}
},
  {
  "id": 36,
  "title": "Findings of the WMT 2020 Shared Task on Chat Translation",
  "authors": "M. Amin Farajian, Ant\u00f3nio V. Lopes, Andr\u00e9 F. T. Martins, Sameen Maruf, Gholamreza Haffari",
  "venue": "WMT",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>We report the results of the first edition of the WMT shared task on chat translation. The task consisted of translating bilingual conversational text, in particular customer support chats for the English-German language pair (English agent, German customer). This task varies from the other translation shared tasks, i.e. news and biomedical, mainly due to the fact that the conversations are bilingual, less planned, more informal, and often ungrammatical. Furthermore, such conversations are usually characterized by shorter and simpler sentences and contain more pronouns. We received 14 submissions from 6 participating teams, all of them covering both directions, i.e. En-&gt;De for agent utterances and De-&gt;En for customer messages. We used automatic metrics (BLEU and TER) for evaluating the translations of both agent and customer messages and human document-level direct assessments (DDA) to evaluate the agent translations.</p>`,
  "streams": [
  "dialogue-context",
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/2020.wmt-1.3/",
  "bibtex": `@inproceedings{farajian-etal-2020-findings,
    title = "Findings of the {WMT} 2020 Shared Task on Chat Translation",
    author = "Farajian, M. Amin  and
      Lopes, Ant{\\'o}nio V.  and
      Martins, Andr{\\'e} F. T.  and
      Maruf, Sameen  and
      Haffari, Gholamreza",
    editor = {Barrault, Lo{\\"i}c  and
      Bojar, Ond{\\v{r}}ej  and
      Bougares, Fethi  and
      Chatterjee, Rajen  and
      Costa-juss{\\\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Fraser, Alexander  and
      Graham, Yvette  and
      Guzman, Paco  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\\'e}  and
      Morishita, Makoto  and
      Monz, Christof  and
      Nagata, Masaaki  and
      Nakazawa, Toshiaki  and
      Negri, Matteo},
    booktitle = "Proceedings of the Fifth Conference on Machine Translation",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.wmt-1.3/",
    pages = "65--75",
    abstract = "We report the results of the first edition of the WMT shared task on chat translation. The task consisted of translating bilingual conversational text, in particular customer support chats for the English-German language pair (English agent, German customer). This task varies from the other translation shared tasks, i.e. news and biomedical, mainly due to the fact that the conversations are bilingual, less planned, more informal, and often ungrammatical. Furthermore, such conversations are usually characterized by shorter and simpler sentences and contain more pronouns. We received 14 submissions from 6 participating teams, all of them covering both directions, i.e. En-{\\ensuremath{>}}De for agent utterances and De-{\\ensuremath{>}}En for customer messages. We used automatic metrics (BLEU and TER) for evaluating the translations of both agent and customer messages and human document-level direct assessments (DDA) to evaluate the agent translations."
}`
}
},
  {
  "id": 35,
  "title": "Project MAIA: Multilingual AI Agent Assistant",
  "authors": "Andr\u00e9 F. T. Martins, Jo\u00e4o Gra\u00e7a, Paulo Dimas, Helena Moniz, Graham Neubig",
  "venue": "EAMT",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>This paper presents the Multilingual Artificial Intelligence Agent Assistant (MAIA), a project led by Unbabel with the collaboration of CMU, INESC-ID and IT Lisbon. MAIA will employ cutting-edge machine learning and natural language processing technologies to build multilingual AI agent assistants, eliminating language barriers. MAIA’s translation layer will empower human agents to provide customer support in real-time, in any language, with human quality.</p>`,
  "streams": [
  "dialogue-context",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2020.eamt-1.68/",
  "bibtex": `@inproceedings{martins-etal-2020-project,
    title = "Project {MAIA}: Multilingual {AI} Agent Assistant",
    author = "Martins, Andr{\\'e} F. T.  and
      Graca, Joao  and
      Dimas, Paulo  and
      Moniz, Helena  and
      Neubig, Graham",
    editor = "Martins, Andr{\\'e}  and
      Moniz, Helena  and
      Fumega, Sara  and
      Martins, Bruno  and
      Batista, Fernando  and
      Coheur, Luisa  and
      Parra, Carla  and
      Trancoso, Isabel  and
      Turchi, Marco  and
      Bisazza, Arianna  and
      Moorkens, Joss  and
      Guerberof, Ana  and
      Nurminen, Mary  and
      Marg, Lena  and
      Forcada, Mikel L.",
    booktitle = "Proceedings of the 22nd Annual Conference of the European Association for Machine Translation",
    month = nov,
    year = "2020",
    address = "Lisboa, Portugal",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2020.eamt-1.68/",
    pages = "495--496",
    abstract = "This paper presents the Multilingual Artificial Intelligence Agent Assistant (MAIA), a project led by Unbabel with the collaboration of CMU, INESC-ID and IT Lisbon. MAIA will employ cutting-edge machine learning and natural language processing technologies to build multilingual AI agent assistants, eliminating language barriers. MAIA{'}s translation layer will empower human agents to provide customer support in real-time, in any language, with human quality."
}`
}
},
  {
  "id": 34,
  "title": "MLQE-PE: A Multilingual Quality Estimation and Post-Editing Dataset",
  "authors": "Marina Fomicheva, Shuo Sun, Erick Fonseca, Chrysoula Zerva, Fr\u00e9d\u00e9ric Blain, Vishrav Chaudhary, Francisco Guzm\u00e1n, Nina Lopatina, Lucia Specia, Andr\u00e9 F. T. Martins",
  "venue": "LREC",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>We present MLQE-PE, a new dataset for Machine Translation (MT) Quality Estimation (QE) and Automatic Post-Editing (APE). The dataset contains annotations for eleven language pairs, including both high- and low-resource languages. Specifically, it is annotated for translation quality with human labels for up to 10,000 translations per language pair in the following formats: sentence-level direct assessments and post-editing effort, and word-level binary good/bad labels. Apart from the quality-related scores, each source-translation sentence pair is accompanied by the corresponding post-edited sentence, as well as titles of the articles where the sentences were extracted from, and information on the neural MT models used to translate the text. We provide a thorough description of the data collection and annotation process as well as an analysis of the annotation distribution for each language pair. We also report the performance of baseline systems trained on the MLQE-PE dataset. The dataset is freely available and has already been used for several WMT shared tasks.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2022.lrec-1.530/",
  "bibtex": `@inproceedings{fomicheva-etal-2022-mlqe,
    title = "{MLQE}-{PE}: A Multilingual Quality Estimation and Post-Editing Dataset",
    author = "Fomicheva, Marina  and
      Sun, Shuo  and
      Fonseca, Erick  and
      Zerva, Chrysoula  and
      Blain, Fr{\\'e}d{\\'e}ric  and
      Chaudhary, Vishrav  and
      Guzm{\\'a}n, Francisco  and
      Lopatina, Nina  and
      Specia, Lucia  and
      Martins, Andr{\\'e} F. T.",
    editor = "Calzolari, Nicoletta  and
      B{\\'e}chet, Fr{\\'e}d{\\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\\'e}l{\\\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.530/",
    pages = "4963--4974",
    abstract = "We present MLQE-PE, a new dataset for Machine Translation (MT) Quality Estimation (QE) and Automatic Post-Editing (APE). The dataset contains annotations for eleven language pairs, including both high- and low-resource languages. Specifically, it is annotated for translation quality with human labels for up to 10,000 translations per language pair in the following formats: sentence-level direct assessments and post-editing effort, and word-level binary good/bad labels. Apart from the quality-related scores, each source-translation sentence pair is accompanied by the corresponding post-edited sentence, as well as titles of the articles where the sentences were extracted from, and information on the neural MT models used to translate the text. We provide a thorough description of the data collection and annotation process as well as an analysis of the annotation distribution for each language pair. We also report the performance of baseline systems trained on the MLQE-PE dataset. The dataset is freely available and has already been used for several WMT shared tasks."
}`
}
},
  {
  "id": 33,
  "title": "Understanding the Mechanics of SPIGOT: Surrogate Gradients for Latent Structure Learning",
  "authors": "Tsvetomila Mihaylova, Vlad Niculae, Andr\u00e9 F. T. Martins",
  "venue": "EMNLP",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>Latent structure models are a powerful tool for modeling language data: they can mitigate the error propagation and annotation bottleneck in pipeline systems, while simultaneously uncovering linguistic insights about the data. One challenge with end-to-end training of these models is the argmax operation, which has null gradient. In this paper, we focus on surrogate gradients, a popular strategy to deal with this problem. We explore latent structure learning through the angle of pulling back the downstream learning objective. In this paradigm, we discover a principled motivation for both the straight-through estimator (STE) as well as the recently-proposed SPIGOT – a variant of STE for structured models. Our perspective leads to new algorithms in the same family. We empirically compare the known and the novel pulled-back estimators against the popular alternatives, yielding new insight for practitioners and revealing intriguing failure cases.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2020.emnlp-main.171/",
  "code": "https://github.com/deep-spin/understanding-spigot",
  "bibtex": `@inproceedings{mihaylova-etal-2020-understanding,
    title = "Understanding the Mechanics of {SPIGOT}: Surrogate Gradients for Latent Structure Learning",
    author = "Mihaylova, Tsvetomila  and
      Niculae, Vlad  and
      Martins, Andr{\\'e} F. T.",
    editor = "Webber, Bonnie  and
      Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.171/",
    doi = "10.18653/v1/2020.emnlp-main.171",
    pages = "2186--2202",
    abstract = "Latent structure models are a powerful tool for modeling language data: they can mitigate the error propagation and annotation bottleneck in pipeline systems, while simultaneously uncovering linguistic insights about the data. One challenge with end-to-end training of these models is the argmax operation, which has null gradient. In this paper, we focus on surrogate gradients, a popular strategy to deal with this problem. We explore latent structure learning through the angle of pulling back the downstream learning objective. In this paradigm, we discover a principled motivation for both the straight-through estimator (STE) as well as the recently-proposed SPIGOT {--} a variant of STE for structured models. Our perspective leads to new algorithms in the same family. We empirically compare the known and the novel pulled-back estimators against the popular alternatives, yielding new insight for practitioners and revealing intriguing failure cases."
}`
}
},
  {
  "id": 32,
  "title": "One-Size-Fits-All Multilingual Models",
  "authors": "Ben Peters, Andr\u00e9 F. T. Martins",
  "venue": "SIGMORPHON",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>This paper presents DeepSPIN’s submissions to Tasks 0 and 1 of the SIGMORPHON 2020 Shared Task. For both tasks, we present multilingual models, training jointly on data in all languages. We perform no language-specific hyperparameter tuning – each of our submissions uses the same model for all languages. Our basic architecture is the sparse sequence-to-sequence model with entmax attention and loss, which allows our models to learn sparse, local alignments while still being trainable with gradient-based techniques. For Task 1, we achieve strong performance with both RNN- and transformer-based sparse models. For Task 0, we extend our RNN-based model to a multi-encoder set-up in which separate modules encode the lemma and inflection sequences. Despite our models’ lack of language-specific tuning, they tie for first in Task 0 and place third in Task 1.</p>`,
  "streams": [
  "attention",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/2020.sigmorphon-1.4/",
  "bibtex": `@inproceedings{peters-martins-2020-one,
    title = "One-Size-Fits-All Multilingual Models",
    author = "Peters, Ben  and
      Martins, Andr{\\'e} F. T.",
    editor = "Nicolai, Garrett  and
      Gorman, Kyle  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 17th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.sigmorphon-1.4/",
    doi = "10.18653/v1/2020.sigmorphon-1.4",
    pages = "63--69",
    abstract = "This paper presents DeepSPIN{'}s submissions to Tasks 0 and 1 of the SIGMORPHON 2020 Shared Task. For both tasks, we present multilingual models, training jointly on data in all languages. We perform no language-specific hyperparameter tuning {--} each of our submissions uses the same model for all languages. Our basic architecture is the sparse sequence-to-sequence model with entmax attention and loss, which allows our models to learn sparse, local alignments while still being trainable with gradient-based techniques. For Task 1, we achieve strong performance with both RNN- and transformer-based sparse models. For Task 0, we extend our RNN-based model to a multi-encoder set-up in which separate modules encode the lemma and inflection sequences. Despite our models' lack of language-specific tuning, they tie for first in Task 0 and place third in Task 1."
}`
}
},
  {
  "id": 31,
  "title": "Revisiting Higher-Order Dependency Parsers",
  "authors": "Erick Fonseca, Andr\u00e9 F. T. Martins",
  "venue": "ACL",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>Neural encoders have allowed dependency parsers to shift from higher-order structured models to simpler first-order ones, making decoding faster and still achieving better accuracy than non-neural parsers. This has led to a belief that neural encoders can implicitly encode structural constraints, such as siblings and grandparents in a tree. We tested this hypothesis and found that neural parsers may benefit from higher-order features, even when employing a powerful pre-trained encoder, such as BERT. While the gains of higher-order features are small in the presence of a powerful encoder, they are consistent for long-range dependencies and long sentences. In particular, higher-order models are more accurate on full sentence parses and on the exact match of modifier lists, indicating that they deal better with larger, more complex structures.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2020.acl-main.776/",
  "bibtex": `@inproceedings{fonseca-martins-2020-revisiting,
    title = "Revisiting Higher-Order Dependency Parsers",
    author = "Fonseca, Erick  and
      Martins, Andr{\\'e} F. T.",
    editor = "Jurafsky, Dan  and
      Chai, Joyce  and
      Schluter, Natalie  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.776/",
    doi = "10.18653/v1/2020.acl-main.776",
    pages = "8795--8800",
    abstract = "Neural encoders have allowed dependency parsers to shift from higher-order structured models to simpler first-order ones, making decoding faster and still achieving better accuracy than non-neural parsers. This has led to a belief that neural encoders can implicitly encode structural constraints, such as siblings and grandparents in a tree. We tested this hypothesis and found that neural parsers may benefit from higher-order features, even when employing a powerful pre-trained encoder, such as BERT. While the gains of higher-order features are small in the presence of a powerful encoder, they are consistent for long-range dependencies and long sentences. In particular, higher-order models are more accurate on full sentence parses and on the exact match of modifier lists, indicating that they deal better with larger, more complex structures."
}`
}
},
  {
  "id": 30,
  "title": "Learning Non-Monotonic Automatic Post-Editing of Translations from Human Orderings",
  "authors": "Ant\u00f3nio G\u00f3is, Kyunghyun Cho, Andr\u00e9 F. T. Martins",
  "venue": "EAMT",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>Recent research in neural machine translation has explored flexible generation orders, as an alternative to left-to-right generation. However, training non-monotonic models brings a new complication: how to search for a good ordering when there is a combinatorial explosion of orderings arriving at the same final result? Also, how do these automatic orderings compare with the actual behaviour of human translators? Current models rely on manually built biases or are left to explore all possibilities on their own. In this paper, we analyze the orderings produced by human post-editors and use them to train an automatic post-editing system. We compare the resulting system with those trained with left-to-right and random post-editing orderings. We observe that humans tend to follow a nearly left-to-right order, but with interesting deviations, such as preferring to start by correcting punctuation or verbs.</p>`,
  "streams": [
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2020.eamt-1.22/",
  "bibtex": `@inproceedings{gois-etal-2020-learning,
    title = "Learning Non-Monotonic Automatic Post-Editing of Translations from Human Orderings",
    author = "G{\\'o}is, Ant{\\'o}nio  and
      Cho, Kyunghyun  and
      Martins, Andr{\\'e}",
    editor = "Martins, Andr{\\'e}  and
      Moniz, Helena  and
      Fumega, Sara  and
      Martins, Bruno  and
      Batista, Fernando  and
      Coheur, Luisa  and
      Parra, Carla  and
      Trancoso, Isabel  and
      Turchi, Marco  and
      Bisazza, Arianna  and
      Moorkens, Joss  and
      Guerberof, Ana  and
      Nurminen, Mary  and
      Marg, Lena  and
      Forcada, Mikel L.",
    booktitle = "Proceedings of the 22nd Annual Conference of the European Association for Machine Translation",
    month = nov,
    year = "2020",
    address = "Lisboa, Portugal",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2020.eamt-1.22/",
    pages = "205--214",
    abstract = "Recent research in neural machine translation has explored flexible generation orders, as an alternative to left-to-right generation. However, training non-monotonic models brings a new complication: how to search for a good ordering when there is a combinatorial explosion of orderings arriving at the same final result? Also, how do these automatic orderings compare with the actual behaviour of human translators? Current models rely on manually built biases or are left to explore all possibilities on their own. In this paper, we analyze the orderings produced by human post-editors and use them to train an automatic post-editing system. We compare the resulting system with those trained with left-to-right and random post-editing orderings. We observe that humans tend to follow a nearly left-to-right order, but with interesting deviations, such as preferring to start by correcting punctuation or verbs."
}`
}
},
  {
  "id": 29,
  "title": "Sparse Text Generation",
  "authors": "Pedro Henrique Martins, Zita Marinho, Andr\u00e9 F. T. Martins",
  "venue": "EMNLP",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>Current state-of-the-art text generators build on powerful language models such as GPT-2, achieving impressive performance. However, to avoid degenerate text, they require sampling from a modified softmax, via temperature parameters or ad-hoc truncation techniques, as in top-$k$ or nucleus sampling. This creates a mismatch between training and testing conditions. In this paper, we use the recently introduced entmax transformation to train and sample from a natively sparse language model, avoiding this mismatch. The result is a text generator with favorable performance in terms of fluency and consistency, fewer repetitions, and n-gram diversity closer to human text. In order to evaluate our model, we propose three new metrics for comparing sparse or truncated distributions: $\\epsilon$-perplexity, sparsemax score, and Jensen-Shannon divergence. Human-evaluated experiments in story completion and dialogue generation show that entmax sampling leads to more engaging and coherent stories and conversations.</p>`,
  "streams": [
  "attention",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/2020.emnlp-main.348/",
  "code": "https://github.com/deep-spin/sparse_text_generation",
  "bibtex": `@inproceedings{martins-etal-2020-sparse,
    title = "Sparse Text Generation",
    author = "Martins, Pedro Henrique  and
      Marinho, Zita  and
      Martins, Andr{\\'e} F. T.",
    editor = "Webber, Bonnie  and
      Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.348/",
    doi = "10.18653/v1/2020.emnlp-main.348",
    pages = "4252--4273",
    abstract = "Current state-of-the-art text generators build on powerful language models such as GPT-2, achieving impressive performance. However, to avoid degenerate text, they require sampling from a modified softmax, via temperature parameters or ad-hoc truncation techniques, as in top-$k$ or nucleus sampling. This creates a mismatch between training and testing conditions. In this paper, we use the recently introduced entmax transformation to train and sample from a natively sparse language model, avoiding this mismatch. The result is a text generator with favorable performance in terms of fluency and consistency, fewer repetitions, and n-gram diversity closer to human text. In order to evaluate our model, we propose three new metrics for comparing sparse or truncated distributions: $\\epsilon$-perplexity, sparsemax score, and Jensen-Shannon divergence. Human-evaluated experiments in story completion and dialogue generation show that entmax sampling leads to more engaging and coherent stories and conversations."
}`
}
},
  {
  "id": 28,
  "title": "Document-level Neural MT: A Systematic Comparison",
  "authors": "Ant\u00f3nio V. Lopes, M. Amin Farajian, Rachel Bawden, Michael Zhang, Andr\u00e9 F. T. Martins",
  "venue": "EAMT",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>In this paper we provide a systematic comparison of existing and new document-level neural machine translation solutions. As part of this comparison, we introduce and evaluate a document-level variant of the recently proposed Star Transformer architecture. In addition to using the traditional metric BLEU, we report the accuracy of the models in handling anaphoric pronoun translation as well as coherence and cohesion using contrastive test sets. Finally, we report the results of human evaluation in terms of Multidimensional Quality Metrics (MQM) and analyse the correlation of the results obtained by the automatic metrics with human judgments.</p>`,
  "streams": [
  "dialogue-context",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/2020.eamt-1.24/",
  "bibtex": `@inproceedings{lopes-etal-2020-document,
    title = "Document-level Neural {MT}: A Systematic Comparison",
    author = "Lopes, Ant{\\'o}nio  and
      Farajian, M. Amin  and
      Bawden, Rachel  and
      Zhang, Michael  and
      Martins, Andr{\\'e} F. T.",
    editor = "Martins, Andr{\\'e}  and
      Moniz, Helena  and
      Fumega, Sara  and
      Martins, Bruno  and
      Batista, Fernando  and
      Coheur, Luisa  and
      Parra, Carla  and
      Trancoso, Isabel  and
      Turchi, Marco  and
      Bisazza, Arianna  and
      Moorkens, Joss  and
      Guerberof, Ana  and
      Nurminen, Mary  and
      Marg, Lena  and
      Forcada, Mikel L.",
    booktitle = "Proceedings of the 22nd Annual Conference of the European Association for Machine Translation",
    month = nov,
    year = "2020",
    address = "Lisboa, Portugal",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2020.eamt-1.24/",
    pages = "225--234",
    abstract = "In this paper we provide a systematic comparison of existing and new document-level neural machine translation solutions. As part of this comparison, we introduce and evaluate a document-level variant of the recently proposed Star Transformer architecture. In addition to using the traditional metric BLEU, we report the accuracy of the models in handling anaphoric pronoun translation as well as coherence and cohesion using contrastive test sets. Finally, we report the results of human evaluation in terms of Multidimensional Quality Metrics (MQM) and analyse the correlation of the results obtained by the automatic metrics with human judgments."
}`
}
},
  {
  "id": 27,
  "title": "Efficient Marginalization of Discrete and Structured Latent Variables via Sparsity",
  "authors": "Gon\u00e7alo M. Correia, Vlad Niculae, Wilker Aziz, Andr\u00e9 F. T. Martins",
  "venue": "arXiv (Cornell University)",
  "year": 2020,
  "type": "conference",
  "abstract": `<p>Training neural network models with discrete (categorical or structured) latent variables can be computationally challenging, due to the need for marginalization over large or combinatorial sets. To circumvent this issue, one typically resorts to sampling-based approximations of the true marginal, requiring noisy gradient estimators (e.g., score function estimator) or continuous relaxations with lower-variance reparameterized gradients (e.g., Gumbel-Softmax). In this paper, we propose a new training strategy which replaces these estimators by an exact yet efficient marginalization. To achieve this, we parameterize discrete distributions over latent assignments using differentiable sparse mappings: sparsemax and its structured counterparts. In effect, the support of these distributions is greatly reduced, which enables efficient marginalization. We report successful results in three tasks covering a range of latent variable modeling applications: a semisupervised deep generative model, a latent communication game, and a generative model with a bit-vector latent representation. In all cases, we obtain good performance while still achieving the practicality of sampling-based approximations.</p>`,
  "streams": [
  "attention",
  "efficiency",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/2007.01919",
  "code": "https://github.com/deep-spin/sparse-marginalization-lvm",
  "bibtex": `@article{correia2020efficient,
  title={Efficient marginalization of discrete and structured latent variables via sparsity},
  author={Correia, Gon{\\c{c}}alo and Niculae, Vlad and Aziz, Wilker and Martins, Andr{\\'e}},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={11789--11802},
  year={2020}
}`
}
},
  {
  "id": 26,
  "title": "Learning with Fenchel-Young Losses",
  "authors": "Mathieu Blondel, Andr\u00e9 F. T. Martins, Vlad Niculae",
  "venue": "JMLR",
  "year": 2020,
  "type": "journal",
  "abstract": `<p>Over the past decades, numerous loss functions have been been proposed for a variety of supervised learning tasks, including regression, classification, ranking, and more generally structured prediction. Understanding the core principles and theoretical properties underpinning these losses is key to choose the right loss for the right problem, as well as to create new losses which combine their strengths. In this paper, we introduce Fenchel-Young losses, a generic way to construct a convex loss function for a regularized prediction function. We provide an in-depth study of their properties in a very broad setting, covering all the aforementioned supervised learning tasks, and revealing new connections between sparsity, generalized entropies, and separation margins. We show that Fenchel-Young losses unify many well-known loss functions and allow to create useful new ones easily. Finally, we derive efficient predictive and training algorithms, making Fenchel-Young losses appealing both in theory and practice.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/1901.02324",
  "bibtex": `@article{blondel2020learning,
  title={Learning with fenchel-young losses},
  author={Blondel, Mathieu and Martins, Andr{\\'e} FT and Niculae, Vlad},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={35},
  pages={1--69},
  year={2020}
}`
}
},
  {
  "id": 25,
  "title": "OpenKiwi: An Open Source Framework for Quality Estimation",
  "authors": "Fabio Kepler, Jonay Tr\u00e9nous, Marcos Treviso, Miguel Vera, Andr\u00e9 F. T. Martins",
  "venue": "ACL",
  "year": 2019,
  "type": "conference",
  "award": "Best Demo",
  "abstract": `<p>We introduce OpenKiwi, a PyTorch-based open source framework for translation quality estimation. OpenKiwi supports training and testing of word-level and sentence-level quality estimation systems, implementing the winning systems of the WMT 2015-18 quality estimation campaigns. We benchmark OpenKiwi on two datasets from WMT 2018 (English-German SMT and NMT), yielding state-of-the-art performance on the word-level tasks and near state-of-the-art in the sentence-level tasks.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/P19-3020/",
  "bibtex": `@inproceedings{kepler-etal-2019-openkiwi,
    title = "{O}pen{K}iwi: An Open Source Framework for Quality Estimation",
    author = "Kepler, Fabio  and
      Tr{\\'e}nous, Jonay  and
      Treviso, Marcos  and
      Vera, Miguel  and
      Martins, Andr{\\'e} F. T.",
    editor = "Costa-juss{\\\`a}, Marta R.  and
      Alfonseca, Enrique",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-3020/",
    doi = "10.18653/v1/P19-3020",
    pages = "117--122",
    abstract = "We introduce OpenKiwi, a Pytorch-based open source framework for translation quality estimation. OpenKiwi supports training and testing of word-level and sentence-level quality estimation systems, implementing the winning systems of the WMT 2015{--}18 quality estimation campaigns. We benchmark OpenKiwi on two datasets from WMT 2018 (English-German SMT and NMT), yielding state-of-the-art performance on the word-level tasks and near state-of-the-art in the sentence-level tasks."
}`
}
},
  {
  "id": 24,
  "title": "Unbabel\u2019s Participation in the WMT19 Translation Quality Estimation Shared Task",
  "authors": "Fabio Kepler, Jonay Tr\u00e9nous, Marcos Treviso, Miguel Vera, Ant\u00f3nio G\u00f3is, M. Amin Farajian, Ant\u00f3nio V. Lopes, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>We present the contribution of the Unbabel team to the WMT 2019 Shared Task on Quality Estimation. We participated on the word, sentence, and document-level tracks, encompassing 3 language pairs: English-German, English-Russian, and English-French. Our submissions build upon the recent OpenKiwi framework: we combine linear, neural, and predictor-estimator systems with new transfer learning approaches using BERT and XLM pre-trained models. We compare systems individually and propose new ensemble techniques for word and sentence-level predictions. We also propose a simple technique for converting word labels into document-level predictions. Overall, our submitted systems achieve the best results on all tracks and language pairs by a considerable margin.</p>`,
  "streams": [
  "evaluation-metrics",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/W19-5406/",
  "bibtex": `@inproceedings{kepler-etal-2019-unbabels,
    title = "Unbabel{'}s Participation in the {WMT}19 Translation Quality Estimation Shared Task",
    author = "Kepler, Fabio  and
      Tr{\\'e}nous, Jonay  and
      Treviso, Marcos  and
      Vera, Miguel  and
      G{\\'o}is, Ant{\\'o}nio  and
      Farajian, M. Amin  and
      Lopes, Ant{\\'o}nio V.  and
      Martins, Andr{\\'e} F. T.",
    editor = "Bojar, Ond{\\v{r}}ej  and
      Chatterjee, Rajen  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\\'e}  and
      Monz, Christof  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Post, Matt  and
      Turchi, Marco  and
      Verspoor, Karin",
    booktitle = "Proceedings of the Fourth Conference on Machine Translation (Volume 3: Shared Task Papers, Day 2)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5406/",
    doi = "10.18653/v1/W19-5406",
    pages = "78--84",
    abstract = "We present the contribution of the Unbabel team to the WMT 2019 Shared Task on Quality Estimation. We participated on the word, sentence, and document-level tracks, encompassing 3 language pairs: English-German, English-Russian, and English-French. Our submissions build upon the recent OpenKiwi framework: We combine linear, neural, and predictor-estimator systems with new transfer learning approaches using BERT and XLM pre-trained models. We compare systems individually and propose new ensemble techniques for word and sentence-level predictions. We also propose a simple technique for converting word labels into document-level predictions. Overall, our submitted systems achieve the best results on all tracks and language pairs by a considerable margin."
}`
}
},
  {
  "id": 23,
  "title": "Adaptively Sparse Transformers",
  "authors": "Gon\u00e7alo M. Correia, Vlad Niculae, Andr\u00e9 F. T. Martins",
  "venue": "EMNLP",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>Attention mechanisms have become ubiquitous in NLP. Recent architectures, notably the Transformer, learn powerful context-aware word representations through layered, multi-headed attention. The multiple heads learn diverse types of word relationships. However, with standard softmax attention, all attention heads are dense, assigning a non-zero weight to all context words. In this work, we introduce the adaptively sparse Transformer, wherein attention heads have flexible, context-dependent sparsity patterns. This sparsity is accomplished by replacing softmax with $\\alpha$-entmax: a differentiable generalization of softmax that allows low-scoring words to receive precisely zero weight. Moreover, we derive a method to automatically learn the $\\alpha$ parameter -- which controls the shape and sparsity of $\\alpha$-entmax -- allowing attention heads to choose between focused or spread-out behavior. Our adaptively sparse Transformer improves interpretability and head diversity when compared to softmax Transformers on machine translation datasets. Findings of the quantitative and qualitative analysis of our approach include that heads in different layers learn different sparsity preferences and tend to be more diverse in their attention distributions than softmax Transformers. Furthermore, at no cost in accuracy, sparsity in attention heads helps to uncover different head specializations.</p>`,
  "streams": [
  "attention",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/D19-1223/",
  "bibtex": `@inproceedings{correia-etal-2019-adaptively,
    title = "Adaptively Sparse Transformers",
    author = "Correia, Gon{\\c{c}}alo M.  and
      Niculae, Vlad  and
      Martins, Andr{\\'e} F. T.",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1223/",
    doi = "10.18653/v1/D19-1223",
    pages = "2174--2184",
    abstract = "Attention mechanisms have become ubiquitous in NLP. Recent architectures, notably the Transformer, learn powerful context-aware word representations through layered, multi-headed attention. The multiple heads learn diverse types of word relationships. However, with standard softmax attention, all attention heads are dense, assigning a non-zero weight to all context words. In this work, we introduce the adaptively sparse Transformer, wherein attention heads have flexible, context-dependent sparsity patterns. This sparsity is accomplished by replacing softmax with alpha-entmax: a differentiable generalization of softmax that allows low-scoring words to receive precisely zero weight. Moreover, we derive a method to automatically learn the alpha parameter {--} which controls the shape and sparsity of alpha-entmax {--} allowing attention heads to choose between focused or spread-out behavior. Our adaptively sparse Transformer improves interpretability and head diversity when compared to softmax Transformers on machine translation datasets. Findings of the quantitative and qualitative analysis of our approach include that heads in different layers learn different sparsity preferences and tend to be more diverse in their attention distributions than softmax Transformers. Furthermore, at no cost in accuracy, sparsity in attention heads helps to uncover different head specializations."
}`
}
},
  {
  "id": 22,
  "title": "Findings of the WMT 2019 Shared Tasks on Quality Estimation",
  "authors": "Erick Fonseca, Lisa Yankovskaya, Andr\u00e9 F. T. Martins, Mark Fishel, Christian Federmann",
  "venue": "WMT",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>We report the results of the WMT19 shared task on Quality Estimation, i.e. the task of predicting the quality of the output of machine translation systems given just the source text and the hypothesis translations. The task includes estimation at three granularity levels: word, sentence and document. A novel addition is evaluating sentence-level QE against human judgments: in other words, designing MT metrics that do not need a reference translation. This year we include three language pairs, produced solely by neural machine translation systems. Participating teams from eleven institutions submitted a variety of systems to different task variants and language pairs.</p>`,
  "streams": [
  "evaluation-metrics",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/W19-5401/",
  "bibtex": `@inproceedings{fonseca-etal-2019-findings,
    title = "Findings of the {WMT} 2019 Shared Tasks on Quality Estimation",
    author = "Fonseca, Erick  and
      Yankovskaya, Lisa  and
      Martins, Andr{\\'e} F. T.  and
      Fishel, Mark  and
      Federmann, Christian",
    editor = "Bojar, Ond{\\v{r}}ej  and
      Chatterjee, Rajen  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\\'e}  and
      Monz, Christof  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Post, Matt  and
      Turchi, Marco  and
      Verspoor, Karin",
    booktitle = "Proceedings of the Fourth Conference on Machine Translation (Volume 3: Shared Task Papers, Day 2)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5401/",
    doi = "10.18653/v1/W19-5401",
    pages = "1--10",
    abstract = "We report the results of the WMT19 shared task on Quality Estimation, i.e. the task of predicting the quality of the output of machine translation systems given just the source text and the hypothesis translations. The task includes estimation at three granularity levels: word, sentence and document. A novel addition is evaluating sentence-level QE against human judgments: in other words, designing MT metrics that do not need a reference translation. This year we include three language pairs, produced solely by neural machine translation systems. Participating teams from eleven institutions submitted a variety of systems to different task variants and language pairs."
}`
}
},
  {
  "id": 21,
  "title": "IT\u2013IST at the SIGMORPHON 2019 Shared Task: Sparse Two-headed Models for Inflection",
  "authors": "Ben Peters, Andr\u00e9 F. T. Martins",
  "venue": "SIGMORPHON",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>This paper presents the Instituto de Telecomunicações–Instituto Superior Técnico submission to Task 1 of the SIGMORPHON 2019 Shared Task. Our models combine sparse sequence-to-sequence models with a two-headed attention mechanism that learns separate attention distributions for the lemma and inflectional tags. Among submissions to Task 1, our models rank second and third. Despite the low data setting of the task (only 100 in-language training examples), they learn plausible inflection patterns and often concentrate all probability mass into a small set of hypotheses, making beam search exact.</p>`,
  "streams": [
  "attention",
  "multilingual-translation",
  "resources",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/W19-4207/",
  "bibtex": `@inproceedings{peters-martins-2019-ist,
    title = "{IT}{--}{IST} at the {SIGMORPHON} 2019 Shared Task: Sparse Two-headed Models for Inflection",
    author = "Peters, Ben  and
      Martins, Andr{\\'e} F. T.",
    editor = "Nicolai, Garrett  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 16th Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-4207/",
    doi = "10.18653/v1/W19-4207",
    pages = "50--56",
    abstract = "This paper presents the Instituto de Telecomunica{\\c{c}}{\\~o}es{--}Instituto Superior T{\\'e}cnico submission to Task 1 of the SIGMORPHON 2019 Shared Task. Our models combine sparse sequence-to-sequence models with a two-headed attention mechanism that learns separate attention distributions for the lemma and inflectional tags. Among submissions to Task 1, our models rank second and third. Despite the low data setting of the task (only 100 in-language training examples), they learn plausible inflection patterns and often concentrate all probability mass into a small set of hypotheses, making beam search exact."
}`
}
},
  {
  "id": 20,
  "title": "Translator2Vec: Understanding and Representing Human Post-Editors",
  "authors": "Ant\u00f3nio G\u00f3is, Andr\u00e9 F. T. Martins",
  "venue": "MTSummit",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>The combination of machines and humans for translation is effective, with many studies showing productivity gains when humans post-edit machine-translated output instead of translating from scratch. To take full advantage of this combination, we need a fine-grained understanding of how human translators work, and which post-editing styles are more effective than others. In this paper, we release and analyze a new dataset with document-level post-editing action sequences, including edit operations from keystrokes, mouse actions, and waiting times. Our dataset comprises 66,268 full document sessions post-edited by 332 humans, the largest of the kind released to date. We show that action sequences are informative enough to identify post-editors accurately, compared to baselines that only look at the initial and final text. We build on this to learn and visualize continuous representations of post-editors, and we show that these representations improve the downstream task of predicting post-editing time.</p>`,
  "streams": [
  "evaluation-metrics",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/W19-6605/",
  "bibtex": `@inproceedings{gois-martins-2019-translator2vec,
    title = "{T}ranslator2{V}ec: Understanding and Representing Human Post-Editors",
    author = "G{\\'o}is, Ant{\\'o}nio  and
      Martins, Andr{\\'e} F. T.",
    editor = "Forcada, Mikel  and
      Way, Andy  and
      Haddow, Barry  and
      Sennrich, Rico",
    booktitle = "Proceedings of Machine Translation Summit XVII: Research Track",
    month = aug,
    year = "2019",
    address = "Dublin, Ireland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/W19-6605/",
    pages = "43--54"
}`
}
},
  {
  "id": 19,
  "title": "Notes on Latent Structure Models and SPIGOT",
  "authors": "Andr\u00e9 F. T. Martins, Vlad Niculae",
  "venue": "arXiv",
  "year": 2019,
  "type": "preprint",
  "abstract": `<p>These notes aim to shed light on the recently proposed structured projected intermediate gradient optimization technique (SPIGOT, Peng et al., 2018). SPIGOT is a variant of the straight-through estimator (Bengio et al., 2013) which bypasses gradients of the argmax function by back-propagating a surrogate "gradient." We provide a new interpretation to the proposed gradient and put this technique into perspective, linking it to other methods for training neural networks with discrete latent variables. As a by-product, we suggest alternate variants of SPIGOT which will be further explored in future work.</p>`,
  "streams": [
  "resources",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/1907.10348",
  "bibtex": `@article{martins2019notes,
  title={Notes on Latent Structure Models and SPIGOT},
  author={Martins, Andr{\\'e} FT and Niculae, Vlad},
  journal={arXiv preprint arXiv:1907.10348},
  year={2019}
}`
}
},
  {
  "id": 18,
  "title": "Joint Learning of Named Entity Recognition and Entity Linking",
  "authors": "Pedro Henrique Martins, Zita Marinho, Andr\u00e9 F. T. Martins",
  "venue": "ACL SRW",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>Named entity recognition (NER) and entity linking (EL) are two fundamentally related tasks, since in order to perform EL, first the mentions to entities have to be detected. However, most entity linking approaches disregard the mention detection part, assuming that the correct mentions have been previously detected. In this paper, we perform joint learning of NER and EL to leverage their relatedness and obtain a more robust and generalisable system. For that, we introduce a model inspired by the Stack-LSTM approach (Dyer et al., 2015). We observe that, in fact, doing multi-task learning of NER and EL improves the performance in both tasks when comparing with models trained with individual objectives. Furthermore, we achieve results competitive with the state-of-the-art in both NER and EL.</p>`,
  "streams": [
  "interpretability",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/P19-2026/",
  "bibtex": `@inproceedings{martins-etal-2019-joint,
    title = "Joint Learning of Named Entity Recognition and Entity Linking",
    author = "Martins, Pedro Henrique  and
      Marinho, Zita  and
      Martins, Andr{\\'e} F. T.",
    editor = "Alva-Manchego, Fernando  and
      Choi, Eunsol  and
      Khashabi, Daniel",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-2026/",
    doi = "10.18653/v1/P19-2026",
    pages = "190--196",
    abstract = "Named entity recognition (NER) and entity linking (EL) are two fundamentally related tasks, since in order to perform EL, first the mentions to entities have to be detected. However, most entity linking approaches disregard the mention detection part, assuming that the correct mentions have been previously detected. In this paper, we perform joint learning of NER and EL to leverage their relatedness and obtain a more robust and generalisable system. For that, we introduce a model inspired by the Stack-LSTM approach. We observe that, in fact, doing multi-task learning of NER and EL improves the performance in both tasks when comparing with models trained with individual objectives. Furthermore, we achieve results competitive with the state-of-the-art in both NER and EL."
}`
}
},
  {
  "id": 17,
  "title": "Latent Structure Models for Natural Language Processing",
  "authors": "Andr\u00e9 F. T. Martins, Tsvetomila Mihaylova, Nikita Nangia, Vlad Niculae",
  "venue": "ACL",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>Latent structure models are a powerful tool for modeling compositional data, discovering linguistic structure, and building NLP pipelines. They are appealing for two main reasons: they allow incorporating structural bias during training, leading to more accurate models; and they allow discovering hidden linguistic structure, which provides better interpretability. This tutorial will cover recent advances in discrete latent structure models. We discuss their motivation, potential, and limitations, then explore in detail three strategies for designing such models: gradient approximation, reinforcement learning, and end-to-end differentiable methods. We highlight connections among all these methods, enumerating their strengths and weaknesses. The models we present and analyze have been applied to a wide variety of NLP tasks, including sentiment analysis, natural language inference, language modeling, machine translation, and semantic parsing. Examples and evaluation will be covered throughout. After attending the tutorial, a practitioner will be better informed about which method is best suited for their problem.</p>`,
  "streams": [
  "resources",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/P19-4001/",
  "code": "https://github.com/deep-spin/tutorial",
  "bibtex": `@inproceedings{martins-etal-2019-latent,
    title = "Latent Structure Models for Natural Language Processing",
    author = "Martins, Andr{\\'e} F. T.  and
      Mihaylova, Tsvetomila  and
      Nangia, Nikita  and
      Niculae, Vlad",
    editor = "Nakov, Preslav  and
      Palmer, Alexis",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-4001/",
    doi = "10.18653/v1/P19-4001",
    pages = "1--5",
    abstract = "Latent structure models are a powerful tool for modeling compositional data, discovering linguistic structure, and building NLP pipelines. They are appealing for two main reasons: they allow incorporating structural bias during training, leading to more accurate models; and they allow discovering hidden linguistic structure, which provides better interpretability. This tutorial will cover recent advances in discrete latent structure models. We discuss their motivation, potential, and limitations, then explore in detail three strategies for designing such models: gradient approximation, reinforcement learning, and end-to-end differentiable methods. We highlight connections among all these methods, enumerating their strengths and weaknesses. The models we present and analyze have been applied to a wide variety of NLP tasks, including sentiment analysis, natural language inference, language modeling, machine translation, and semantic parsing. Examples and evaluation will be covered throughout. After attending the tutorial, a practitioner will be better informed about which method is best suited for their problem."
}`
}
},
  {
  "id": 16,
  "title": "Scheduled Sampling for Transformers",
  "authors": "Tsvetomila Mihaylova, Andr\u00e9 F. T. Martins",
  "venue": "ACL",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>Scheduled sampling is a technique for avoiding one of the known problems in sequence-to-sequence generation: exposure bias. It consists of feeding the model a mix of the teacher forced embeddings and the model predictions from the previous step in training time. The technique has been used for improving model performance with recurrent neural networks (RNN). In the Transformer model, unlike the RNN, the generation of a new word attends to the full sentence generated so far, not only to the last word, and it is not straightforward to apply the scheduled sampling technique. We propose some structural changes to allow scheduled sampling to be applied to Transformer architectures, via a two-pass decoding strategy. Experiments on two language pairs achieve performance close to a teacher-forcing baseline and show that this technique is promising for further exploration.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/P19-2049/",
  "code": "https://github.com/deep-spin/scheduled-sampling-transformers",
  "bibtex": `@inproceedings{mihaylova-martins-2019-scheduled,
    title = "Scheduled Sampling for Transformers",
    author = "Mihaylova, Tsvetomila  and
      Martins, Andr{\\'e} F. T.",
    editor = "Alva-Manchego, Fernando  and
      Choi, Eunsol  and
      Khashabi, Daniel",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-2049/",
    doi = "10.18653/v1/P19-2049",
    pages = "351--356",
    abstract = "Scheduled sampling is a technique for avoiding one of the known problems in sequence-to-sequence generation: exposure bias. It consists of feeding the model a mix of the teacher forced embeddings and the model predictions from the previous step in training time. The technique has been used for improving model performance with recurrent neural networks (RNN). In the Transformer model, unlike the RNN, the generation of a new word attends to the full sentence generated so far, not only to the last word, and it is not straightforward to apply the scheduled sampling technique. We propose some structural changes to allow scheduled sampling to be applied to Transformer architectures, via a two-pass decoding strategy. Experiments on two language pairs achieve performance close to a teacher-forcing baseline and show that this technique is promising for further exploration."
}`
}
},
  {
  "id": 15,
  "title": "A Simple and Effective Approach to Automatic Post-Editing with Transfer Learning",
  "authors": "Gon\u00e7alo M. Correia, Andr\u00e9 F. T. Martins",
  "venue": "ACL",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>Automatic post-editing (APE) seeks to automatically refine the output of a black-box machine translation (MT) system through human post-edits. APE systems are usually trained by complementing human post-edited data with large, artificial data generated through back-translations, a time-consuming process often no easier than training a MT system from scratch. in this paper, we propose an alternative where we fine-tune pre-trained BERT models on both the encoder and decoder of an APE system, exploring several parameter sharing strategies. By only training on a dataset of 23K sentences for 3 hours on a single GPU we obtain results that are competitive with systems that were trained on 5M artificial sentences. When we add this artificial data our method obtains state-of-the-art results.</p>`,
  "streams": [
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/P19-1292/",
  "bibtex": `@inproceedings{correia-martins-2019-simple,
    title = "A Simple and Effective Approach to Automatic Post-Editing with Transfer Learning",
    author = "Correia, Gon{\\c{c}}alo M.  and
      Martins, Andr{\\'e} F. T.",
    editor = "Korhonen, Anna  and
      Traum, David  and
      M{\\\`a}rquez, Llu{\\'i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1292/",
    doi = "10.18653/v1/P19-1292",
    pages = "3050--3056",
    abstract = "Automatic post-editing (APE) seeks to automatically refine the output of a black-box machine translation (MT) system through human post-edits. APE systems are usually trained by complementing human post-edited data with large, artificial data generated through back-translations, a time-consuming process often no easier than training a MT system from scratch. in this paper, we propose an alternative where we fine-tune pre-trained BERT models on both the encoder and decoder of an APE system, exploring several parameter sharing strategies. By only training on a dataset of 23K sentences for 3 hours on a single GPU we obtain results that are competitive with systems that were trained on 5M artificial sentences. When we add this artificial data our method obtains state-of-the-art results."
}`
}
},
  {
  "id": 14,
  "title": "Unbabel\u2019s Submission to the WMT2019 APE Shared Task: BERT-Based Encoder-Decoder for Automatic Post-Editing",
  "authors": "Ant\u00f3nio V. Lopes, M. Amin Farajian, Gon\u00e7alo M. Correia, Jonay Tr\u00e9nous, Andr\u00e9 F. T. Martins",
  "venue": "WMT",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>This paper describes Unbabel’s submission to the WMT2019 APE Shared Task for the English-German language pair. Following the recent rise of large, powerful, pre-trained models, we adapt the BERT pretrained model to perform Automatic Post-Editing in an encoder-decoder framework. Analogously to dual-encoder architectures we develop a BERT-based encoder-decoder (BED) model in which a single pretrained BERT encoder receives both the source src and machine translation mt strings. Furthermore, we explore a conservativeness factor to constrain the APE system to perform fewer edits. As the official results show, when trained on a weighted combination of in-domain and artificial training data, our BED system with the conservativeness penalty improves significantly the translations of a strong NMT system by -0.78 and +1.23 in terms of TER and BLEU, respectively. Finally, our submission achieves a new state-of-the-art, ex-aequo, in English-German APE of NMT.</p>`,
  "streams": [
  "multilingual-translation",
  "shared-task"
],
  "links": {
  "paper": "https://aclanthology.org/W19-5413/",
  "bibtex": `@inproceedings{lopes-etal-2019-unbabels,
    title = "Unbabel{'}s Submission to the {WMT}2019 {APE} Shared Task: {BERT}-Based Encoder-Decoder for Automatic Post-Editing",
    author = "Lopes, Ant{\\'o}nio V.  and
      Farajian, M. Amin  and
      Correia, Gon{\\c{c}}alo M.  and
      Tr{\\'e}nous, Jonay  and
      Martins, Andr{\\'e} F. T.",
    editor = "Bojar, Ond{\\v{r}}ej  and
      Chatterjee, Rajen  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\\'e}  and
      Monz, Christof  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Post, Matt  and
      Turchi, Marco  and
      Verspoor, Karin",
    booktitle = "Proceedings of the Fourth Conference on Machine Translation (Volume 3: Shared Task Papers, Day 2)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5413/",
    doi = "10.18653/v1/W19-5413",
    pages = "118--123",
    abstract = "This paper describes Unbabel{'}s submission to the WMT2019 APE Shared Task for the English-German language pair. Following the recent rise of large, powerful, pre-trained models, we adapt the BERT pretrained model to perform Automatic Post-Editing in an encoder-decoder framework. Analogously to dual-encoder architectures we develop a BERT-based encoder-decoder (BED) model in which a single pretrained BERT encoder receives both the source src and machine translation mt strings. Furthermore, we explore a conservativeness factor to constrain the APE system to perform fewer edits. As the official results show, when trained on a weighted combination of in-domain and artificial training data, our BED system with the conservativeness penalty improves significantly the translations of a strong NMT system by -0.78 and +1.23 in terms of TER and BLEU, respectively. Finally, our submission achieves a new state-of-the-art, ex-aequo, in English-German APE of NMT."
}`
}
},
  {
  "id": 13,
  "title": "Sparse Sequence-to-Sequence Models",
  "authors": "Ben Peters, Vlad Niculae, Andr\u00e9 F. T. Martins",
  "venue": "ACL",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>Sequence-to-sequence models are a powerful workhorse of NLP. Most variants employ a softmax transformation in both their attention mechanism and output layer, leading to dense alignments and strictly positive output probabilities. This density is wasteful, making models less interpretable and assigning probability mass to many implausible outputs. In this paper, we propose sparse sequence-to-sequence models, rooted in a new family of 𝛼-entmax transformations, which includes softmax and sparsemax as particular cases, and is sparse for any 𝛼 &gt; 1. We provide fast algorithms to evaluate these transformations and their gradients, which scale well for large vocabulary sizes. Our models are able to produce sparse alignments and to assign nonzero probability to a short list of plausible outputs, sometimes rendering beam search exact. Experiments on morphological inflection and machine translation reveal consistent gains over dense models.</p>`,
  "streams": [
  "attention",
  "multilingual-translation",
  "resources",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/P19-1146/",
  "code": "https://github.com/deep-spin/entmax",
  "bibtex": `@inproceedings{peters-etal-2019-sparse,
    title = "Sparse Sequence-to-Sequence Models",
    author = "Peters, Ben  and
      Niculae, Vlad  and
      Martins, Andr{\\'e} F. T.",
    editor = "Korhonen, Anna  and
      Traum, David  and
      M{\\\`a}rquez, Llu{\\'i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1146/",
    doi = "10.18653/v1/P19-1146",
    pages = "1504--1519",
    abstract = "Sequence-to-sequence models are a powerful workhorse of NLP. Most variants employ a softmax transformation in both their attention mechanism and output layer, leading to dense alignments and strictly positive output probabilities. This density is wasteful, making models less interpretable and assigning probability mass to many implausible outputs. In this paper, we propose sparse sequence-to-sequence models, rooted in a new family of $\\alpha$-entmax transformations, which includes softmax and sparsemax as particular cases, and is sparse for any $\\alpha > 1$. We provide fast algorithms to evaluate these transformations and their gradients, which scale well for large vocabulary sizes. Our models are able to produce sparse alignments and to assign nonzero probability to a short list of plausible outputs, sometimes rendering beam search exact. Experiments on morphological inflection and machine translation reveal consistent gains over dense models."
}`
}
},
  {
  "id": 12,
  "title": "Learning Classifiers with Fenchel-Young Losses: Generalized Entropies, Margins, and Algorithms",
  "authors": "Mathieu Blondel, Andr\u00e9 F. T. Martins, Vlad Niculae",
  "venue": "AISTATS",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>This paper studies Fenchel-Young losses, a generic way to construct convex loss functions from a regularization function. We analyze their properties in depth, showing that they unify many well-known loss functions and allow to create useful new ones easily. Fenchel-Young losses constructed from a generalized entropy, including the Shannon and Tsallis entropies, induce predictive probability distributions. We formulate conditions for a generalized entropy to yield losses with a separation margin, and probability distributions with sparse support. Finally, we derive efficient algorithms, making Fenchel-Young losses appealing both in theory and practice.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "http://proceedings.mlr.press/v89/blondel19a/blondel19a.pdf",
  "bibtex": `@InProceedings{pmlr-v89-blondel19a,
  title = 	 {Learning Classifiers with Fenchel-Young Losses: Generalized Entropies, Margins, and Algorithms},
  author =       {Blondel, Mathieu and Martins, Andre and Niculae, Vlad},
  booktitle = 	 {Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics},
  pages = 	 {606--615},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Sugiyama, Masashi},
  volume = 	 {89},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {16--18 Apr},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v89/blondel19a/blondel19a.pdf},
  url = 	 {https://proceedings.mlr.press/v89/blondel19a.html},
  abstract = 	 {This paper studies Fenchel-Young losses, a generic way to construct convex loss functions from a regularization function.  We analyze their properties in depth, showing that they unify many well-known loss functions and allow to create useful new ones easily.  Fenchel-Young losses constructed from a generalized entropy, including the Shannon and Tsallis entropies, induce predictive probability distributions.  We formulate conditions for a generalized entropy to yield losses with a separation margin, and probability distributions with sparse support.  Finally, we derive efficient algorithms, making Fenchel-Young losses appealing both in theory and practice.}
}`
}
},
  {
  "id": 11,
  "title": "Jointly Extracting and Compressing Documents with Summary State Representations",
  "authors": "Afonso Mendes, Shashi Narayan, Sebasti\u00e3o Miranda, Zita Marinho, Andr\u00e9 F. T. Martins, Shay B. Cohen",
  "venue": "NAACL",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>We present a new neural model for text summarization that first extracts sentences from a document and then compresses them. The pro-posed model offers a balance that sidesteps thedifficulties in abstractive methods while gener-ating more concise summaries than extractivemethods. In addition, our model dynamically determines the length of the output summary based on the gold summaries it observes during training and does not require length constraints typical to extractive summarization. The model achieves state-of-the-art results on the CNN/DailyMail and Newsroom datasets, improving over current extractive and abstractive methods. Human evaluations demonstratethat our model generates concise and informa-tive summaries. We also make available a new dataset of oracle compressive summaries derived automatically from the CNN/DailyMailreference summaries.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/N19-1397/",
  "bibtex": `@inproceedings{mendes-etal-2019-jointly,
    title = "Jointly Extracting and Compressing Documents with Summary State Representations",
    author = "Mendes, Afonso  and
      Narayan, Shashi  and
      Miranda, Sebasti{\\~a}o  and
      Marinho, Zita  and
      Martins, Andr{\\'e} F. T.  and
      Cohen, Shay B.",
    editor = "Burstein, Jill  and
      Doran, Christy  and
      Solorio, Thamar",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1397/",
    doi = "10.18653/v1/N19-1397",
    pages = "3955--3966",
    abstract = "We present a new neural model for text summarization that first extracts sentences from a document and then compresses them. The pro-posed model offers a balance that sidesteps thedifficulties in abstractive methods while gener-ating more concise summaries than extractivemethods. In addition, our model dynamically determines the length of the output summary based on the gold summaries it observes during training and does not require length constraints typical to extractive summarization. The model achieves state-of-the-art results on the CNN/DailyMail and Newsroom datasets, improving over current extractive and abstractive methods. Human evaluations demonstratethat our model generates concise and informa-tive summaries. We also make available a new dataset of oracle compressive summaries derived automatically from the CNN/DailyMailreference summaries."
}`
}
},
  {
  "id": 10,
  "title": "Selective Attention for Context-aware Neural Machine Translation",
  "authors": "Sameen Maruf, Andr\u00e9 F. T. Martins, Gholamreza Haffari",
  "venue": "NAACL",
  "year": 2019,
  "type": "conference",
  "abstract": `<p>Despite the progress made in sentence-level NMT, current systems still fall short at achieving fluent, good quality translation for a full document. Recent works in context-aware NMT consider only a few previous sentences as context and may not scale to entire documents. To this end, we propose a novel and scalable top-down approach to hierarchical attention for context-aware NMT which uses sparse attention to selectively focus on relevant sentences in the document context and then attends to key words in those sentences. We also propose single-level attention approaches based on sentence or word-level information in the context. The document-level context representation, produced from these attention modules, is integrated into the encoder or decoder of the Transformer model depending on whether we use monolingual or bilingual context. Our experiments and evaluation on English-German datasets in different document MT settings show that our selective attention approach not only significantly outperforms context-agnostic baselines but also surpasses context-aware baselines in most cases.</p>`,
  "streams": [
  "attention",
  "dialogue-context",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/N19-1313/",
  "bibtex": `@inproceedings{maruf-etal-2019-selective,
    title = "Selective Attention for Context-aware Neural Machine Translation",
    author = "Maruf, Sameen  and
      Martins, Andr{\\'e} F. T.  and
      Haffari, Gholamreza",
    editor = "Burstein, Jill  and
      Doran, Christy  and
      Solorio, Thamar",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1313/",
    doi = "10.18653/v1/N19-1313",
    pages = "3092--3102",
    abstract = "Despite the progress made in sentence-level NMT, current systems still fall short at achieving fluent, good quality translation for a full document. Recent works in context-aware NMT consider only a few previous sentences as context and may not scale to entire documents. To this end, we propose a novel and scalable top-down approach to hierarchical attention for context-aware NMT which uses sparse attention to selectively focus on relevant sentences in the document context and then attends to key words in those sentences. We also propose single-level attention approaches based on sentence or word-level information in the context. The document-level context representation, produced from these attention modules, is integrated into the encoder or decoder of the Transformer model depending on whether we use monolingual or bilingual context. Our experiments and evaluation on English-German datasets in different document MT settings show that our selective attention approach not only significantly outperforms context-agnostic baselines but also surpasses context-aware baselines in most cases."
}`
}
},
  {
  "id": 9,
  "title": "Interpretable Structure Induction via Sparse Attention",
  "authors": "Ben Peters, Vlad Niculae, Andr\u00e9 F. T. Martins",
  "venue": "BlackBoxNLP",
  "year": 2018,
  "type": "conference",
  "abstract": `<p>Neural network methods are experiencing wide adoption in NLP, thanks to their empirical performance on many tasks. Modern neural architectures go way beyond simple feedforward and recurrent models: they are complex pipelines that perform soft, differentiable computation instead of discrete logic. The price of such soft computing is the introduction of dense dependencies, which make it hard to disentangle the patterns that trigger a prediction. Our recent work on sparse and structured latent computation presents a promising avenue for enhancing interpretability of such neural pipelines. Through this extended abstract, we aim to discuss and explore the potential and impact of our methods.</p>`,
  "streams": [
  "attention",
  "interpretability"
],
  "links": {
  "paper": "https://aclanthology.org/W18-5450/",
  "bibtex": `@inproceedings{peters-etal-2018-interpretable,
    title = "Interpretable Structure Induction via Sparse Attention",
    author = "Peters, Ben  and
      Niculae, Vlad  and
      Martins, Andr{\\'e} F. T.",
    editor = "Linzen, Tal  and
      Chrupa{\\l}a, Grzegorz  and
      Alishahi, Afra",
    booktitle = "Proceedings of the 2018 {EMNLP} Workshop {B}lackbox{NLP}: Analyzing and Interpreting Neural Networks for {NLP}",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-5450/",
    doi = "10.18653/v1/W18-5450",
    pages = "365--367",
    abstract = "Neural network methods are experiencing wide adoption in NLP, thanks to their empirical performance on many tasks. Modern neural architectures go way beyond simple feedforward and recurrent models: they are complex pipelines that perform soft, differentiable computation instead of discrete logic. The price of such soft computing is the introduction of dense dependencies, which make it hard to disentangle the patterns that trigger a prediction. Our recent work on sparse and structured latent computation presents a promising avenue for enhancing interpretability of such neural pipelines. Through this extended abstract, we aim to discuss and explore the potential and impact of our methods."
}`
}
},
  {
  "id": 8,
  "title": "Towards Dynamic Computation Graphs via Sparse Latent Structure",
  "authors": "Vlad Niculae, Andr\u00e9 F. T. Martins, Claire Cardie",
  "venue": "EMNLP",
  "year": 2018,
  "type": "conference",
  "abstract": `<p>Deep NLP models benefit from underlying structures in the data---e.g., parse trees---typically extracted using off-the-shelf parsers. Recent attempts to jointly learn the latent structure encounter a tradeoff: either make factorization assumptions that limit expressiveness, or sacrifice end-to-end differentiability. Using the recently proposed SparseMAP inference, which retrieves a sparse distribution over latent structures, we propose a novel approach for end-to-end learning of latent structure predictors jointly with a downstream predictor. To the best of our knowledge, our method is the first to enable unrestricted dynamic computation graph construction from the global latent structure, while maintaining differentiability.</p>`,
  "streams": [
  "interpretability",
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/D18-1108/",
  "bibtex": `@inproceedings{niculae-etal-2018-towards,
    title = "Towards Dynamic Computation Graphs via Sparse Latent Structure",
    author = "Niculae, Vlad  and
      Martins, Andr{\\'e} F. T.  and
      Cardie, Claire",
    editor = "Riloff, Ellen  and
      Chiang, David  and
      Hockenmaier, Julia  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1108/",
    doi = "10.18653/v1/D18-1108",
    pages = "905--911",
    abstract = "Deep NLP models benefit from underlying structures in the data{---}e.g., parse trees{---}typically extracted using off-the-shelf parsers. Recent attempts to jointly learn the latent structure encounter a tradeoff: either make factorization assumptions that limit expressiveness, or sacrifice end-to-end differentiability. Using the recently proposed SparseMAP inference, which retrieves a sparse distribution over latent structures, we propose a novel approach for end-to-end learning of latent structure predictors jointly with a downstream predictor. To the best of our knowledge, our method is the first to enable unrestricted dynamic computation graph construction from the global latent structure, while maintaining differentiability."
}`
}
},
  {
  "id": 7,
  "title": "Contextual Neural Model for Translating Bilingual Multi-Speaker Conversations",
  "authors": "Sameen Maruf, Andr\u00e9 F. T. Martins, Gholamreza Haffari",
  "venue": "WMT",
  "year": 2018,
  "type": "conference",
  "abstract": `<p>Recent works in neural machine translation have begun to explore document translation. However, translating online multi-speaker conversations is still an open problem. In this work, we propose the task of translating Bilingual Multi-Speaker Conversations, and explore neural architectures which exploit both source and target-side conversation histories for this task. To initiate an evaluation for this task, we introduce datasets extracted from Europarl v7 and OpenSubtitles2016. Our experiments on four language-pairs confirm the significance of leveraging conversation history, both in terms of BLEU and manual evaluation.</p>`,
  "streams": [
  "dialogue-context",
  "multilingual-translation",
  "resources"
],
  "links": {
  "paper": "https://aclanthology.org/W18-6311/",
  "bibtex": `@inproceedings{maruf-etal-2018-contextual,
    title = "Contextual Neural Model for Translating Bilingual Multi-Speaker Conversations",
    author = "Maruf, Sameen  and
      Martins, Andr{\\'e} F. T.  and
      Haffari, Gholamreza",
    editor = "Bojar, Ond{\\v{r}}ej  and
      Chatterjee, Rajen  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Monz, Christof  and
      Negri, Matteo  and
      N{\\'e}v{\\'e}ol, Aur{\\'e}lie  and
      Neves, Mariana  and
      Post, Matt  and
      Specia, Lucia  and
      Turchi, Marco  and
      Verspoor, Karin",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-6311/",
    doi = "10.18653/v1/W18-6311",
    pages = "101--112",
    abstract = "Recent works in neural machine translation have begun to explore document translation. However, translating online multi-speaker conversations is still an open problem. In this work, we propose the task of translating Bilingual Multi-Speaker Conversations, and explore neural architectures which exploit both source and target-side conversation histories for this task. To initiate an evaluation for this task, we introduce datasets extracted from Europarl v7 and OpenSubtitles2016. Our experiments on four language-pairs confirm the significance of leveraging conversation history, both in terms of BLEU and manual evaluation."
}`
}
},
  {
  "id": 6,
  "title": "SparseMAP: Differentiable Sparse Structured Inference",
  "authors": "Vlad Niculae, Andr\u00e9 F. T. Martins, Mathieu Blondel, Claire Cardie",
  "venue": "ICML",
  "year": 2018,
  "type": "conference",
  "abstract": `<p>Structured prediction requires searching over a combinatorial number of structures. To tackle it, we introduce SparseMAP: a new method for sparse structured inference, and its natural loss function. SparseMAP automatically selects only a few global structures: it is situated between MAP inference, which picks a single structure, and marginal inference, which assigns probability mass to all structures, including implausible ones. Importantly, SparseMAP can be computed using only calls to a MAP oracle, making it applicable to problems with intractable marginal inference, e.g., linear assignment. Sparsity makes gradient backpropagation efficient regardless of the structure, enabling us to augment deep neural networks with generic and sparse structured hidden layers. Experiments in dependency parsing and natural language inference reveal competitive accuracy, improved interpretability, and the ability to capture natural language ambiguities, which is attractive for pipeline systems.</p>`,
  "streams": [
  "interpretability",
  "theory"
],
  "links": {
  "paper": "http://proceedings.mlr.press/v80/niculae18a/niculae18a.pdf",
  "bibtex": `@InProceedings{pmlr-v80-niculae18a,
  title = 	 {{S}parse{MAP}: Differentiable Sparse Structured Inference},
  author =       {Niculae, Vlad and Martins, Andre and Blondel, Mathieu and Cardie, Claire},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {3799--3808},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--15 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v80/niculae18a/niculae18a.pdf},
  url = 	 {https://proceedings.mlr.press/v80/niculae18a.html},
  abstract = 	 {Structured prediction requires searching over a combinatorial number of structures. To tackle it, we introduce SparseMAP, a new method for sparse structured inference, together with corresponding loss functions. SparseMAP inference is able to automatically select only a few global structures: it is situated between MAP inference, which picks a single structure, and marginal inference, which assigns probability mass to all structures, including implausible ones. Importantly, SparseMAP can be computed using only calls to a MAP oracle, hence it is applicable even to problems where marginal inference is intractable, such as linear assignment. Moreover, thanks to the solution sparsity, gradient backpropagation is efficient regardless of the structure. SparseMAP thus enables us to augment deep neural networks with generic and sparse structured hidden layers. Experiments in dependency parsing and natural language inference reveal competitive accuracy, improved interpretability, and the ability to capture natural language ambiguities, which is attractive for pipeline systems.}
}`
}
},
  {
  "id": 5,
  "title": "Sparse and Constrained Attention for Neural Machine Translation",
  "authors": "Chaitanya Malaviya, Pedro Ferreira, Andr\u00e9 F. T. Martins",
  "venue": "ACL",
  "year": 2018,
  "type": "conference",
  "abstract": `<p>In NMT, words are sometimes dropped from the source or generated repeatedly in the translation. We explore novel strategies to address the coverage problem that change only the attention transformation. Our approach allocates fertilities to source words, used to bound the attention each word can receive. We experiment with various sparse and constrained attention transformations and propose a new one, constrained sparsemax, shown to be differentiable and sparse. Empirical evaluation is provided in three languages pairs.</p>`,
  "streams": [
  "attention",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/P18-2059/",
  "bibtex": `@inproceedings{malaviya-etal-2018-sparse,
    title = "Sparse and Constrained Attention for Neural Machine Translation",
    author = "Malaviya, Chaitanya  and
      Ferreira, Pedro  and
      Martins, Andr{\\'e} F. T.",
    editor = "Gurevych, Iryna  and
      Miyao, Yusuke",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-2059/",
    doi = "10.18653/v1/P18-2059",
    pages = "370--376",
    abstract = "In neural machine translation, words are sometimes dropped from the source or generated repeatedly in the translation. We explore novel strategies to address the coverage problem that change only the attention transformation. Our approach allocates fertilities to source words, used to bound the attention each word can receive. We experiment with various sparse and constrained attention transformations and propose a new one, constrained sparsemax, shown to be differentiable and sparse. Empirical evaluation is provided in three languages pairs."
}`
}
},
  {
  "id": 4,
  "title": "Marian: Fast Neural Machine Translation in C++",
  "authors": "Marcin Junczys-Dowmunt, Roman Grundkiewicz, Tomasz Dwojak, Hieu Hoang, Kenneth Heafield, Tom Neckermann, Frank Seide, Ulrich Germann, Alham Fikri Aji, Nikolay Bogoychev, Andr\u00e9 F. T. Martins, Alexandra Birch",
  "venue": "ACL",
  "year": 2018,
  "type": "conference",
  "abstract": `<p>We present Marian, an efficient and self-contained Neural Machine Translation framework with an integrated automatic differentiation engine based on dynamic computation graphs. Marian is written entirely in C++. We describe the design of the encoder-decoder framework and demonstrate that a research-friendly toolkit can achieve high training and translation speed.</p>`,
  "streams": [
  "efficiency",
  "multilingual-translation"
],
  "links": {
  "paper": "https://aclanthology.org/P18-4020/",
  "bibtex": `@inproceedings{junczys-dowmunt-etal-2018-marian,
    title = "{M}arian: Fast Neural Machine Translation in {C}++",
    author = "Junczys-Dowmunt, Marcin  and
      Grundkiewicz, Roman  and
      Dwojak, Tomasz  and
      Hoang, Hieu  and
      Heafield, Kenneth  and
      Neckermann, Tom  and
      Seide, Frank  and
      Germann, Ulrich  and
      Aji, Alham Fikri  and
      Bogoychev, Nikolay  and
      Martins, Andr{\\'e} F. T.  and
      Birch, Alexandra",
    editor = "Liu, Fei  and
      Solorio, Thamar",
    booktitle = "Proceedings of {ACL} 2018, System Demonstrations",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-4020/",
    doi = "10.18653/v1/P18-4020",
    pages = "116--121",
    abstract = "We present Marian, an efficient and self-contained Neural Machine Translation framework with an integrated automatic differentiation engine based on dynamic computation graphs. Marian is written entirely in C++. We describe the design of the encoder-decoder framework and demonstrate that a research-friendly toolkit can achieve high training and translation speed."
}`
}
},
  {
  "id": 3,
  "title": "Learning What\u2019s Easy: Fully Differentiable Neural Easy-First Taggers",
  "authors": "Andr\u00e9 F. T. Martins, Julia Kreutzer",
  "venue": "EMNLP",
  "year": 2017,
  "type": "conference",
  "abstract": `<p>We introduce a novel neural easy-first decoder that learns to solve sequence tagging tasks in a flexible order. In contrast to previous easy-first decoders, our models are end-to-end differentiable. The decoder iteratively updates a “sketch” of the predictions over the sequence. At its core is an attention mechanism that controls which parts of the input are strategically the best to process next. We present a new constrained softmax transformation that ensures the same cumulative attention to every word, and show how to efficiently evaluate and backpropagate over it. Our models compare favourably to BILSTM taggers on three sequence tagging tasks.</p>`,
  "streams": [
  "theory"
],
  "links": {
  "paper": "https://aclanthology.org/D17-1036/",
  "bibtex": `@inproceedings{martins-kreutzer-2017-learning,
    title = "Learning What{'}s Easy: Fully Differentiable Neural Easy-First Taggers",
    author = "Martins, Andr{\\'e} F. T.  and
      Kreutzer, Julia",
    editor = "Palmer, Martha  and
      Hwa, Rebecca  and
      Riedel, Sebastian",
    booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D17-1036/",
    doi = "10.18653/v1/D17-1036",
    pages = "349--362",
    abstract = "We introduce a novel neural easy-first decoder that learns to solve sequence tagging tasks in a flexible order. In contrast to previous easy-first decoders, our models are end-to-end differentiable. The decoder iteratively updates a \`\`sketch'' of the predictions over the sequence. At its core is an attention mechanism that controls which parts of the input are strategically the best to process next. We present a new constrained softmax transformation that ensures the same cumulative attention to every word, and show how to efficiently evaluate and backpropagate over it. Our models compare favourably to BILSTM taggers on three sequence tagging tasks."
}`
}
},
  {
  "id": 2,
  "title": "Pushing the Limits of Translation Quality Estimation",
  "authors": "Andr\u00e9 F. T. Martins, Marcin Junczys-Dowmunt, Fabio N. Kepler, Ram\u00f3n Astudillo, Chris Hokamp, Roman Grundkiewicz",
  "venue": "TACL",
  "year": 2017,
  "type": "journal",
  "abstract": `<p>Translation quality estimation is a task of growing importance in NLP, due to its potential to reduce post-editing human effort in disruptive ways. However, this potential is currently limited by the relatively low accuracy of existing systems. In this paper, we achieve remarkable improvements by exploiting synergies between the related tasks of word-level quality estimation and automatic post-editing. First, we stack a new, carefully engineered, neural model into a rich feature-based word-level quality estimation system. Then, we use the output of an automatic post-editing system as an extra feature, obtaining striking results on WMT16: a word-level F MULT 1 score of 57.47% (an absolute gain of +7.95% over the current state of the art), and a Pearson correlation score of 65.56% for sentence-level HTER prediction (an absolute gain of +13.36%).</p>`,
  "streams": [
  "evaluation-metrics"
],
  "links": {
  "paper": "https://aclanthology.org/Q17-1015/",
  "bibtex": `@article{martins-etal-2017-pushing,
    title = "Pushing the Limits of Translation Quality Estimation",
    author = "Martins, Andr{\\'e} F. T.  and
      Junczys-Dowmunt, Marcin  and
      Kepler, Fabio N.  and
      Astudillo, Ram{\\'o}n  and
      Hokamp, Chris  and
      Grundkiewicz, Roman",
    editor = "Lee, Lillian  and
      Johnson, Mark  and
      Toutanova, Kristina",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "5",
    year = "2017",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q17-1015/",
    doi = "10.1162/tacl_a_00056",
    pages = "205--218",
    abstract = "Translation quality estimation is a task of growing importance in NLP, due to its potential to reduce post-editing human effort in disruptive ways. However, this potential is currently limited by the relatively low accuracy of existing systems. In this paper, we achieve remarkable improvements by exploiting synergies between the related tasks of word-level quality estimation and automatic post-editing. First, we stack a new, carefully engineered, neural model into a rich feature-based word-level quality estimation system. Then, we use the output of an automatic post-editing system as an extra feature, obtaining striking results on WMT16: a word-level FMULT1 score of 57.47{\\%} (an absolute gain of +7.95{\\%} over the current state of the art), and a Pearson correlation score of 65.56{\\%} for sentence-level HTER prediction (an absolute gain of +13.36{\\%})."
}`
}
},
  {
  "id": 1,
  "title": "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification",
  "authors": "Andr\u00e9 F. T. Martins, Ram\u00f3n Fern\u00e1ndez Astudillo",
  "venue": "ICML",
  "year": 2016,
  "type": "conference",
  "abstract": `<p>We propose sparsemax, a new activation function similar to the traditional softmax, but able to output sparse probabilities. After deriving its properties, we show how its Jacobian can be efficiently computed, enabling its use in a network trained with backpropagation. Then, we propose a new smooth and convex loss function which is the sparsemax analogue of the logistic loss. We reveal an unexpected connection between this new loss and the Huber classification loss. We obtain promising empirical results in multi-label classification problems and in attention-based neural networks for natural language inference. For the latter, we achieve a similar performance as the traditional softmax, but with a selective, more compact, attention focus.</p>`,
  "streams": [
  "attention",
  "resources",
  "theory"
],
  "links": {
  "paper": "https://arxiv.org/abs/1602.02068",
  "bibtex": `@InProceedings{pmlr-v48-martins16,
  title = 	 {From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification},
  author = 	 {Martins, Andre and Astudillo, Ramon},
  booktitle = 	 {Proceedings of The 33rd International Conference on Machine Learning},
  pages = 	 {1614--1623},
  year = 	 {2016},
  editor = 	 {Balcan, Maria Florina and Weinberger, Kilian Q.},
  volume = 	 {48},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {New York, New York, USA},
  month = 	 {20--22 Jun},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v48/martins16.pdf},
  url = 	 {https://proceedings.mlr.press/v48/martins16.html},
  abstract = 	 {We propose sparsemax, a new activation function similar to the traditional softmax, but able to output sparse probabilities. After deriving its properties, we show how its Jacobian can be efficiently computed, enabling its use in a network trained with backpropagation. Then, we propose a new smooth and convex loss function which is the sparsemax analogue of the logistic loss. We reveal an unexpected connection between this new loss and the Huber classification loss. We obtain promising empirical results in multi-label classification problems and in attention-based neural networks for natural language inference. For the latter, we achieve a similar performance as the traditional softmax, but with a selective, more compact, attention focus.}
}`
}
}
];

// Function to get recent publications (used by index.html)
function getRecentPublications(limit = 3) {
    return publicationsData.slice(0, limit);
}

// Function to get all publications (used by publications.html)
function getAllPublications() {
    return publicationsData;
}

// Function to filter publications by type
function filterPublicationsByType(type) {
    if (type === 'all') return publicationsData;
    return publicationsData.filter(pub => pub.type === type);
}

// Function to filter publications by year
function filterPublicationsByYear(year) {
    if (year === 'all') return publicationsData;
    return publicationsData.filter(pub => pub.year.toString() === year);
}

// Function to filter publications by streams
function filterPublicationsByStreams(selectedStreams) {
    if (!selectedStreams || selectedStreams.length === 0) return publicationsData;
    return publicationsData.filter(pub => 
        selectedStreams.some(stream => pub.streams.includes(stream))
    );
}

// Function to search publications
function searchPublications(query) {
    const searchTerm = query.toLowerCase();
    return publicationsData.filter(pub => 
        pub.title.toLowerCase().includes(searchTerm) ||
        pub.authors.toLowerCase().includes(searchTerm) ||
        pub.abstract.toLowerCase().includes(searchTerm) ||
        pub.streams.some(stream => stream.toLowerCase().includes(searchTerm))
    );
}

// Function to get unique years for filtering
function getPublicationYears() {
    const years = [...new Set(publicationsData.map(pub => pub.year))];
    return years.sort((a, b) => b - a);
}

// Function to get unique types for filtering
function getPublicationTypes() {
    return [...new Set(publicationsData.map(pub => pub.type))];
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        publicationsData,
        getRecentPublications,
        getAllPublications,
        filterPublicationsByType,
        filterPublicationsByYear,
        filterPublicationsByStreams,
        searchPublications,
        getPublicationYears,
        getPublicationTypes
    };
}