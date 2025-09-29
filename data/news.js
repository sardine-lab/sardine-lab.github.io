// News data for SARDINE Lab website
// Add new news items to the top of the array to maintain chronological order

const newsData = [
  {
  "date": "23/09/2025",
  "type": "team",
  "title": "Andr\u00e9 is looking for post-docs",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> is looking for highly motivated post-docs to join our SARDINE Lab in Lisbon to work on the <a href="https://andre-martins.github.io/pages/decollage.html">DECOLLAGE project</a> (ERC Consolidator). Please reach out by DM or email! </p>`,
  "tags": [
  "team",
  "postdocs",
  "job"
]
},
  {
  "date": "30/07/2025",
  "type": "award",
  "title": "Outstanding Paper Award at ACL 2025",
  "content": `<p><img alt="Hugo receiving the ACL award" src="assets/figs/hugo-award.jpeg" />
Congratulations to <strong>Hugo Pitorro</strong> and <strong>Marcos Treviso</strong> for receiving an <strong>Outstanding Paper Award</strong> at <strong>ACL 2025</strong> for <a href="https://arxiv.org/abs/2502.15612">LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models</a>! 🎉  </p>`,
  "tags": [
  "acl2025",
  "award",
  "outstanding-paper",
  "research"
]
},
  {
  "date": "25/07/2025",
  "type": "publication",
  "title": "Book: Discrete Latent Structure in Neural Networks",
  "content": `<p>We’re excited to announce the release of the book <strong>“Discrete Latent Structure in Neural Networks”</strong> by <a href="https://andre-martins.github.io/">André Martins</a> and former SARDINEs <strong>Vlad</strong>, <strong>Nikita</strong>, and <strong>Tsvety</strong>. Read it on arXiv: <a href="https://arxiv.org/abs/2301.07473">https://arxiv.org/abs/2301.07473</a>.</p>`,
  "tags": [
  "book",
  "publication",
  "discrete-latent-structure",
  "neural-networks"
]
},
  {
  "date": "23/07/2025",
  "type": "team",
  "title": "Ant\u00f3nio Farinhas passes his PhD with distinction and honor!",
  "content": `<p><img alt="António PhD Defense" src="assets/figs/antonio-defense.jpg" />
Big applause to <strong>António Farinhas</strong> for successfully defending his PhD <strong>with distinction and honor</strong>! 🥳 </p>`,
  "tags": [
  "team",
  "phd",
  "defense",
  "distinction",
  "celebration"
]
},
  {
  "date": "16/06/2025",
  "type": "release",
  "title": "Our Triton Tutorial is live!",
  "content": `<p><img alt="Triton tutorial cartoon" src="assets/figs/triton-tutorial.jpeg" />
<strong>Marcos Treviso</strong> and <strong>Nuno Gonçalves</strong> released our in-house <strong>Triton Tutorial</strong>, with a series of notebooks ranging from simple vector addition to sparsemax attention! Check it out: <a href="https://github.com/deep-spin/triton-tutorial">https://github.com/deep-spin/triton-tutorial</a>.</p>`,
  "tags": [
  "triton",
  "tutorial",
  "gpu",
  "kernels",
  "education",
  "release"
]
},
  {
  "date": "15/12/2024",
  "type": "presentation",
  "title": "NeurIPS 2024 Tutorial on Dynamic Sparsity",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> presented a tutorial at NeurIPS 2024 with <a href="https://ducdauge.github.io/">Edoardo Ponti</a> on <a href="https://dynamic-sparsity.github.io/">Dynamic Sparsity in Machine Learning: Routing Information through Neural Pathways</a>. Check it out! We have lots of materials, including slides and Jupyter notebooks.</p>`,
  "tags": [
  "neurips",
  "tutorial",
  "dynamic-sparsity",
  "materials"
]
},
  {
  "date": "15/11/2024",
  "type": "presentation",
  "title": "EuroHPC Summit Presentation on Multilingual LLMs",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> presented “<a href="./docs/EuroHPCSummit2025.pdf">Open &amp; Multilingual LLMs for Europe</a>” in a discussion panel at <a href="https://www.eurohpcsummit.eu/">EuroHPC Summit</a> in Krakow about the AI Factories. There, he covered some of the team’s recent successes with CroissantLLM, TowerLLM, EuroLLM, and EuroBERT.</p>`,
  "tags": [
  "eurohpc",
  "multilingual",
  "llm",
  "presentation"
]
},
  {
  "date": "01/11/2024",
  "type": "release",
  "title": "EuroLLM Website Launch and Success Story",
  "content": `<p>New <a href="https://eurollm.io">EuroLLM website</a>! We trained two LLMs from scratch, <a href="https://huggingface.co/utter-project/EuroLLM-1.7B">EuroLLM-1.7B</a> and <a href="https://huggingface.co/utter-project/EuroLLM-9B">EuroLLM-9B</a>, using the European supercomputing infrastructure (EuroHPC). These models support 35 languages (including all 24 EU official languages). They were released fully open and are <a href="https://huggingface.co/blog/eurollm-team/eurollm-9b">among the best in several benchmarks</a>. They have 300k+ downloads so far! This was done in collaboration with Instituto Superior Técnico, Instituto de Telecomunicações, Unbabel, The University of Edinburgh, CentraleSupélec among others and was recently featured as <a href="https://eurohpc-ju.europa.eu/eurohpc-success-story-speaking-freely-eurollm_en">a success story at EuroHPC</a>. Larger and more powerful models are on the making now!</p>`,
  "tags": [
  "eurollm",
  "multilingual",
  "open-source",
  "eurohpc"
]
},
  {
  "date": "20/10/2024",
  "type": "event",
  "title": "Invited Talks at Cornell Tech, MIT, and EPFL",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> gave recent talks at Cornell Tech “<a href="https://events.cornell.edu/event/lmss-cornell-tech-andre-f-t-martins-tecnico-lisboa">Quality-Aware Generation: Reranking Laws and Insights from Communication Theory</a>”, MIT (“Dynamic Sparsity for Machine Learning”), and EPFL “<a href="https://memento.epfl.ch/event/xcomet-tower-eurollm-open-multilingual-llms-for-eu/">xCOMET, Tower, EuroLLM: Open &amp; Multilingual LLMs for Europe</a>”.</p>`,
  "tags": [
  "invited-talk",
  "mit",
  "cornell",
  "epfl"
]
},
  {
  "date": "01/09/2024",
  "type": "publication",
  "title": "TowerLLM Release and COLM 2024 Spotlight",
  "content": `<p>We released TowerLLM 7B and 13B: multilingual LLMs for translation-related tasks. Check our <a href="https://huggingface.co/collections/Unbabel/tower-659eaedfe36e6dd29eb1805c">Tower Collection</a> at Hugging Face. These models and datasets are now widely used by the community (200k+ downloads so far). We presented this work as a spotlight paper at COLM 2024.</p>`,
  "tags": [
  "tower-llm",
  "multilingual",
  "translation",
  "colm",
  "spotlight"
]
},
  {
  "date": "01/08/2024",
  "type": "award",
  "title": "WMT 2024 Shared Task Victory",
  "content": `<p>We participated for the first time in the WMT 2024 shared task on General Translation — and we were the best participating system, with the best results in 8 out of 11 languages! (Bonus: we also won the Biomedical and the Chat Translation task!) Fruit of this work, Unbabel launched <a href="https://www.widn.ai/">Widn.Ai</a> — the highest quality MT engine which can be personalized with instructions and used as an API. Try it out!</p>`,
  "tags": [
  "wmt",
  "machine-translation",
  "shared-task",
  "winner"
]
},
  {
  "date": "15/07/2024",
  "type": "publication",
  "title": "xCOMET: State-of-the-Art MT Evaluation Model",
  "content": `<p>We built <a href="https://huggingface.co/collections/Unbabel/xcomet-659eca973b3be2ae4ac023bb">xCOMET</a>, a state-of-the-art interpretable model for MT evaluation and quality estimation. Give it a try! It was published in TACL and presented at ACL 2024.</p>`,
  "tags": [
  "xcomet",
  "evaluation",
  "tacl",
  "acl",
  "interpretable"
]
},
  {
  "date": "01/06/2024",
  "type": "funding",
  "title": "AI Boost Large AI Grand Challenge Grant",
  "content": `<p>We were one of the 4 winning projects of the Large AI Grand Challenge grant (AI Boost), a highly competitive grant which comes with 2M GPU hours. We are using this allocation to train a mixture-of-experts version of Tower.</p>`,
  "tags": [
  "ai-boost",
  "grant",
  "gpu-hours",
  "mixture-of-experts"
]
},
  {
  "date": "01/05/2024",
  "type": "publication",
  "title": "20+ Papers at Top 2024 Conferences",
  "content": `<p>In 2024, our team presented 20+ papers in top conferences (including NeurIPS, ICLR, ICML, COLM, TACL, EMNLP, COLM, ICML, ...). We had spotlight/oral papers at ICML, NeurIPS, and COLM. We presented in several keynote talks in workshops and other events.</p>`,
  "tags": [
  "conferences",
  "publications",
  "spotlight",
  "keynote"
]
},
  {
  "date": "01/04/2024",
  "type": "service",
  "title": "ACL 2024 Program Co-Chair",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> served as Program Co-Chair of ACL 2024.</p>`,
  "tags": [
  "acl",
  "program-chair",
  "service"
]
},
  {
  "date": "15/01/2024",
  "type": "funding",
  "title": "ERC Consolidator Grant DECOLLAGE Awarded",
  "content": `<p><strong>Great news: <a href="https://andre-martins.github.io/">André Martins</a> received an ERC (European Research Council) Consolidator Grant on “Deep Cognition Learning for Language Generation (DECOLLAGE)”. <a href="pages/jobs.html">He is now looking for Post-Doc and PhD Students</a>.</strong></p>`,
  "tags": [
  "erc",
  "consolidator-grant",
  "decollage",
  "recruitment"
]
},
  {
  "date": "15/09/2022",
  "type": "presentation",
  "title": "SEPLN 2022 Keynote",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> gave a keynote talk at the <a href="https://sepln2022.grupolys.org/">SEPLN 2022</a> conference.</p>`,
  "tags": [
  "sepln",
  "keynote",
  "structured-prediction"
]
},
  {
  "date": "01/08/2022",
  "type": "presentation",
  "title": "Mercury Machine Learning Lab Keynote",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> gave a keynote at the Mercury Machine Learning Lab (<a href="https://icai.ai/mercury-machine-learning-lab/">MMLL</a>) seminar series. He talked about how to go from sparse modeling to sparse communication. Check the video <a href="https://www.youtube.com/watch?v=UFsCAr4kIc0&amp;list=PLTg_E6ob657XajMOqJ4HxfQcv49f8xD_Z&amp;t=8s">here</a>!</p>`,
  "tags": [
  "mmll",
  "keynote",
  "sparse-modeling",
  "video"
]
},
  {
  "date": "01/07/2022",
  "type": "event",
  "title": "LxMLS 2022 Back In-Person",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> co-organized LxMLS 2022 (Lisbon Machine Learning School), back to in-person that year! See <a href="http://lxmls.it.pt">here</a> for details!</p>`,
  "tags": [
  "lxmls",
  "machine-learning-school",
  "in-person",
  "organizing"
]
},
  {
  "date": "01/06/2022",
  "type": "publication",
  "title": "Multiple 2022 Conference Acceptances",
  "content": `<p>We have new papers accepted at CLeaR 2022, ICLR 2022, ACL 2022, NAACL 2022, and ICML 2022.</p>`,
  "tags": [
  "clear",
  "iclr",
  "acl",
  "naacl",
  "icml"
]
},
  {
  "date": "01/05/2022",
  "type": "presentation",
  "title": "TRITON Conference Keynote",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> gave a keynote talk at the <a href="https://triton-conference.org/">TRITON</a> conference.</p>`,
  "tags": [
  "triton",
  "keynote"
]
},
  {
  "date": "01/04/2022",
  "type": "presentation",
  "title": "TALN Keynote on DeepSPIN Project",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> gave a keynote talk at <a href="https://talnrecital2021.inria.fr">TALN</a> where he presented some of the work we did in the <a href="https://deep-spin.github.io/">DeepSPIN</a> project. <a href="docs/taln2021.pdf">Here</a> are the slides.</p>`,
  "tags": [
  "taln",
  "keynote",
  "deepspin",
  "slides"
]
},
  {
  "date": "01/07/2021",
  "type": "event",
  "title": "LxMLS 2021 Goes Virtual",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> co-organized LxMLS 2021 (Lisbon Machine Learning School), which was a fully remote school that year. See <a href="http://lxmls.it.pt">here</a> for details!</p>`,
  "tags": [
  "lxmls",
  "virtual",
  "remote",
  "organizing"
]
},
  {
  "date": "01/06/2021",
  "type": "publication",
  "title": "NAACL and ACL 2021 Publications",
  "content": `<p>We have new papers accepted at NAACL 2021 and ACL 2021.</p>`,
  "tags": [
  "naacl",
  "acl",
  "publications"
]
},
  {
  "date": "01/01/2021",
  "type": "service",
  "title": "ELLIS NLP Program Co-Direction",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> is co-directing the <a href="https://ellis.eu/programs/natural-language-processing">ELLIS NLP program</a> with <a href="https://www.informatik.tu-darmstadt.de/ukp/ukp_home/staff_ukp/prof_dr_iryna_gurevych/index.en.jsp">Iryna Guleyvich</a> and <a href="http://ivan-titov.org/">Ivan Titov</a>, with an amazing list of fellows and scholars!</p>`,
  "tags": [
  "ellis",
  "nlp-program",
  "co-director",
  "network"
]
},
  {
  "date": "01/09/2019",
  "type": "presentation",
  "title": "EurNLP Summit Invited Talk",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> gave an invited talk at the <a href="https://www.eurnlp.org">First EurNLP Summit</a> in London.</p>`,
  "tags": [
  "eurnlp",
  "invited-talk",
  "london"
]
},
  {
  "date": "01/08/2019",
  "type": "presentation",
  "title": "Summer School Speaking Tour",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> gave invited talks at 3 Summer schools: <a href="http://lxmls.it.pt">LxMLS 2019</a> in Lisbon, <a href="http://athnlp.iit.demokritos.gr">AthNLP 2019</a> in Athens, and <a href="https://www.mlrs.ai">MLRS 2019</a> in Bangkok.</p>`,
  "tags": [
  "summer-school",
  "lxmls",
  "athnlp",
  "mlrs",
  "invited-talk"
]
},
  {
  "date": "15/07/2019",
  "type": "event",
  "title": "LxMLS 2019 Organization",
  "content": `<p>Several deep spinners are organizing/serving as monitors in LxMLS 2019 (Lisbon Machine Learning School). See <a href="http://lxmls.it.pt">here</a> for details!</p>`,
  "tags": [
  "lxmls",
  "organizing",
  "machine-learning-school"
]
},
  {
  "date": "01/07/2019",
  "type": "presentation",
  "title": "ACL 2019 Tutorial on Latent Structure Models",
  "content": `<p>We presented a <a href="https://deep-spin.github.io/tutorial/">tutorial on Latent Structure Models for NLP</a> at ACL 2019 in Florence.</p>`,
  "tags": [
  "acl",
  "tutorial",
  "latent-structure",
  "florence"
]
},
  {
  "date": "01/06/2019",
  "type": "publication",
  "title": "AISTATS, NAACL, and ACL 2019 Acceptances",
  "content": `<p>We have new papers accepted at AISTATS, NAACL, and ACL 2019.</p>`,
  "tags": [
  "aistats",
  "naacl",
  "acl",
  "publications"
]
},
  {
  "date": "15/05/2019",
  "type": "award",
  "title": "WMT 2019 Quality Estimation Victory",
  "content": `<p>A joint team with Unbabel and IT won the WMT 2019 Shared Task on Quality Estimation! Check the results <a href="http://www.statmt.org/wmt19/qe-results.html">here</a>!</p>`,
  "tags": [
  "wmt",
  "quality-estimation",
  "shared-task",
  "winner",
  "unbabel"
]
},
  {
  "date": "01/05/2019",
  "type": "award",
  "title": "OpenKiwi Best Demo Paper Award",
  "content": `<p>We received the <strong>best system demo paper award</strong> for OpenKiwi, a Pytorch-based software toolkit for translation quality estimation. Check the <a href="https://github.com/Unbabel/OpenKiwi">repo</a> and the <a href="https://arxiv.org/abs/1902.08646">demo paper</a> at ACL 2019!</p>`,
  "tags": [
  "openkiwi",
  "demo-paper",
  "award",
  "pytorch",
  "quality-estimation"
]
},
  {
  "date": "01/04/2019",
  "type": "team",
  "title": "Marcos Treviso Joins Team",
  "content": `<p>Marcos Treviso joined the <a href="team.html">team</a> as a PhD student!</p>`,
  "tags": [
  "team",
  "phd-student",
  "marcos-treviso"
]
},
  {
  "date": "01/11/2018",
  "type": "publication",
  "title": "EMNLP 2018 SparseMAP Paper",
  "content": `<p>We have a new EMNLP paper where we propose <em>SparseMAP</em> to build dynamic computation graphs via sparse latent structure (work done in collaboration with <a href="http://www.cs.cornell.edu/home/cardie">Claire Cardie</a>). Keep posted!</p>`,
  "tags": [
  "emnlp",
  "sparsemap",
  "dynamic-computation",
  "sparse-structure"
]
},
  {
  "date": "15/09/2018",
  "type": "team",
  "title": "New PhD Students Join Team",
  "content": `<p>Tsvetomila Mihaylova and Ben Peters joined the <a href="team.html">team</a> as PhD students!</p>`,
  "tags": [
  "team",
  "phd-students",
  "tsvetomila-mihaylova",
  "ben-peters"
]
},
  {
  "date": "01/09/2018",
  "type": "team",
  "title": "Vlad Niculae Joins as Post-Doc",
  "content": `<p>Vlad Niculae joined the <a href="team.html">team</a> as a post-doc researcher!</p>`,
  "tags": [
  "team",
  "post-doc",
  "vlad-niculae"
]
},
  {
  "date": "01/08/2018",
  "type": "team",
  "title": "Erick Fonseca Joins as Post-Doc",
  "content": `<p>Erick Fonseca joined the <a href="team.html">team</a> as a post-doc researcher!</p>`,
  "tags": [
  "team",
  "post-doc",
  "erick-fonseca"
]
},
  {
  "date": "15/07/2018",
  "type": "presentation",
  "title": "ACL 2018 Workshop Invited Talk",
  "content": `<p><a href="https://andre-martins.github.io/">André Martins</a> gave an invited talk in the <a href="https://sites.google.com/site/wnmt18">ACL 2018 Workshop on Neural Machine Translation and Generation</a>. Here are the <a href="https://docs.google.com/viewer?a=v&amp;pid=sites&amp;srcid=ZGVmYXVsdGRvbWFpbnx3bm10MTh8Z3g6NzM2ZWNhMTk2MTdlODQ2Yw">slides</a>.</p>`,
  "tags": [
  "acl",
  "workshop",
  "invited-talk",
  "neural-mt",
  "slides"
]
},
  {
  "date": "01/06/2018",
  "type": "event",
  "title": "LxMLS 2018 Co-Organization",
  "content": `<p>We co-organized LxMLS 2018 (Lisbon Machine Learning School). See <a href="http://lxmls.it.pt">here</a> for details!</p>`,
  "tags": [
  "lxmls",
  "co-organizing",
  "machine-learning-school"
]
},
  {
  "date": "01/05/2018",
  "type": "publication",
  "title": "ICML 2018 SparseMAP Paper",
  "content": `<p>We have a new <a href="http://proceedings.mlr.press/v80/niculae18a.html">ICML paper</a> where we propose <em>SparseMAP</em> as a new inference procedure for sparse structured prediction (work done in collaboration with <a href="http://vene.ro">Vlad Niculae</a>, <a href="http://mblondel.org">Mathieu Blondel</a>, and <a href="http://www.cs.cornell.edu/home/cardie">Claire Cardie</a>).</p>`,
  "tags": [
  "icml",
  "sparsemap",
  "inference",
  "structured-prediction"
]
},
  {
  "date": "01/04/2018",
  "type": "publication",
  "title": "ACL 2018 Sparse Attention Paper",
  "content": `<p>We have a new <a href="http://aclweb.org/anthology/P18-2059">ACL short paper</a> where we use new forms of sparse and constrained attention within neural machine translation (work done in collaboration with Chaitanya Malaviya and Pedro Ferreira).</p>`,
  "tags": [
  "acl",
  "sparse-attention",
  "neural-mt",
  "constrained-attention"
]
},
  {
  "date": "01/03/2018",
  "type": "event",
  "title": "WMT18 Quality Estimation Co-Organization",
  "content": `<p>With Unbabel, we co-organized the WMT18 shared task in quality estimation. See <a href="http://www.statmt.org/wmt18/quality-estimation-task.html">here</a> for details!</p>`,
  "tags": [
  "wmt",
  "quality-estimation",
  "shared-task",
  "co-organizing",
  "unbabel"
]
},
  {
  "date": "01/02/2018",
  "type": "team",
  "title": "Gon\u00e7alo Correia Joins as PhD Student",
  "content": `<p>Gonçalo Correia joined the <a href="team.html">team</a> as a PhD student!</p>`,
  "tags": [
  "team",
  "phd-student",
  "goncalo-correia"
]
}
];

// Function to get latest news (used by index.html)
function getLatestNews(limit = 5) {
    return newsData.slice(0, limit);
}

// Function to get all news (used by news.html)
function getAllNews() {
    return newsData;
}

// Function to filter news by type
function filterNewsByType(type) {
    if (type === 'all') return newsData;
    return newsData.filter(news => news.type === type);
}

// Function to search news
function searchNews(query) {
    const searchTerm = query.toLowerCase();
    return newsData.filter(news => 
        news.title.toLowerCase().includes(searchTerm) ||
        news.content.toLowerCase().includes(searchTerm) ||
        news.tags.some(tag => tag.toLowerCase().includes(searchTerm))
    );
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        newsData,
        getLatestNews,
        getAllNews,
        filterNewsByType,
        searchNews
    };
}