// Projects data for SARDINE Lab website

const projectsData = {
  "current": [
  {
  "name": "SMURF4EU",
  "title": "A Suite of Multimodal Reasoning Foundation Models for Europe",
  "status": "current",
  "funding": "EuroHPC - Extreme Scale Access, 1.9M GPU-hours",
  "pi": "Andr\u00e9 Martins, Giuseppe Attanasio,  Marcos Treviso",
  "period": "2026-2027",
  "team_members": [
  "Duarte Alves",
  "Guilherme Viveiros",
  "Javier Gilabert",
  "Manos Zaranis",
  "Matthias Lindemann",
  "Miguel Faria",
  "Miguel Ramos",
  "Saul Santos",
  "Sonal Sannigrahi"
],
  "collaborators": [
  "Alexandra Birch",
  "Ben Kabongo",
  "Barry Haddow",
  "Dominik Machacek",
  "Edoardo Ponti",
  "Francois Yvon",
  "Giulio Zhou",
  "Laurent Bessacier",
  "Luisa Bentivogli",
  "Marco Gaido",
  "Matteo Negri",
  "Patrick Fernandes",
  "Pierre Colombo",
  "Sara Papi",
  "Sherrie Shen",
  "Syrielle Montariol",
  "Tsz Kin Lam"
],
  "keywords": [
  "multimodality",
  "multilinguality",
  "language generation",
  "long-context modeling",
  "reasoning",
  "test-time scaling"
],
  "figure": "smurf4eu.jpeg",
  "description": `<p><strong>SMURF4EU</strong> will develop and release a suite of fully open, high-performance <strong>multimodal reasoning foundation models</strong> for Europe, spanning <strong>text, code, speech, vision, and video</strong>. The project targets a major gap in today’s landscape: the lack of transparent, reproducible, and EU-language-capable alternatives to proprietary multimodal systems. Building on <a href="https://eurollm.io/">EuroLLM</a>, SMURF4EU will deliver multiple model sizes designed to support <strong>all 24 official EU languages</strong> and to enable <strong>multimodal reasoning</strong> through reinforcement learning and verifier-style rewards..</p>
<p>&nbsp;</p>
<p>A core technical goal is a <strong>modular training recipe</strong> that makes it efficient to add modalities to strong text models via late fusion and shared token-like representations, while scaling to <strong>long multimodal contexts (up to 1M tokens)</strong> using efficient attention and memory-compression techniques. Beyond releasing model weights, the project commits to <strong>open science</strong> by publishing checkpoints, datasets (where licensing permits), code, and documentation, following best practices for safety and compliance (including GDPR-aware data handling and alignment practices) to foster European research, innovation, and industrial adoption.</p>`
},
  {
  "name": "DECOLLAGE",
  "title": "Deep Cognition Learning for Language Generation",
  "status": "current",
  "funding": "ERC Consolidator Grant",
  "pi": "Andr\u00e9 Martins",
  "period": "2023-2028",
  "team_members": [
  "Matthias Lindemann",
  "Beatriz Canaverde",
  "Duarte Alves",
  "Miguel Ramos",
  "Saul Santos",
  "Sophia Sklaviadis"
],
  "keywords": [
  "sparse models",
  "dynamic memory",
  "multimodality",
  "cognitive science",
  "language generation"
],
  "website": "https://andre-martins.github.io/pages/decollage.html",
  "figure": "decollage.jpeg",
  "publications": [
  "\u221e-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation"
],
  "description": `<p>The DECOLLAGE project aims to overcome fundamental limitations of today’s large-scale language models by combining advances in machine learning, sparse modeling, information theory, and cognitive science. It pursues three core directions: developing mechanisms for self-assessment and contextualization, designing dynamic memory structures for continual learning, and creating mathematical models for sparse communication that bridge symbolic and continuous representations across modalities. By integrating these innovations, DECOLLAGE seeks to build modular, efficient, and cognitively inspired architectures, ultimately enabling more adaptive, context-aware, and intelligent systems for demanding language generation tasks such as machine translation and open-ended text generation.</p>`
},
  {
  "name": "AMALIA",
  "title": "European-Portugese LLM",
  "status": "current",
  "funding": "Portugal",
  "pi": "Andr\u00e9 Martins, Jo\u00e3o Magalh\u00e3es",
  "period": "2025-2026",
  "team_members": [
  "Marcos Treviso",
  "Giuseppe Attanasio",
  "Miguel Ramos",
  "Duarte Alves"
],
  "collaborators": [
  "UNL (Universidade Nova de Lisboa)"
],
  "keywords": [
  "european portuguese llm",
  "cultural representativeness",
  "sovereignty AI",
  "open model"
],
  "website": "https://amaliallm.pt/",
  "figure": "amalia.jpeg",
  "description": `<p><strong>AMALIA</strong> is an <strong>open large language model</strong> developed specifically for <strong>Portuguese as used in Portugal</strong> and for <strong>Portuguese cultural context</strong>, aiming to strengthen linguistic quality across Portuguese variants, preserve culturally grounded language use (idioms, references), and support <strong>data sovereignty</strong>—especially for use cases in the <strong>Public Administration</strong> where sensitive data should remain under national control. </p>
<p>&nbsp;</p>
<p>Technically, AMALIA’s first phase centers on a model of about <strong>9B parameters</strong>, pre-trained on roughly <strong>4 trillion words</strong>, and fine-tuned on curated Portuguese data (including material filtered from <strong>Arquivo.PT</strong>), leveraging prior Portuguese-focused models from the team (e.g., EuroLLM and GlorIA). The project is built by a national consortium (NOVA, IST, Coimbra, Porto, Minho, and FCT/Arquivo.PT), trained using <strong>large-scale supercomputing</strong> (including <strong>MareNostrum 5</strong> and <strong>Deucalion</strong>) and is being rolled out progressively via research and institutional access (e.g., through <strong>IAEdu</strong> via API), with a roadmap toward broader capability expansion (including multimodality) and a <strong>targeted final, robust release around June 2026</strong>. </p>`
},
  {
  "name": "NextGenAI",
  "title": "NextGenAI: Center for Responsible AI",
  "status": "current",
  "funding": "IAPMEI - Plan for Recovery and Resilience",
  "pi": "Andr\u00e9 Martins",
  "period": "2023-2026",
  "team_members": [
  "Miguel Faria",
  "Giuseppe Attanasio",
  "Guilherme Viveiros",
  "Margarida Campos",
  "Emmanouil Zaranis",
  "Pavlo Vasylenko"
],
  "keywords": [
  "multilinguality",
  "context-aware translation",
  "dialogue systems",
  "quality estimation",
  "human-in-the-loop AI"
],
  "website": "https://centerforresponsible.ai/",
  "figure": "crai.jpg",
  "publications": [
  "Non-Exchangeable Conformal Risk Control"
],
  "description": `<p>The Center for Responsible AI (CRAI) is dedicated to advancing Artificial Intelligence that is fair, transparent, efficient, and socially responsible. Its mission is to ensure that AI technologies are developed and applied in ways that benefit people, respect human rights, and support sustainable digital transformation.</p>`
},
  {
  "name": "Ouvia",
  "title": "A User-Centered Benchmark for Usability Assessment of Speech Translation Systems",
  "status": "current",
  "funding": "European Association for Machine Translation",
  "pi": "Giuseppe Attanasio",
  "period": "2025-2026",
  "team_members": [
  "Giuseppe Attanasio",
  "Sweta Agrawal",
  "Andr\u00e9 Martins"
],
  "collaborators": [
  "Google"
],
  "keywords": [
  "speech translation",
  "usability",
  "user-centered evaluation",
  "benchmark"
],
  "website": "https://www.it.pt/Projects/Index/4932",
  "figure": "ouvia.jpeg",
  "description": `<p>This project seeks to reduce disparities in speech translation (ST) technologies by analyzing performance across demographic groups, particularly gender and ethnicity. It involves creating a multilingual benchmark to assess real-world usability, testing ST tools in both high- and low-stakes scenarios, measuring how well automatic metrics align with perceived usability, and if demographic performance gaps exist.</p>`
}
],
  "past": [
  {
  "name": "UTTER",
  "title": "Unified Transcription and Translation",
  "status": "finished",
  "funding": "EUROPE HORIZON",
  "pi": "Andr\u00e9 Martins",
  "period": "2022-2025",
  "team_members": [
  "Ben Peters",
  "Marcos Treviso",
  "Hugo Pitorro",
  "Jos\u00e9 Pombal",
  "Sonal Sannigrahi"
],
  "keywords": [
  "multilinguality",
  "meeting assistant",
  "customer support",
  "speech dialogue",
  "chat dialogue",
  "translation"
],
  "website": "https://he-utter.eu/",
  "figure": "utter.jpg",
  "publications": [
  "Tower: An Open Multilingual Large Language Model for Translation-Related Tasks"
],
  "description": `<p>The UTTER project advances online and hybrid interaction by developing extended reality (XR) technologies that seamlessly integrate human and AI agents in conversation platforms. It focuses on creating intelligent assistants for meetings and customer support, capable of real-time multilingual translation, summarization, minuting, and multimodal dialogue support. Through innovations in XR model efficiency, usability, and adaptability across languages and domains, UTTER will deliver new resources such as multilingual datasets and pretrained models, alongside prototypes including a meeting assistant with live captioning and summaries, and a multilingual customer support assistant enhanced with quality estimates and emotion tracking.</p>`
},
  {
  "name": "MAIA",
  "title": "Multilingual AI Agent Assistants for Customer Service",
  "status": "finished",
  "funding": "CMU-Portugal Large-Scale Collaborative Research project",
  "pi": "Andr\u00e9 Martins",
  "period": "2020-2023",
  "team_members": [
  "Ant\u00f3nio Farinhas",
  "Patrick Fernandes",
  "Nuno Guerreiro",
  "Taisiya Glushkova"
],
  "collaborators": [
  "Unbabel",
  "Carnegie Mellon University"
],
  "keywords": [
  "multilinguality",
  "context-aware translation",
  "dialogue systems",
  "quality estimation",
  "human-in-the-loop AI"
],
  "website": "https://unbabel.com/research/maia",
  "figure": "maia.jpeg",
  "publications": [
  "Project MAIA: Multilingual AI Agent Assistant"
],
  "description": `<p>The <strong>MAIA</strong> project is rooted in a vibrant ecosystem fostered by <strong><a href="https://tecnico.ulisboa.pt/">Instituto Superior Técnico (IST)</a></strong> and the <a href="https://cmuportugal.org">CMU-Portugal Program</a>, an enduring collaboration between Carnegie Mellon University and Portuguese research institutions that has bolstered Portugal’s innovation in ICT, AI, and machine learning since 2006. This initiative, supported by CMU-Portugal’s industry–academia partnerships and significant R&amp;D funding, has incubated several successful startups—including <strong><a href="https://en.wikipedia.org/wiki/Unbabel">Unbabel</a></strong>, a Lisbon-based AI-powered human-in-the-loop translation platform. Unbabel, born from CMU-Portugal’s research ecosystem, leads the MAIA project to develop multilingual virtual agents that enhance customer service by assisting human agents with context-aware translation, response suggestions, and emotionally intelligent quality assessment—demonstrating the powerful synergy between IST’s academic expertise, CMU-Portugal’s innovation framework, and Unbabel’s applied technology leadership.</p>`
},
  {
  "name": "DeepSPIN",
  "title": "Deep Structured Prediction in Natural Language Processing",
  "status": "finished",
  "funding": "ERC Starting Grant",
  "pi": "Andr\u00e9 Martins",
  "period": "2018-2023",
  "team_members": [
  "Vlad Niculae",
  "Erick Fonseca",
  "Chunchuan Lyu",
  "Ben Peters",
  "Gon\u00e7alo Correia",
  "Tsvetomila Mihaylova",
  "Marcos Treviso",
  "Pedro Martins",
  "Chryssa Zerva"
],
  "keywords": [
  "structured prediction",
  "deep learning",
  "latent structure",
  "weak supervision",
  "dependency parsing",
  "interpretability"
],
  "website": "https://deep-spin.github.io",
  "figure": "deepspin.jpeg",
  "publications": [
  "DeepSPIN: Deep Structured Prediction for Natural Language Processing"
],
  "description": `<p>The <strong>DeepSPIN</strong> project (“Deep Structured Prediction in Natural Language Processing”), funded by the European Research Council (ERC) and coordinated by André Martins at <strong><a href="https://www.it.pt/">Instituto de Telecomunicações</a></strong> and <strong><a href="https://tecnico.ulisboa.pt/">Instituto Superior Técnico</a></strong> in collaboration with <strong><a href="https://unbabel.com/">Unbabel</a></strong>, explored new ways to combine deep learning with structured prediction between 2018 and 2023. Tackling the limitations of current neural NLP systems—which often ignore the structural complexity of language and make critical errors in tasks like machine translation—DeepSPIN developed novel approaches for integrating planning mechanisms, latent structure induction, and weak supervision into neural networks. These advances aimed to make NLP systems more expressive, interpretable, and efficient, with applications tested in machine translation, quality estimation, and dependency parsing, and with strong ties to real-world impact through Unbabel’s multilingual translation platforms.</p>`
}
]
};

// Function to get current projects
function getCurrentProjects(limit = null) {
    if (limit === null)
        return projectsData.current
    return projectsData.current.slice(0, limit);
}


// Function to get past projects
function getPastProjects(limit = null) {
    if (limit === null)
        return projectsData.past
    return projectsData.past.slice(0, limit);
}

// Function to get all projects
function getAllProjects() {
    return [...projectsData.current, ...projectsData.past];
}

// Function to search projects
function searchProjects(query) {
    const searchTerm = query.toLowerCase();
    const allProjects = getAllProjects();
    
    return allProjects.filter(project => 
        project.name.toLowerCase().includes(searchTerm) ||
        project.title.toLowerCase().includes(searchTerm) ||
        project.description.toLowerCase().includes(searchTerm) ||
        project.keywords.some(keyword => keyword.toLowerCase().includes(searchTerm))
    );
}

// Function to get projects by status
function getProjectsByStatus(status) {
    if (status === 'current') return projectsData.current;
    if (status === 'past') return projectsData.past;
    return getAllProjects();
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        projectsData,
        getCurrentProjects,
        getPastProjects,
        getAllProjects,
        searchProjects,
        getProjectsByStatus
    };
}