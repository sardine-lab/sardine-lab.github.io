// Metadata and common content for SARDINE Lab website

const SARDINE_METADATA = {
    // Site information
    site: {
        name: "SARDINE Lab",
        fullName: "Structure AwaRe moDelIng for Natural LanguagE",
        icon: "🐟",
        logo: {
            path: "assets/figs/logo.png",
            alt: "SARDINE Lab Logo",
            width: "220"
        },
        logoDark: {
            path: "assets/figs/logo-dark.png",
            alt: "SARDINE Lab Logo",
            width: "200"
        }
    },

    // Contact information
    contact: {
        email: "sardinelab@gmail.com",
        address: "Av. Rovisco Pais 1, 1049-001 Lisboa",
        location: "Lisbon, Portugal"
    },

    // Research streams that appear in the footer
    researchStreams: [
        "Natural Language Processing",
        "Deep Learning", 
        "Structured Prediction",
        "Sparse Modeling",
        "Machine Translation",
    ],

    // Social networks
    socialNetworks: [
        {
            name: "GitHub",
            url: "https://github.com/deep-spin",
            icon: "fab fa-github"
        },
        
        {
            name: "Twitter",
            url: "https://x.com/deep_spin",
            icon: "fab fa-twitter"
        },
        {
            name: "BlueSky", 
            url: "https://web-cdn.bsky.app/profile/sardine-lab-it.bsky.social",
            icon: "fab fa-bluesky"
        },
        {
            name: "Email",
            url: "mailto:sardinelab@gmail.com",
            icon: "fas fa-envelope"
        }
    ],

    // Navigation items
    navigation: [
        { name: "Home", href: "index.html", id: "home" },
        { name: "News", href: "news.html", id: "news" },
        { name: "Publications", href: "publications.html", id: "publications" },
        { name: "Projects", href: "projects.html", id: "projects" }
    ],

    // Affiliations
    affiliations: [
        {
            name: "Instituto de Telecomunicações",
            url: "https://www.it.pt/"
        },
        {
            name: "Instituto Superior Técnico", 
            url: "https://tecnico.ulisboa.pt/"
        },
        {
            name: "University of Lisbon", 
            url: "https://www.ulisboa.pt/en"
        }
    ]
};


class MetadataManager {
    constructor() {
        this.metadata = SARDINE_METADATA;
    }

    // Update page header/navigation
    updateHeader() {
        const header = document.querySelector('header .max-w-6xl');
        if (!header) return;

        const currentPage = this.getCurrentPage();
        
        header.innerHTML = `
            <div class="flex items-center justify-between h-16">
                <a href="index.html" class="flex items-center gap-2 group">
                    <img width="${this.metadata.site.logo.width}" src="${this.metadata.site.logo.path}" title="${this.metadata.site.name}" alt="${this.metadata.site.logo.alt}">
                    <!--<span class="font-semibold text-slate-900">${this.metadata.site.name}</span>-->
                </a>
                <nav class="hidden md:flex items-center gap-6 text-sm font-medium">
                    ${this.metadata.navigation.map(item => 
                        `<a href="${item.href}" class="navlink ${this.isActivePage(item.href, currentPage) ? 'active' : ''} text-slate-600 hover:text-slate-900 transition-colors">${item.name}</a>`
                    ).join('')}
                </nav>
                <button id="mobileMenuBtn" class="md:hidden inline-flex items-center justify-center p-2 rounded-lg border border-slate-300 text-slate-700">
                    <i class="fas fa-bars text-lg"></i>
                </button>
            </div>
            <div id="mobileMenu" class="hidden md:hidden">
                <nav class="px-4 pb-3">
                    ${this.metadata.navigation.map(item => 
                        `<a href="${item.href}" class="block px-3 py-2 rounded-lg hover:bg-slate-100 ${this.isActivePage(item.href, currentPage) ? 'active' : ''}">${item.name}</a>`
                    ).join('')}
                </nav>
            </div>
        `;

        // Re-initialize mobile menu after updating header
        this.initializeMobileMenu();
    }

    // Update footer
    updateFooter() {
        const footer = document.querySelector('footer .max-w-6xl');
        let currentYear = new Date().getFullYear();
        if (!footer) return;

        footer.innerHTML = `
            <div class="px-4 grid gap-8 md:grid-cols-3">
                <div>
                    <div class="flex items-center gap-2 mb-3">
                        <img width="${this.metadata.site.logoDark.width}" src="${this.metadata.site.logoDark.path}" title="${this.metadata.site.name}" alt="${this.metadata.site.logo.alt}">
                        <!--<span class="text-2xl">${this.metadata.site.icon}</span>-->
                        <!--<span class="font-semibold">${this.metadata.site.name}</span>-->
                    </div>
                    <p class="text-sm text-slate-400">${this.metadata.site.fullName}</p>
                    <p class="text-sm text-slate-400 mt-2">
                        ${this.metadata.affiliations.map(aff => 
                            `<a href="${aff.url}" target="_blank" class="hover:text-white transition-colors">${aff.name}</a>`
                        ).join(' <br> ')} 
                    </p>
                    <p class="text-sm text-slate-400 mt-3">${this.metadata.contact.address}</p> 
                </div>
                <div>
                    <h3 class="text-sm font-semibold text-slate-100 mb-3">Research Streams</h3>
                    <ul class="space-y-2 text-sm text-slate-400">
                        ${this.metadata.researchStreams.map(area => `<li>${area}</li>`).join('')}
                    </ul>
                </div>
                <div>
                    <h3 class="text-sm font-semibold text-slate-100 mb-3">Contact & Links</h3>
                    <div class="space-y-2 text-sm text-slate-400">
                        <p>Email: <a class="underline hover:text-white transition-colors" href="mailto:${this.metadata.contact.email}">${this.metadata.contact.email}</a></p>
                        <p>Talks: <a class="underline hover:text-white transition-colors" target="_blank" href="https://sardine-lab.github.io/ellis-sardine-seminars/">ELLIS-SARDINE Seminars</a></p>
                        <div class="flex space-x-4 mt-4">
                            ${this.metadata.socialNetworks.map(social => 
                                `<a href="${social.url}" class="text-slate-400 hover:text-white transition-colors" ${social.url.startsWith('http') ? 'target="_blank"' : ''}>
                                    <i class="${social.icon} text-lg"></i>
                                </a>`
                            ).join('')}
                        </div>
                    </div>
                </div>
            </div>
            <div class="border-t border-cyan-800 mt-8 pt-8 text-center text-sm" style="color:#4f8591;">
                <div class="max-w-6xl mx-auto px-4 flex items-center justify-between">
                    <span>&copy; ${currentYear} ${this.metadata.site.name}. All rights reserved.</span>
                    <span>Made with ${this.metadata.site.icon} in Lisbon.</span>
                </div>
            </div>
        `;
    }

    // Update page title
    updatePageTitle(pageTitle = '') {
        const baseTitle = this.metadata.site.name;
        document.title = pageTitle ? `${baseTitle} — ${pageTitle}` : baseTitle;
    }

    // Update favicon
    updateFavicon() {
        let favicon = document.querySelector('link[rel="icon"]');
        if (!favicon) {
            favicon = document.createElement('link');
            favicon.rel = 'icon';
            document.head.appendChild(favicon);
        }
        favicon.href = `data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%22100%22>${this.metadata.site.icon}</text></svg>`;
    }

    // Get current page from URL
    getCurrentPage() {
        const path = window.location.pathname;
        const page = path.split('/').pop() || 'index.html';
        return page === '' ? 'index.html' : page;
    }

    // Check if page is active
    isActivePage(href, currentPage) {
        // Handle index page special cases
        if (currentPage === 'index.html' || currentPage === '') {
            return href === 'index.html';
        }
        return href === currentPage;
    }

    // Initialize mobile menu functionality
    initializeMobileMenu() {
        const mobileMenuBtn = document.getElementById('mobileMenuBtn');
        const mobileMenu = document.getElementById('mobileMenu');

        if (mobileMenuBtn && mobileMenu) {
            // Remove any existing event listeners
            const newBtn = mobileMenuBtn.cloneNode(true);
            mobileMenuBtn.parentNode.replaceChild(newBtn, mobileMenuBtn);

            newBtn.addEventListener('click', () => {
                mobileMenu.classList.toggle('hidden');
            });

            // Close mobile menu when clicking outside
            document.addEventListener('click', (e) => {
                if (!newBtn.contains(e.target) && !mobileMenu.contains(e.target)) {
                    mobileMenu.classList.add('hidden');
                }
            });
        }
    }

    // Initialize all metadata updates
    init() {
        this.updateHeader();
        this.updateFooter();
        this.updateFavicon();
        
        // Set page-specific titles
        const currentPage = this.getCurrentPage();
        const pageMap = {
            'news.html': 'News',
            'publications.html': 'Publications', 
            'projects.html': 'Projects'
        };
        
        if (pageMap[currentPage]) {
            this.updatePageTitle(pageMap[currentPage]);
        }
    }

    // Get metadata for external use
    getMetadata() {
        return this.metadata;
    }

    // Get research streams
    getResearchStreams() {
        return RESEARCH_STREAMS;
    }
}

// Function to get all available streams
function getAllStreams() {
    return Object.keys(RESEARCH_STREAMS).sort();
}

// Function to get stream metadata
function getStreamMetadata(streamId) {
    if (!RESEARCH_STREAMS[streamId]) {
        console.warn(`Stream "${streamId}" not found in RESEARCH_STREAMS`);
    }
    return RESEARCH_STREAMS[streamId] || {
        name: streamId.charAt(0).toUpperCase() + streamId.slice(1),
        color: "#6b7280",
        icon: "fas fa-tag",
        description: `Undefined stream: ${streamId}`
    };
}

// Initialize metadata when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const metadataManager = new MetadataManager();
    metadataManager.init();
    
    // Make it globally available
    window.SardineMetadata = metadataManager;
    
    // Make stream functions globally available
    window.getAllStreams = getAllStreams;
    window.getStreamMetadata = getStreamMetadata;
    window.RESEARCH_STREAMS = RESEARCH_STREAMS;
});

// Export for module usage (if using modules)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        MetadataManager, 
        SARDINE_METADATA, 
        RESEARCH_STREAMS,
        getAllStreams,
        getStreamMetadata
    };
}
