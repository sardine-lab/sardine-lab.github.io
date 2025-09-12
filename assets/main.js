// Main JavaScript functionality for SARDINE Lab website
// Shared utilities and SardineWebsite class

class SardineWebsite {
    constructor() {
        this.init();
    }

    init() {
        this.setupSmoothScrolling();
        this.setupScrollEffects();
        this.setActiveNavigation();
    }

    // Smooth scrolling for anchor links
    setupSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
                // Close mobile menu if open
                const mobileMenu = document.getElementById('mobileMenu');
                if (mobileMenu) {
                    mobileMenu.classList.add('hidden');
                }
            });
        });
    }

    // Scroll effects for navigation
    setupScrollEffects() {
        window.addEventListener('scroll', () => {
            const header = document.querySelector('header');
            if (header) {
                if (window.scrollY > 100) {
                    header.classList.add('backdrop-blur-sm', 'bg-white/95');
                } else {
                    header.classList.remove('backdrop-blur-sm', 'bg-white/95');
                }
            }
        });
    }

    // Set active navigation based on current page
    setActiveNavigation() {
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        const navLinks = document.querySelectorAll('.navlink');
        
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === currentPage || 
                (currentPage === 'index.html' && href === '#about') ||
                (currentPage === '' && href === '#about')) {
                link.classList.add('active');
            }
        });
    }

    // Utility function to format dates
    // Returns a nicely formatted string or the original if parsing fails.
    static formatDate(dateString, locale = 'en-US') {
      if (!dateString) return '';

      const s = String(dateString).trim();

      // ISO: yyyy-mm-dd
      let m = /^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(s);
      if (m) {
        const [, y, mo, d] = m;
        const dt = new Date(Number(y), Number(mo) - 1, Number(d));
        return dt.toLocaleDateString(locale, { year: 'numeric', month: 'long', day: 'numeric' });
      }

      // D/M/Y or DD/MM/YYYY
      m = /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/.exec(s);
      if (m) {
        const [, d, mo, y] = m;
        const dt = new Date(Number(y), Number(mo) - 1, Number(d));
        return dt.toLocaleDateString(locale, { year: 'numeric', month: 'long', day: 'numeric' });
      }

      // Fallback: let Date parse other formats it knows
      const t = Date.parse(s);
      if (!Number.isNaN(t)) {
        const dt = new Date(t);
        return dt.toLocaleDateString(locale, { year: 'numeric', month: 'long', day: 'numeric' });
      }

      // Last resort: show the original text
      return s;
    }

    // Utility function to create filter buttons
    static createFilterButtons(container, filters, activeFilter, onFilterChange) {
        container.innerHTML = '';
        
        filters.forEach(filter => {
            const btn = document.createElement('button');
            btn.className = `filter-btn ${filter.value === activeFilter ? 'active' : ''}`;
            btn.textContent = filter.label;
            btn.onclick = () => onFilterChange(filter.value);
            container.appendChild(btn);
        });
    }

    // Utility function to debounce search input
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Index page functionality
class IndexPage {
    constructor() {
        this.homeSelectedStreams = new Set();
        this.numRecentPublications = 4;
        this.numLatestNews = 4;
        this.numProjects = 2;
        this.currentPhotoIndex = 0;
        this.init();
    }

    init() {
        this.initializeHomePage();
        this.initPhotoSlider();
        this.setupKeyboardNavigation();
    }

    initializeHomePage() {
        this.renderLatestNews();
        this.renderHomeStreamFilters();
        this.renderRecentPublications();
        this.renderCurrentProjects();
        this.renderTeamSection();
    }

    renderLatestNews() {
        if (typeof newsData !== 'undefined' && typeof NewsManager !== 'undefined') {
            const newsManager = new NewsManager(newsData);
            const latestNewsContainer = document.getElementById('latestNews');
            if (latestNewsContainer) {
                newsManager.renderLatestNewsTimeline(latestNewsContainer, this.numLatestNews);
            }
        }
    }


    renderRecentPublications() {
        if (typeof publicationsData === 'undefined' || typeof PublicationsManager === 'undefined') return;

        const container = document.getElementById('homePublications');
        if (!container) return;

        if (!this.pubManager) this.pubManager = new PublicationsManager(publicationsData);

        this.pubManager.renderRecentPublications(container, {
            limit: this.numRecentPublications,
            streams: Array.from(this.homeSelectedStreams)
        });
    }

    renderHomeStreamFilters() {
        const container = document.getElementById('homeStreamFilters');
        if (!container) return;
        
        const allStreams = getAllStreams();
        
        container.innerHTML = '';
        
        // Add "All Streams" button
        const allButton = document.createElement('button');
        allButton.className = `px-3 py-1 rounded-full text-sm font-medium transition-all ${
            this.homeSelectedStreams.size === 0 ? 'bg-sardine-blue text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
        }`;
        allButton.innerHTML = '<i class="fas fa-stream mr-2"></i>All Streams';
        allButton.onclick = () => {
            this.homeSelectedStreams.clear();
            this.renderHomeStreamFilters();
            this.renderRecentPublications();
        };
        container.appendChild(allButton);

        // Add individual stream buttons
        allStreams.forEach(streamId => {
            const metadata = getStreamMetadata(streamId);
            const isActive = this.homeSelectedStreams.has(streamId);
            
            const button = document.createElement('button');
            button.className = `px-3 py-1 rounded-full text-sm font-medium transition-all ${
                isActive ? 'text-white shadow-lg' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`;
            
            if (isActive) {
                button.style.backgroundColor = metadata.color;
                button.style.boxShadow = `0 4px 14px 0 ${metadata.color}40`;
            }
            
            button.innerHTML = `<i class="${metadata.icon} mr-2"></i>${metadata.name}`;
            button.title = metadata.description;
            button.onclick = () => {
                if (this.homeSelectedStreams.has(streamId)) {
                    this.homeSelectedStreams.delete(streamId);
                } else {
                    this.homeSelectedStreams.add(streamId);
                }
                this.renderHomeStreamFilters();
                this.renderRecentPublications();
            };
            
            container.appendChild(button);
        });
    }

    renderCurrentProjects() {
        if (typeof projectsData === 'undefined' || typeof ProjectsManager === 'undefined') return;
        
        const projectsContainer = document.getElementById('currentProjects');
        if (!projectsContainer) return;
        
        const projectsManager = new ProjectsManager(projectsData);
        const currentProjects = projectsManager.getCurrentProjects(this.numProjects);
        
        projectsContainer.innerHTML = '';

        currentProjects.forEach(project => {
            const projectCard = projectsManager.createProjectCard(project);
            projectsContainer.appendChild(projectCard);
        });
    }

    renderTeamSection() {
        if (typeof teamData === 'undefined' || typeof TeamManager === 'undefined' || typeof COUNTRY_DATABASE === 'undefined') return;
        
        const teamManager = new TeamManager(teamData, COUNTRY_DATABASE);
        teamManager.renderTeamSection('teamMembers');
    }

    // Photo slider functionality
    initPhotoSlider() {
        if (typeof GROUP_PHOTOS === 'undefined' || typeof PhotoSliderManager === 'undefined') {
            console.warn('Group photos requirements not met.');
            return;
        }

        window.photoSlider = new PhotoSliderManager(GROUP_PHOTOS);
        window.photoSlider.init();
    }

    setupKeyboardNavigation() {
        // Keyboard navigation is handled by PhotoSliderManager
    }
}

// Initialize website based on current page
document.addEventListener('DOMContentLoaded', () => {
    new SardineWebsite();
    
    // Initialize page-specific functionality
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    
    if (currentPage === 'index.html' || currentPage === '') {
        new IndexPage();
    }
});