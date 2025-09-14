// Publications management functionality for SARDINE Lab website

class PublicationsManager {
    constructor(publicationsData) {
        this.publicationsData = publicationsData;
        this.filteredPublications = [...publicationsData];
        this.sortBy = 'date';
        this.sortOrder = 'desc';
        this.pagination = null;
    }

    setPagination(paginationManager) {
        this.pagination = paginationManager;
    }

    renderRecentPublications(container, opts = {}) {
        const {
            limit = 4,
            streams = [], // array of stream ids to filter by; empty = no filter
        } = opts;

        if (!container) return;

        // Start from all publications
        let pubs = Array.isArray(this.publicationsData) ? [...this.publicationsData] : [];

        // Optional stream filter
        if (streams.length > 0) {
            pubs = pubs.filter(pub =>
                Array.isArray(pub.streams) && streams.some(s => pub.streams.includes(s))
            );
        }

        // Sort by recency (id desc as primary, then year desc)
        pubs.sort((a, b) => (b.id || 0) - (a.id || 0) || (b.year || 0) - (a.year || 0));

        // Take top N
        pubs = pubs.slice(0, limit);

        // If nothing to show
        container.innerHTML = '';
        if (pubs.length === 0) {
            container.innerHTML = `
                <div class="text-center py-8 text-slate-500 col-span-full">
                    <i class="fas fa-search text-2xl mb-2"></i>
                    <p>No publications found for selected streams</p>
                    <p class="text-sm mt-2">Try selecting different streams or view all publications</p>
                </div>
            `;
            return;
        }

        const typeColors = {
            'conference': 'bg-blue-100 text-blue-800',
            'journal': 'bg-green-100 text-green-800',
            'preprint': 'bg-orange-100 text-orange-800',
            'book': 'bg-indigo-100 text-indigo-800',
        };

        pubs.forEach(pub => {
            const links = Object.entries(pub.links || {}).map(([type, url]) => {
                const icons = {
                    'paper': 'fas fa-file',
                    'code': 'fab fa-github',
                    'demo': 'fas fa-play',
                    'bibtex': 'fas fa-quote-right'
                };
                
                if (type === 'bibtex') {
                    // url = bibtex content
                    return `
                        <button class="bibtex-btn inline-flex items-center px-3 py-1 bg-slate-100 text-slate-700 rounded text-sm hover:bg-slate-200 transition-colors" data-bibtex="${encodeURIComponent(url)}">
                            <i class="${icons[type]} mr-1"></i>
                            BibTeX
                        </button>
                    `;
                } else {
                    return `
                        <a href="${url}" class="inline-flex items-center px-3 py-1 bg-slate-100 text-slate-700 rounded text-sm hover:bg-slate-200 transition-colors" target="_blank" rel="noopener">
                            <i class="${icons[type]} mr-1"></i>
                            ${type.charAt(0).toUpperCase() + type.slice(1)}
                        </a>
                    `;
                }
            }).join('');


            // stream tags (max 3)
            const streamTags = Array.isArray(pub.streams) ? pub.streams.slice(0, 4).map(streamId => {
                const metadata = getStreamMetadata(streamId);
                return `<span class="inline-flex items-center px-2 py-1 text-xs font-medium rounded-full stream-tag" style="background-color: ${metadata.color}20; color: ${metadata.color}; border: 1px solid ${metadata.color}40;">
                    <i class="${metadata.icon} mr-1 text-xs"></i>
                    ${metadata.name}
                </span>`;
            }).join('') : '';

            const isAwardPaper = this.checkIfAwardPaper(pub);
            const awardBadge = isAwardPaper ? `
                <div class="award-inline px-2 py-1 text-xs font-medium rounded ml-3 mt-2 flex-shrink-0">
                    <i class="fas fa-trophy text-yellow-600"></i>
                    ${isAwardPaper}
                </div>
            ` : '';

            const card = document.createElement('div');
            card.className = 'publication-card bg-white rounded-xl shadow-sm hover:shadow-lg border border-slate-200 overflow-hidden';
            card.innerHTML = `
                <div class="p-6">
                    <div class="flex justify-between items-start mb-2">
                        <div>
                            <h3 class="text-lg font-semibold text-slate-900 mb-2 leading-tight">${pub.title}</h3>
                            <p class="text-sm text-slate-600">${pub.authors}</p>
                        </div>
                        <div>
                            <div class="px-2 py-1 text-xs font-medium rounded ${typeColors[pub.type]} ml-3 flex-shrink-0">
                                ${pub.venue}&nbsp;•&nbsp;${pub.year}
                            </div>
                            ${awardBadge}
                        </div>
                    </div>

                    <div class="prose text-sm text-slate-700 mb-4 leading-relaxed abstract-preview">
                        ${pub.abstract}
                    </div>

                    <div class="flex flex-wrap gap-2 mb-4">
                        ${streamTags}
                        ${Array.isArray(pub.streams) && pub.streams.length > 4 ? `<span class="px-2 py-1 text-xs text-slate-500 bg-slate-100 rounded">+${pub.streams.length - 4} more</span>` : ''}
                    </div>

                    <div class="flex justify-between items-center">
                        <div class="flex gap-2">
                            ${links}
                        </div>
                        <button class="text-sardine-blue hover:text-sardine-light text-sm font-medium expand-btn">
                            <i class="fas fa-chevron-down mr-1"></i>More
                        </button>
                    </div>

                    <div class="bibtex-container hidden mt-4">
                        <div class="bg-slate-50 border border-slate-200 rounded-lg p-3">
                            <div class="flex items-center justify-between mb-2">
                                <button class="copy-bibtex-btn text-xs text-sardine-blue hover:text-sardine-light">
                                    <i class="fas fa-copy mr-1"></i>Copy
                                </button>
                            </div>
                            <textarea class="bibtex-content w-full h-32 text-xs font-mono bg-white border border-slate-300 rounded p-2 resize-none" readonly></textarea>
                        </div>
                    </div>

                </div>
            `;
            container.appendChild(card);
        });

        // Add bibtex popup functionality
        setTimeout(() => {
            document.querySelectorAll('.bibtex-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const card = this.closest('.publication-card');
                    const container = card.querySelector('.bibtex-container');
                    const textarea = container.querySelector('.bibtex-content');
                    const bibtexData = decodeURIComponent(this.getAttribute('data-bibtex'));
                    
                    if (container.classList.contains('hidden')) {
                        // Show bibtex container
                        container.classList.remove('hidden');
                        this.innerHTML = '<i class="fas fa-quote-right mr-1"></i>Hide BibTeX';
                        textarea.value = bibtexData;
                    } else {
                        // Hide bibtex container
                        container.classList.add('hidden');
                        this.innerHTML = '<i class="fas fa-quote-right mr-1"></i>BibTeX';
                    }
                });
            });
            
            // Copy bibtex functionality
            document.querySelectorAll('.copy-bibtex-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const textarea = this.closest('.bibtex-container').querySelector('.bibtex-content');
                    textarea.select();
                    document.execCommand('copy');
                    
                    // Visual feedback
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-check mr-1"></i>Copied!';
                    setTimeout(() => {
                        this.innerHTML = originalText;
                    }, 2000);
                });
            });
        }, 100);


        // Add expand/collapse functionality
        setTimeout(() => {
            document.querySelectorAll('.expand-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const card = this.closest('.publication-card');
                    const abstract = card.querySelector('.abstract-preview');
                    const icon = this.querySelector('i');
                    
                    if (abstract.classList.contains('abstract-full')) {
                        abstract.classList.remove('abstract-full');
                        icon.className = 'fas fa-chevron-down mr-1';
                        this.innerHTML = '<i class="fas fa-chevron-down mr-1"></i>More';
                        card.classList.remove('expanded');
                    } else {
                        abstract.classList.add('abstract-full');
                        icon.className = 'fas fa-chevron-up mr-1';
                        this.innerHTML = '<i class="fas fa-chevron-up mr-1"></i>Less';
                        card.classList.add('expanded');
                    }
                });
            });
        }, 100);

    }

    renderPublications(container, publications) {
        container.innerHTML = '';

        publications.forEach(pub => {
            const pubElement = document.createElement('div');
            
            // Check if this is an award-winning paper
            const isAwardPaper = this.checkIfAwardPaper(pub);
            
            // Subtle styling for award papers
            if (isAwardPaper) {
                pubElement.className = 'publication-card award-paper bg-white p-6 rounded-lg shadow-md border border-slate-200 border-l-4 border-amber-300 hover:shadow-xl transition-all duration-300 relative';
            } else {
                pubElement.className = 'publication-card award-paper bg-white p-6 rounded-lg shadow-sm border border-slate-200 hover:shadow-md transition-shadow';
            }
            
            const typeColors = {
                'conference': 'badge-conference',
                'journal': 'badge-journal',
                'preprint': 'badge-preprint',
                'book': 'badge-book'
            };


            const links = Object.entries(pub.links || {}).map(([type, url]) => {
                const icons = {
                    'paper': 'fas fa-file',
                    'code': 'fab fa-github',
                    'demo': 'fas fa-play',
                    'bibtex': 'fas fa-quote-right'
                };
                
                if (type === 'bibtex') {
                    return `
                        <button class="bibtex-btn inline-flex items-center px-3 py-1 bg-slate-100 text-slate-700 rounded text-sm hover:bg-slate-200 transition-colors" data-bibtex="${encodeURIComponent(url)}">
                            <i class="${icons[type]} mr-1"></i>
                            BibTeX
                        </button>
                    `;
                } else {
                    return `
                        <a href="${url}" class="inline-flex items-center px-3 py-1 bg-slate-100 text-slate-700 rounded text-sm hover:bg-slate-200 transition-colors" target="_blank" rel="noopener">
                            <i class="${icons[type]} mr-1"></i>
                            ${type.charAt(0).toUpperCase() + type.slice(1)}
                        </a>
                    `;
                }
            }).join('');
            

            // Generate stream tags with full display
            const streamTags = this.generateStreamTags(pub.streams);

            // Subtle award indicator - just a small badge next to the venue
            let awardBadge = '';
            if (isAwardPaper) {
                awardBadge = `&nbsp;• <span class="inline-flex items-center ml-1 text-amber-700 text-sm">
                    <i class="fas fa-trophy mr-1 text-sm"></i>
                    ${isAwardPaper}
                </span>`;
            }

            pubElement.innerHTML = `
                <div class="flex justify-between items-start mb-4">
                    <div class="flex-1">
                        <h3 class="text-xl font-semibold text-slate-900 mb-2">${pub.title}</h3>
                        <p class="text-slate-700 mb-2">${pub.authors}</p>
                        <div class="flex items-center flex-wrap">
                            <p class="text-slate-600">
                                <strong>${pub.venue}</strong> • ${pub.year}
                                ${awardBadge}
                            </p>
                        </div>
                    </div>
                    <span class="badge ${typeColors[pub.type]} ml-4">
                        ${pub.type.charAt(0).toUpperCase() + pub.type.slice(1)}
                    </span>
                </div>
                
                <div class="text-slate-700 mb-4 leading-relaxed abstract-preview prose">${pub.abstract}</div>
                
                <div class="flex flex-wrap gap-2 mb-4">
                    ${streamTags}
                </div>
                
                <div class="flex justify-between items-center">
                    <div class="flex flex-wrap gap-2">
                        ${links}
                    </div>
                    <button class="text-sardine-blue hover:text-sardine-light text-sm font-medium expand-btn">
                        <i class="fas fa-chevron-down mr-1"></i>More
                    </button>
                </div>

                <div class="bibtex-container hidden mt-4">
                    <div class="bg-slate-50 border border-slate-200 rounded-lg p-3">
                        <div class="flex items-center justify-between mb-2">
                            <button class="copy-bibtex-btn text-xs text-sardine-blue hover:text-sardine-light">
                                <i class="fas fa-copy mr-1"></i>Copy
                            </button>
                        </div>
                        <textarea class="bibtex-content w-full h-32 text-xs font-mono bg-white border border-slate-300 rounded p-2 resize-none" readonly></textarea>
                    </div>
                </div>
            `;
            
            container.appendChild(pubElement);
        });


        // Add bibtex popup functionality
        setTimeout(() => {
            document.querySelectorAll('.bibtex-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const card = this.closest('.publication-card');
                    const container = card.querySelector('.bibtex-container');
                    const textarea = container.querySelector('.bibtex-content');
                    const bibtexData = decodeURIComponent(this.getAttribute('data-bibtex'));
                    
                    if (container.classList.contains('hidden')) {
                        // Show bibtex container
                        container.classList.remove('hidden');
                        this.innerHTML = '<i class="fas fa-quote-right mr-1"></i>Hide BibTeX';
                        textarea.value = bibtexData;
                    } else {
                        // Hide bibtex container
                        container.classList.add('hidden');
                        this.innerHTML = '<i class="fas fa-quote-right mr-1"></i>BibTeX';
                    }
                });
            });
            
            // Copy bibtex functionality
            document.querySelectorAll('.copy-bibtex-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const textarea = this.closest('.bibtex-container').querySelector('.bibtex-content');
                    textarea.select();
                    document.execCommand('copy');
                    
                    // Visual feedback
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-check mr-1"></i>Copied!';
                    setTimeout(() => {
                        this.innerHTML = originalText;
                    }, 2000);
                });
            });
        }, 100);

        // Add expand/collapse functionality
        setTimeout(() => {
            document.querySelectorAll('.expand-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const card = this.closest('.publication-card');
                    const abstract = card.querySelector('.abstract-preview');
                    const icon = this.querySelector('i');
                    
                    if (abstract.classList.contains('abstract-full')) {
                        abstract.classList.remove('abstract-full');
                        icon.className = 'fas fa-chevron-down mr-1';
                        this.innerHTML = '<i class="fas fa-chevron-down mr-1"></i>More';
                        card.classList.remove('expanded');
                    } else {
                        abstract.classList.add('abstract-full');
                        icon.className = 'fas fa-chevron-up mr-1';
                        this.innerHTML = '<i class="fas fa-chevron-up mr-1"></i>Less';
                        card.classList.add('expanded');
                    }
                });
            });
        }, 100);
    }

    generateStreamTags(streams, limit = null) {
        if (!streams || streams.length === 0) return '';
        
        const displayStreams = limit ? streams.slice(0, limit) : streams;
        const streamTags = displayStreams.map(streamId => {
            const metadata = getStreamMetadata(streamId);
            return `<span class="inline-flex items-center px-2 py-1 text-xs font-medium rounded-full" style="background-color: ${metadata.color}20; color: ${metadata.color}; border: 1px solid ${metadata.color}40;">
                <i class="${metadata.icon} mr-1 text-xs"></i>
                ${metadata.name}
            </span>`;
        }).join('');

        // Add "more" indicator if limited
        const moreIndicator = limit && streams.length > limit ? 
            `<span class="px-2 py-1 text-xs text-slate-500 bg-slate-100 rounded-full">+${streams.length - limit} more</span>` : '';

        return streamTags + moreIndicator;
    }

    checkIfAwardPaper(pub) {
        // Check if publication has an award
        return pub.award && pub.award.trim() !== '' ? pub.award.trim() : null;
    }

    renderAllPublications(container, paginationContainer) {
        if (!this.pagination) {
            console.error('Pagination manager not set for PublicationsManager');
            return;
        }

        this.pagination.setTotalItems(this.filteredPublications.length);
        const { startIndex, endIndex } = this.pagination.getCurrentPageItems();
        const pagePublications = this.filteredPublications.slice(startIndex, endIndex);

        this.renderPublications(container, pagePublications);
        this.pagination.render(paginationContainer);
    }

    filterPublications(searchTerm, typeFilter, yearFilter, streams = []) {
        this.filteredPublications = this.publicationsData.filter(pub => {
            const lower = (s) => (s || '').toLowerCase();
            const matchesSearch =
                searchTerm === '' ||
                lower(pub.title).includes(lower(searchTerm)) ||
                lower(pub.authors).includes(lower(searchTerm)) ||
                lower(pub.abstract).includes(lower(searchTerm));

            const matchesType = typeFilter === 'all' || pub.type === typeFilter;
            const matchesYear = yearFilter === 'all' || pub.year.toString() === yearFilter;

            // Streams filter: no streams selected = no restriction.
            const matchesStreams =
                !streams || streams.length === 0 ||
                (Array.isArray(pub.streams) && streams.some(s => pub.streams.includes(s)));

            return matchesSearch && matchesType && matchesYear && matchesStreams;
        });

        this.sortPublications();

        if (this.pagination) {
            this.pagination.currentPage = 1;
        }
    }

    sortPublications() {
        this.filteredPublications.sort((a, b) => {
            let valueA, valueB;
        
            if (this.sortBy === 'date') {
                // Sort by ID (newer publications have higher IDs)
                valueA = a.id || 0;
                valueB = b.id || 0;
            } else if (this.sortBy === 'title' || this.sortBy === 'authors') {
                valueA = a[this.sortBy].toLowerCase();
                valueB = b[this.sortBy].toLowerCase();
            } else {
                // Default to year or other numeric sorting
                valueA = a[this.sortBy];
                valueB = b[this.sortBy];
            }
            
            if (this.sortOrder === 'asc') {
                return valueA > valueB ? 1 : -1;
            } else {
                return valueA < valueB ? 1 : -1;
            }
        });
    }
}

// Publications page functionality
class PublicationsPage {
    constructor() {
        this.publicationsManager = new PublicationsManager(publicationsData);
        this.currentTypeFilter = 'all';
        this.currentYearFilter = 'all';
        this.currentStreams = new Set();
        this.currentSearch = '';
        this.currentSort = 'date-desc';
        this.itemsPerPage = 20;
        
        // Initialize pagination
        this.pagination = new PaginationManager({
            itemsPerPage: this.itemsPerPage,
            onPageChange: (page) => {
                this.renderPublications();
                // Scroll to top of publications list
                document.getElementById('mainPublicationsContainer').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }
        });
        
        this.publicationsManager.setPagination(this.pagination);
        this.init();
    }

    init() {
        this.parseURLParameters();
        this.setupEventListeners();
        this.populateYearFilter();
        this.renderStreamFilters();
        this.updateStatistics();
        this.filterAndRender();
    }

    parseURLParameters() {
        const hash = window.location.hash.substring(1);
        const params = new URLSearchParams(hash);
        const streamsParam = params.get('stream');
        
        if (streamsParam) {
            const streams = streamsParam.split('|').map(s => s.trim()).filter(s => s);
            this.currentStreams = new Set(streams);
        }
    }

    updateURL() {
        const params = new URLSearchParams();
        if (this.currentStreams.size > 0) {
            params.set('stream', Array.from(this.currentStreams).join('|'));
        }
        
        const hash = params.toString();
        const newURL = window.location.pathname + (hash ? '#' + hash : '');
        window.history.replaceState({}, '', newURL);
    }

    renderStreamFilters() {
        const container = document.getElementById('streamFilters');
        const allStreams = getAllStreams();
        
        container.innerHTML = '';
        
        // Add "All Streams" button
        const allButton = document.createElement('button');
        allButton.className = `px-3 py-1 rounded-full text-sm font-medium transition-all ${
            this.currentStreams.size === 0 ? 'bg-sardine-blue text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
        }`;
        allButton.innerHTML = '<i class="fas fa-stream mr-2"></i>All Streams';
        allButton.onclick = () => this.clearStreamFilters();
        container.appendChild(allButton);

        // Add individual stream buttons
        allStreams.forEach(streamId => {
            const metadata = getStreamMetadata(streamId);
            const isActive = this.currentStreams.has(streamId);
            
            const button = document.createElement('button');
            button.className = `px-3 py-1 rounded-full text-sm font-medium transition-all ${
                isActive ? 'active text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`;

            if (isActive) {
                button.style.backgroundColor = metadata.color;
                button.style.borderColor = metadata.color;
            }
            
            button.textContent = metadata.name;
            button.innerHTML = `<i class="${metadata.icon} mr-2"></i>${metadata.name}`;
            button.title = metadata.description;
            button.onclick = () => this.toggleStream(streamId);
            
            container.appendChild(button);
        });
    }

    toggleStream(streamId) {
        if (this.currentStreams.has(streamId)) {
            this.currentStreams.delete(streamId);
        } else {
            this.currentStreams.add(streamId);
        }
        
        this.pagination.currentPage = 1;
        this.renderStreamFilters();
        this.updateURL();
        this.filterAndRender();
    }

    clearStreamFilters() {
        this.currentStreams.clear();
        this.pagination.currentPage = 1;
        this.renderStreamFilters();
        this.updateURL();
        this.filterAndRender();
    }

    setupEventListeners() {
        // Search input with debouncing
        const searchInput = document.getElementById('searchInput');
        const debouncedSearch = SardineWebsite.debounce((value) => {
            this.currentSearch = value;
            this.pagination.currentPage = 1;
            this.filterAndRender();
        }, 300);

        searchInput.addEventListener('input', (e) => {
            debouncedSearch(e.target.value);
        });

        // Type filter dropdown
        const typeFilter = document.getElementById('typeFilter');
        typeFilter.addEventListener('change', (e) => {
            this.currentTypeFilter = e.target.value;
            this.pagination.currentPage = 1;
            this.filterAndRender();
        });

        // Year filter dropdown
        const yearFilter = document.getElementById('yearFilter');
        yearFilter.addEventListener('change', (e) => {
            this.currentYearFilter = e.target.value;
            this.pagination.currentPage = 1;
            this.filterAndRender();
        });

        // Sort dropdown
        const sortSelect = document.getElementById('sortSelect');
        sortSelect.addEventListener('change', (e) => {
            this.currentSort = e.target.value;
            this.pagination.currentPage = 1;
            this.sortAndRender();
        });

        // Items per page dropdown
        const itemsPerPageSelect = document.getElementById('itemsPerPage');
        itemsPerPageSelect.addEventListener('change', (e) => {
            const value = e.target.value;
            this.itemsPerPage = value === 'all' ? this.publicationsManager.filteredPublications.length : parseInt(value);
            this.pagination.setItemsPerPage(this.itemsPerPage);
            this.renderPublications();
        });

        // Reset filters button
        const resetButton = document.getElementById('resetFilters');
        resetButton.addEventListener('click', () => {
            this.resetFilters();
        });
    }

    populateYearFilter() {
        const yearFilter = document.getElementById('yearFilter');
        const years = [...new Set(publicationsData.map(pub => pub.year))].sort((a, b) => b - a);
        
        // Clear existing options except "All Years"
        yearFilter.innerHTML = '<option value="all">All Years</option>';
        
        years.forEach(year => {
            const option = document.createElement('option');
            option.value = year.toString();
            option.textContent = year.toString();
            yearFilter.appendChild(option);
        });
    }

    updateStatistics() {
        const stats = {
            total: publicationsData.length,
            conference: publicationsData.filter(p => p.type === 'conference').length,
            journal: publicationsData.filter(p => p.type === 'journal').length,
            preprint: publicationsData.filter(p => p.type === 'preprint').length,
            filtered: this.publicationsManager.filteredPublications.length,
            stream: getAllStreams().length,
        };

        document.getElementById('totalCount').textContent = stats.total;
        document.getElementById('conferenceCount').textContent = stats.conference;
        document.getElementById('journalCount').textContent = stats.journal;
        document.getElementById('preprintCount').textContent = stats.preprint;
        document.getElementById('streamCount').textContent = stats.stream;
    }

    filterAndRender() {
        // Apply filters
        this.publicationsManager.filterPublications(
            this.currentSearch,
            this.currentTypeFilter,
            this.currentYearFilter,
            Array.from(this.currentStreams)
        );
        
        this.setSortCriteria();
        this.renderPublications();
        this.updateResultsInfo();
        this.updateStatistics();
        this.updatePageInfo();
    }

    sortAndRender() {
        this.setSortCriteria();
        this.publicationsManager.sortPublications();
        this.renderPublications();
    }

    setSortCriteria() {
        const [field, order] = this.currentSort.split('-');
        this.publicationsManager.sortBy = field;
        this.publicationsManager.sortOrder = order;
    }

    updateResultsInfo() {
        const resultsInfo = document.getElementById('resultsInfo');
        const count = this.publicationsManager.filteredPublications.length;
        const total = publicationsData.length;
        
        let message = '';
        if (count === total) {
            message = `Showing all ${total} publications`;
        } else {
            message = `Showing ${count} of ${total} publications`;
        }

        resultsInfo.textContent = message;
    }

    updatePageInfo() {
        const pageInfo = document.getElementById('pageInfo');
        const pageData = this.pagination.getPageInfo();
        
        if (pageData.totalPages <= 1) {
            pageInfo.textContent = '';
        } else {
            pageInfo.textContent = `Page ${pageData.currentPage} of ${pageData.totalPages} (${pageData.startItem}-${pageData.endItem})`;
        }
    }

    renderPublications() {
        const publicationsContainer = document.getElementById('publicationsList');
        const noResults = document.getElementById('noResults');
        const paginationContainer = document.getElementById('paginationContainer');

        if (this.publicationsManager.filteredPublications.length === 0) {
            publicationsContainer.innerHTML = '';
            noResults.classList.remove('hidden');
            paginationContainer.innerHTML = '';
            this.updatePageInfo();
            return;
        }

        noResults.classList.add('hidden');
        this.publicationsManager.renderAllPublications(publicationsContainer, paginationContainer);
        this.updatePageInfo();
    }

    resetFilters() {
        // Reset all filters
        this.currentTypeFilter = 'all';
        this.currentYearFilter = 'all';
        this.currentStreams.clear();
        this.currentSearch = '';
        this.currentSort = 'year-desc';
        this.itemsPerPage = 20;
        this.pagination.currentPage = 1;
        this.pagination.setItemsPerPage(this.itemsPerPage);

        // Reset UI elements
        document.getElementById('typeFilter').value = 'all';
        document.getElementById('yearFilter').value = 'all';
        document.getElementById('searchInput').value = '';
        document.getElementById('sortSelect').value = 'year-desc';
        document.getElementById('itemsPerPage').value = '20';

        // Update URL and render
        this.updateURL();
        this.renderStreamFilters();
        this.filterAndRender();
    }
}