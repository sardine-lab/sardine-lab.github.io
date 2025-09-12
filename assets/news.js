// News management functionality for SARDINE Lab website

const parseDMY = (str) => {
  const [d, m, y] = str.split('/').map(Number);
  return new Date(y, m - 1, d);
};


class NewsManager {
    constructor(newsData) {
        this.newsData = newsData;
        this.filteredNews = [...newsData];
        this.pagination = null;
    }

    setPagination(paginationManager) {
        this.pagination = paginationManager;
    }

    // Render latest news with timeline style for index.html
    renderLatestNewsTimeline(container, limit = 4) {
        const latestNews = this.newsData.slice(0, limit);
        container.innerHTML = '';

        // Create timeline container
        const timelineContainer = document.createElement('div');
        timelineContainer.className = 'relative';
        
        // Timeline line
        const timelineLine = document.createElement('div');
        timelineLine.className = 'timeline-line absolute left-8 top-0 w-1 h-full';
        timelineLine.style.background = 'linear-gradient(180deg, transparent 0%, #007a94 20%, #007a94 80%, transparent 100%)';
        timelineContainer.appendChild(timelineLine);

        // Timeline items container
        const itemsContainer = document.createElement('div');
        itemsContainer.className = 'space-y-8';

        latestNews.forEach((news, index) => {
            // Remove images for home view
            let newsNoImages = news.content
                .replace(/!\[.*?\]\(.*?\)/g, '')   // remove markdown images
                .replace(/<img[^>]*>/g, '');       // remove HTML images

            const newsItem = this.createTimelineNewsItem(
                { ...news, content: newsNoImages }, 
                index
            );
            itemsContainer.appendChild(newsItem);
        });

        timelineContainer.appendChild(itemsContainer);
        container.appendChild(timelineContainer);
    }

    createTimelineNewsItem(news, index) {
        const newsItem = document.createElement('div');
        newsItem.className = 'timeline-item relative flex items-start';
        newsItem.style.opacity = '0';
        newsItem.style.transform = 'translateY(20px)';
        newsItem.style.animation = `slideIn 0.6s ease-out forwards`;
        newsItem.style.animationDelay = `${index * 0.1}s`;

        // Type icon and color mapping
        const typeConfig = this.getNewsTypeConfig(news.type);
        
        newsItem.innerHTML = `
            <div class="timeline-icon absolute left-0 w-16 h-16 ${typeConfig.bgColor} rounded-full flex items-center justify-center text-white z-10">
                <i class="${typeConfig.icon} text-xl"></i>
            </div>
            <div class="ml-24 bg-white rounded-xl shadow-md border border-slate-200 p-6 flex-1 hover:shadow-lg transition-shadow">
                <div class="flex items-center gap-2 mb-3">
                    <span class="px-3 py-1 ${typeConfig.badgeColor} text-sm font-medium rounded-full">${typeConfig.label}</span>
                    <span class="text-sm text-slate-500">${SardineWebsite.formatDate(news.date)}</span>
                </div>
                <h3 class="text-xl font-semibold text-slate-900 mb-3">${news.title}</h3>
                <div class="prose text-slate-700 text-sm leading-relaxed mb-4 news-content">
                    ${news.content}
                </div>
                <div class="flex flex-wrap gap-2">
                    ${news.tags.map(tag => 
                        `<span class="px-2 py-1 bg-slate-50 text-slate-600 text-xs rounded-full">#${tag}</span>`
                    ).join('')}
                </div>
            </div>
        `;

        return newsItem;
    }

    getNewsTypeConfig(type) {
        const configs = {
            'award': {
                icon: 'fas fa-trophy',
                bgColor: 'bg-yellow-500',
                badgeColor: 'bg-yellow-100 text-yellow-800',
                label: 'Award'
            },
            'publication': {
                icon: 'fas fa-file-alt',
                bgColor: 'bg-blue-500',
                badgeColor: 'bg-blue-100 text-blue-800',
                label: 'Publication'
            },
            'presentation': {
                icon: 'fas fa-microphone',
                bgColor: 'bg-purple-500',
                badgeColor: 'bg-purple-100 text-purple-800',
                label: 'Presentation'
            },
            'funding': {
                icon: 'fas fa-euro-sign',
                bgColor: 'bg-green-500',
                badgeColor: 'bg-green-100 text-green-800',
                label: 'Funding'
            },
            'team': {
                icon: 'fas fa-users',
                bgColor: 'bg-indigo-500',
                badgeColor: 'bg-indigo-100 text-indigo-800',
                label: 'Team'
            },
            'event': {
                icon: 'fas fa-calendar-alt',
                bgColor: 'bg-pink-500',
                badgeColor: 'bg-pink-100 text-pink-800',
                label: 'Event'
            },
            'release': {
                icon: 'fas fa-rocket',
                bgColor: 'bg-orange-500',
                badgeColor: 'bg-orange-100 text-orange-800',
                label: 'Release'
            },
            'service': {
                icon: 'fas fa-handshake',
                bgColor: 'bg-cyan-500',
                badgeColor: 'bg-cyan-100 text-cyan-800',
                label: 'Service'
            }
        };

        return configs[type] || {
            icon: 'fas fa-info-circle',
            bgColor: 'bg-gray-500',
            badgeColor: 'bg-gray-100 text-gray-800',
            label: 'News'
        };
    }

    // Render regular news list for news.html
    renderAllNews(container, paginationContainer) {
        if (!this.pagination) {
            console.error('Pagination manager not set for NewsManager');
            return;
        }

        this.pagination.setTotalItems(this.filteredNews.length);
        const { startIndex, endIndex } = this.pagination.getCurrentPageItems();
        const pageNews = this.filteredNews.slice(startIndex, endIndex);

        container.innerHTML = '';

        pageNews.forEach(news => {
            const newsItem = document.createElement('article');
            newsItem.className = 'mb-12 pb-8 border-b border-slate-200 last:border-b-0';
            
            // Type icon mapping
            const typeIcons = {
                'presentation': 'fas fa-microphone',
                'publication': 'fas fa-file-alt',
                'award': 'fas fa-trophy',
                'funding': 'fas fa-euro-sign',
                'event': 'fas fa-calendar-alt',
                'team': 'fas fa-users',
                'release': 'fas fa-rocket',
                'service': 'fas fa-handshake'
            };
            
            newsItem.innerHTML = `
                <!-- Header -->
                <header class="mb-6">
                    <div class="flex items-center gap-4 text-sm text-slate-500 mb-2">
                        <time class="flex items-center gap-1">
                            <i class="fas fa-calendar text-xs"></i>
                            ${SardineWebsite.formatDate(news.date)}
                        </time>
                        <span class="flex items-center gap-1">
                            <i class="${typeIcons[news.type] || 'fas fa-info-circle'} text-xs"></i>
                            ${news.type.charAt(0).toUpperCase() + news.type.slice(1)}
                        </span>
                    </div>
                    <h3 class="text-2xl font-bold text-slate-900 leading-tight">
                        ${news.title}
                    </h3>
                </header>
                
                <!-- Content -->
                <div class="prose news-content text-slate-700 prose-slate">
                    ${news.content}
                </div>
                
                <!-- Footer -->
                <footer class="mt-2 pt-4">
                    <div class="flex items-center justify-between">
                        <div class="flex flex-wrap gap-2">
                            ${news.tags.map(tag => 
                                `<span class="px-3 py-1 bg-slate-100 text-slate-600 text-sm rounded-full">#${tag}</span>`
                            ).join('')}
                        </div>
                    </div>
                </footer>
            `;

            // Ensure images are responsive
            newsItem.querySelectorAll('.news-content img').forEach(img => {
                img.classList.add('mx-auto', 'rounded-lg', 'shadow-none');
                img.style.maxWidth = '512px';
                img.style.height = 'auto';
                img.style.boxShadow = 'none';
            });

            container.appendChild(newsItem);
        });

        // Render pagination
        this.pagination.render(paginationContainer);
    }

    filterNews(searchTerm) {
        this.filteredNews = this.newsData.filter(news => 
            news.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
            news.content.toLowerCase().includes(searchTerm.toLowerCase())
        );
        
        if (this.pagination) {
            this.pagination.currentPage = 1;
        }
    }
}

// News page functionality
class NewsPage {
    constructor() {
        this.newsManager = new NewsManager(newsData);
        this.currentTypeFilter = 'all';
        this.currentYearFilter = 'all';
        this.currentSearch = '';
        this.currentSort = 'date-desc';
        this.itemsPerPage = 20;
        
        // Initialize pagination
        this.pagination = new PaginationManager({
            itemsPerPage: this.itemsPerPage,
            onPageChange: (page) => {
                this.renderNews();
                // Scroll to top of news list
                document.getElementById('newsList').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }
        });
        
        this.newsManager.setPagination(this.pagination);
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.populateYearFilter();
        this.updateStatistics();
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
            this.itemsPerPage = value === 'all' ? this.newsManager.filteredNews.length : parseInt(value);
            this.pagination.setItemsPerPage(this.itemsPerPage);
            this.renderNews();
        });

        // Reset filters button
        const resetButton = document.getElementById('resetFilters');
        resetButton.addEventListener('click', () => {
            this.resetFilters();
        });
    }

    populateYearFilter() {
        const yearFilter = document.getElementById('yearFilter');
        const years = [...new Set(newsData.map(n => parseDMY(n.date).getFullYear()))].sort((a, b) => b - a);

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
            total: newsData.length,
            presentation: newsData.filter(n => n.type === 'presentation').length,
            publication: newsData.filter(n => n.type === 'publication').length,
            award: newsData.filter(n => n.type === 'award').length,
            filtered: this.newsManager.filteredNews.length
        };
    }

    filterAndRender() {
        // Apply filters
        let filteredNews = newsData;

        // Filter by type
        if (this.currentTypeFilter !== 'all') {
            filteredNews = filteredNews.filter(news => news.type === this.currentTypeFilter);
        }

        // Filter by year
        if (this.currentYearFilter !== 'all') {
            filteredNews = filteredNews.filter(news => 
                new Date(news.date).getFullYear().toString() === this.currentYearFilter
            );
        }

        // Filter by search term
        if (this.currentSearch) {
            const searchTerm = this.currentSearch.toLowerCase();
            filteredNews = filteredNews.filter(news => 
                news.title.toLowerCase().includes(searchTerm) ||
                news.content.toLowerCase().includes(searchTerm) ||
                news.tags.some(tag => tag.toLowerCase().includes(searchTerm))
            );
        }

        this.newsManager.filteredNews = filteredNews;
        this.setSortCriteria();
        this.sortNews();
        this.renderNews();
        this.updateResultsInfo();
        this.updateStatistics();
        this.updatePageInfo();
    }

    sortAndRender() {
        this.setSortCriteria();
        this.sortNews();
        this.renderNews();
    }

    setSortCriteria() {
        const [field, order] = this.currentSort.split('-');
        this.sortBy = field;
        this.sortOrder = order;
    }

    sortNews() {
      this.newsManager.filteredNews.sort((a, b) => {
        let valueA, valueB;

        if (this.sortBy === 'date') {
          valueA = parseDMY(a.date).getTime();
          valueB = parseDMY(b.date).getTime();
        } else if (this.sortBy === 'title') {
          valueA = a.title.toLowerCase();
          valueB = b.title.toLowerCase();
        }

        if (this.sortBy === 'title') {
          return this.sortOrder === 'asc'
            ? valueA.localeCompare(valueB)
            : valueB.localeCompare(valueA);
        } else { // date
          return this.sortOrder === 'asc'
            ? valueA - valueB          // old -> new
            : valueB - valueA;         // new -> old (default)
        }
      });
    }

    updateResultsInfo() {
        const resultsInfo = document.getElementById('resultsInfo');
        const count = this.newsManager.filteredNews.length;
        const total = newsData.length;
        
        let message = '';
        if (count === total) {
            message = `Showing all ${total} news items`;
        } else {
            message = `Showing ${count} of ${total} news items`;
        }

        // Add active filters info
        const activeFilters = [];
        if (this.currentTypeFilter !== 'all') {
            activeFilters.push(`Type: ${this.currentTypeFilter}`);
        }
        if (this.currentYearFilter !== 'all') {
            activeFilters.push(`Year: ${this.currentYearFilter}`);
        }
        if (this.currentSearch) {
            activeFilters.push(`Search: "${this.currentSearch}"`);
        }

        if (activeFilters.length > 0) {
            message += ` (${activeFilters.join(', ')})`;
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

    renderNews() {
        const newsContainer = document.getElementById('newsList');
        const noResults = document.getElementById('noResults');
        const paginationContainer = document.getElementById('paginationContainer');

        if (this.newsManager.filteredNews.length === 0) {
            newsContainer.innerHTML = '';
            noResults.classList.remove('hidden');
            paginationContainer.innerHTML = '';
            this.updatePageInfo();
            return;
        }

        noResults.classList.add('hidden');
        this.newsManager.renderAllNews(newsContainer, paginationContainer);
        this.updatePageInfo();
    }

    resetFilters() {
        // Reset all filters
        this.currentTypeFilter = 'all';
        this.currentYearFilter = 'all';
        this.currentSearch = '';
        this.currentSort = 'date-desc';
        this.itemsPerPage = 20;
        this.pagination.currentPage = 1;
        this.pagination.setItemsPerPage(this.itemsPerPage);

        // Reset UI elements
        document.getElementById('typeFilter').value = 'all';
        document.getElementById('yearFilter').value = 'all';
        document.getElementById('searchInput').value = '';
        document.getElementById('sortSelect').value = 'date-desc';
        document.getElementById('itemsPerPage').value = '20';

        // Re-render
        this.filterAndRender();
    }
}