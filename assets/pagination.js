// Pagination utility for SARDINE Lab website

class PaginationManager {
    constructor(options = {}) {
        this.itemsPerPage = options.itemsPerPage || 10;
        this.currentPage = 1;
        this.totalItems = 0;
        this.onPageChange = options.onPageChange || (() => {});
        this.showFirstLast = options.showFirstLast !== false;
        this.showPrevNext = options.showPrevNext !== false;
        this.maxVisiblePages = options.maxVisiblePages || 5;
    }

    setTotalItems(count) {
        this.totalItems = count;
        // Reset to page 1 if current page is beyond available pages
        const totalPages = this.getTotalPages();
        if (this.currentPage > totalPages && totalPages > 0) {
            this.currentPage = 1;
        }
    }

    setItemsPerPage(count) {
        this.itemsPerPage = count;
        // Recalculate current page to maintain position
        const currentFirstItem = (this.currentPage - 1) * this.itemsPerPage + 1;
        this.currentPage = Math.ceil(currentFirstItem / this.itemsPerPage);
        this.currentPage = Math.max(1, Math.min(this.currentPage, this.getTotalPages()));
    }

    getTotalPages() {
        return Math.ceil(this.totalItems / this.itemsPerPage);
    }

    getCurrentPageItems() {
        const startIndex = (this.currentPage - 1) * this.itemsPerPage;
        const endIndex = startIndex + this.itemsPerPage;
        return { startIndex, endIndex };
    }

    goToPage(page) {
        const totalPages = this.getTotalPages();
        const newPage = Math.max(1, Math.min(page, totalPages));
        
        if (newPage !== this.currentPage) {
            this.currentPage = newPage;
            this.onPageChange(this.currentPage);
            return true;
        }
        return false;
    }

    nextPage() {
        return this.goToPage(this.currentPage + 1);
    }

    prevPage() {
        return this.goToPage(this.currentPage - 1);
    }

    firstPage() {
        return this.goToPage(1);
    }

    lastPage() {
        return this.goToPage(this.getTotalPages());
    }

    getPageInfo() {
        const totalPages = this.getTotalPages();
        const { startIndex, endIndex } = this.getCurrentPageItems();
        const startItem = startIndex + 1;
        const endItem = Math.min(endIndex, this.totalItems);

        return {
            currentPage: this.currentPage,
            totalPages,
            startItem,
            endItem,
            totalItems: this.totalItems,
            hasNextPage: this.currentPage < totalPages,
            hasPrevPage: this.currentPage > 1
        };
    }

    render(container) {
        if (!container) return;

        const totalPages = this.getTotalPages();
        
        if (totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        // Create pagination wrapper
        const paginationWrapper = document.createElement('div');
        paginationWrapper.className = 'flex items-center justify-center gap-2 flex-wrap';

        // Previous button
        if (this.showPrevNext) {
            const prevBtn = this.createButton(
                '<i class="fas fa-chevron-left mr-1"></i>Previous',
                () => this.prevPage(),
                this.currentPage === 1
            );
            paginationWrapper.appendChild(prevBtn);
        }

        // Page numbers
        this.renderPageNumbers(paginationWrapper, totalPages);

        // Next button
        if (this.showPrevNext) {
            const nextBtn = this.createButton(
                'Next<i class="fas fa-chevron-right ml-1"></i>',
                () => this.nextPage(),
                this.currentPage === totalPages
            );
            paginationWrapper.appendChild(nextBtn);
        }

        container.innerHTML = '';
        container.appendChild(paginationWrapper);
    }

    renderPageNumbers(wrapper, totalPages) {
        const visibleRange = this.calculateVisibleRange(totalPages);
        
        // First page if not in visible range
        if (this.showFirstLast && visibleRange.start > 1) {
            const firstBtn = this.createButton('1', () => this.goToPage(1));
            wrapper.appendChild(firstBtn);

            // Ellipsis if there's a gap
            if (visibleRange.start > 2) {
                const ellipsis = this.createEllipsis();
                wrapper.appendChild(ellipsis);
            }
        }

        // Visible page numbers
        for (let i = visibleRange.start; i <= visibleRange.end; i++) {
            const pageBtn = this.createButton(
                i.toString(),
                () => this.goToPage(i),
                false,
                i === this.currentPage
            );
            wrapper.appendChild(pageBtn);
        }

        // Last page if not in visible range
        if (this.showFirstLast && visibleRange.end < totalPages) {
            // Ellipsis if there's a gap
            if (visibleRange.end < totalPages - 1) {
                const ellipsis = this.createEllipsis();
                wrapper.appendChild(ellipsis);
            }

            const lastBtn = this.createButton(
                totalPages.toString(),
                () => this.goToPage(totalPages)
            );
            wrapper.appendChild(lastBtn);
        }
    }

    calculateVisibleRange(totalPages) {
        const halfVisible = Math.floor(this.maxVisiblePages / 2);
        let start = Math.max(1, this.currentPage - halfVisible);
        let end = Math.min(totalPages, start + this.maxVisiblePages - 1);
        
        // Adjust start if we're near the end
        if (end === totalPages) {
            start = Math.max(1, end - this.maxVisiblePages + 1);
        }

        return { start, end };
    }

    createButton(content, onClick, disabled = false, active = false) {
        const button = document.createElement('button');
        button.className = `pagination-btn ${active ? 'active' : ''} ${disabled ? 'disabled' : ''}`;
        button.innerHTML = content;
        
        if (disabled || active) {
            button.disabled = true;
            button.style.cursor = disabled ? 'not-allowed' : 'default';
        } else {
            button.onclick = onClick;
        }
        
        return button;
    }

    createEllipsis() {
        const ellipsis = document.createElement('span');
        ellipsis.className = 'pagination-btn disabled';
        ellipsis.textContent = '...';
        ellipsis.style.cursor = 'default';
        return ellipsis;
    }
}

// Utility function for creating pagination instances
function createPagination(options) {
    return new PaginationManager(options);
}

// Legacy compatibility function (matches the old SardineWebsite.createPagination)
function createPaginationLegacy(container, currentPage, totalPages, onPageChange) {
    const pagination = new PaginationManager({
        onPageChange: onPageChange
    });
    
    pagination.currentPage = currentPage;
    pagination.setTotalItems(totalPages * pagination.itemsPerPage); // Approximate
    pagination.render(container);
}

// Export for module usage (if using modules)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PaginationManager, createPagination, createPaginationLegacy };
}