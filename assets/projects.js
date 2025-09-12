// Projects management functionality for SARDINE Lab website

class ProjectsManager {
    constructor(projectsData) {
        this.projectsData = projectsData;
    }

    getCurrentProjects(limit = null) {
        if (typeof getCurrentProjects === 'function') {
            return getCurrentProjects(limit);
        }
        
        // Fallback implementation
        const currentProjects = this.projectsData.filter(project => 
            project.status === 'current' || !project.status
        );
        
        return limit ? currentProjects.slice(0, limit) : currentProjects;
    }

    getPastProjects() {
        if (typeof getPastProjects === 'function') {
            return getPastProjects();
        }
        
        // Fallback implementation
        return this.projectsData.filter(project => 
            project.status === 'past' || project.status === 'completed' || project.status === 'finished'
        );
    }

    getProjectPublications(project) {
        if (!project.publications || project.publications.length === 0) {
            return [];
        }

        // If publicationsData is available, match by title
        if (typeof publicationsData !== 'undefined') {
            return project.publications.map(pubTitle => {
                // Find matching publication by title (case-insensitive partial match)
                const matchedPub = publicationsData.find(pub => 
                    pub.title.toLowerCase().includes(pubTitle.toLowerCase()) ||
                    pubTitle.toLowerCase().includes(pub.title.toLowerCase())
                );
                
                // If found, return the full publication object, otherwise create a simple object
                if (matchedPub) {
                    return matchedPub;
                } else {
                    // Fallback: create a simple publication object
                    return {
                        title: pubTitle,
                        authors: '',
                        venue: '',
                        year: '',
                        award: '',
                        links: {}
                    };
                }
            }).filter(pub => pub); // Remove any null results
        }

        // Fallback: treat as simple titles
        return project.publications.map(pubTitle => ({
            title: pubTitle,
            authors: '',
            venue: '',
            year: '',
            award: '',
            links: {}
        }));
    }

    createProjectCard(project) {
        const projectCard = document.createElement('div');
        projectCard.className = 'bg-slate-50 p-8 rounded-2xl border border-slate-200 hover:shadow-lg transition-shadow';
        
        const teamMembers = project.team_members ? 
            `<p class="text-sm text-slate-600 mb-3"><strong>Team:</strong> ${project.team_members.join(', ')}</p>` : '';
        
        const collaborators = project.collaborators ? 
            `<p class="text-sm text-slate-600 mb-3"><strong>Collaborators:</strong> ${project.collaborators.join(', ')}</p>` : '';
        
        const website = project.website ? 
            `<a href="${project.website}" target="_blank" rel="noopener" class="inline-flex items-center gap-1 text-sardine-blue hover:underline text-sm font-medium">
                <span>Project Website</span>
                <i class="fas fa-external-link-alt text-xs"></i>
            </a>` : '';

        projectCard.innerHTML = `
            <div class="grid lg:grid-cols-3 gap-6">
                <div class="lg:col-span-2">
                    <div class="flex items-start justify-between mb-4">
                        <div>
                            <h3 class="text-2xl font-bold text-slate-900 mb-2">${project.name}</h3>
                            <h4 class="text-lg text-slate-700 mb-3">${project.title}</h4>
                        </div>
                    </div>
                    <div class="prose text-slate-700 text-sm leading-relaxed mb-4">${project.description}</div>
                    ${website}
                </div>
                <div class="bg-white p-6 rounded-xl border border-slate-200">
                    <h5 class="font-semibold text-slate-900 mb-3">Project Details</h5>
                    <div class="space-y-2 text-sm">
                        <p><strong>Funding:</strong> ${project.funding}</p>
                        <p><strong>Period:</strong> ${project.period}</p>
                        <p><strong>PI:</strong> ${project.pi}</p>
                        ${teamMembers}
                        ${collaborators}
                    </div>
                    <div class="mt-4">
                        <h6 class="font-medium text-slate-700 mb-2">Keywords</h6>
                        <div class="flex flex-wrap gap-1">
                            ${project.keywords.map(keyword => 
                                `<span class="px-2 py-1 bg-sardine-blue/10 text-sardine-blue rounded text-xs">${keyword}</span>`
                            ).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        return projectCard;
    }

    createFullProjectCard(project, status) {
        const projectCard = document.createElement('div');
        projectCard.className = 'bg-white p-8 rounded-2xl border border-slate-200 shadow-sm hover:shadow-lg transition-shadow';
        
        const statusBadge = status === 'current' ? 
            '<span class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">Ongoing</span>' :
            '<span class="bg-slate-100 text-slate-600 px-3 py-1 rounded-full text-sm font-medium">Completed</span>';

        const teamMembers = project.team_members ? 
            `<p class="text-sm text-slate-600 mb-2"><strong>Team:</strong> ${project.team_members.join(', ')}</p>` : '';
        
        const collaborators = project.collaborators ? 
            `<p class="text-sm text-slate-600 mb-2"><strong>Collaborators:</strong> ${project.collaborators.join(', ')}</p>` : '';
        
        const website = project.website ? 
            `<a href="${project.website}" target="_blank" rel="noopener" class="inline-flex items-center gap-1 text-sardine-blue hover:underline font-medium mr-4">
                <i class="fas fa-external-link-alt text-sm"></i>
                <span>Project Website</span>
            </a>` : '';

        // Get related publications by matching project publications with actual publication data
        const relatedPublications = this.getProjectPublications(project);
        const publicationsSection = relatedPublications.length > 0 ? 
            `<div class="mt-6">
                <h6 class="font-medium text-slate-700 mb-3">Publications</h6>
                <div class="space-y-2">
                    ${relatedPublications.map(pub => `
                        <div>
                            <h7 class="font-medium text-slate-800 text-sm mb-1">
                                ${pub.links && pub.links.paper ? 
                                    `<a href="${pub.links.paper}" target="_blank" rel="noopener" class="hover:underline">${pub.title}</a>` : 
                                    pub.title
                                }
                            </h7>
                            <span class="text-xs text-slate-500"> • ${pub.venue} ${pub.year} 
                                ${pub.award ? `• <span class="text-amber-700"><i class="fas fa-trophy text-xs"></i> ${pub.award}</span>` : ''}
                            </span>
                            <p class="text-xs text-slate-500">${pub.authors}</p>
                        </div>
                    `).join('')}
                </div>
            </div>` : '';

        projectCard.innerHTML = `
            <div class="flex items-start justify-between mb-6">
                <div class="flex-1 mr-4">
                    <h3 class="text-2xl font-bold text-slate-900 mb-2">${project.name}</h3>
                    <h4 class="text-lg text-slate-700 mb-4">${project.title}</h4>
                </div>
                ${statusBadge}
            </div>
            
            <div class="grid lg:grid-cols-3 gap-8">
                <div class="lg:col-span-2">
                    <!-- Project Figure -->
                    <div class="mb-6">
                        <div class="w-full h-48 rounded-xl bg-gradient-to-br from-sardine-blue/10 to-sardine-light/20 flex items-center justify-center">
                            
                            ${project.figure ? 
                                `<img class="rounded-xl h-48 w-full rounded-xl" src="assets/figs/${project.figure}" />` 
                            : 
                                `<div class="text-center text-sardine-blue/60">
                                <i class="fas fa-image text-3xl mb-2"></i>
                                <p class="text-sm font-medium">Project Visualization</p>
                                <p class="text-xs">(Placeholder)</p>
                            </div>`
                            }
                            



                        </div>
                    </div>
                    
                    <div class="prose text-slate-700 leading-relaxed mb-6">${project.description}</div>
                    
                    <div class="flex flex-wrap items-center gap-4">
                        ${website}
                    </div>
                </div>
                
                <div class="bg-slate-50 p-6 rounded-xl">
                    <h5 class="font-semibold text-slate-900 mb-4">Project Details</h5>
                    <div class="space-y-3 text-sm">
                        <div>
                            <span class="font-medium text-slate-700">Funding:</span>
                            <p class="text-slate-600">${project.funding}</p>
                        </div>
                        <div>
                            <span class="font-medium text-slate-700">Period:</span>
                            <p class="text-slate-600">${project.period}</p>
                        </div>
                        <div>
                            <span class="font-medium text-slate-700">Principal Investigator:</span>
                            <p class="text-slate-600">${project.pi}</p>
                        </div>
                        ${teamMembers}
                        ${collaborators}
                    </div>
                    
                    <div class="mt-6">
                        <h6 class="font-medium text-slate-700 mb-3">Keywords</h6>
                        <div class="flex flex-wrap gap-2">
                            ${project.keywords.map(keyword => 
                                `<span class="px-2 py-1 bg-sardine-blue/10 text-sardine-blue rounded text-xs font-medium">${keyword}</span>`
                            ).join('')}
                        </div>
                    </div>
                </div>
            </div>

            ${publicationsSection}
        `;
        
        return projectCard;
    }

    renderProjectsList(containerId, projects, status) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';

        if (projects.length === 0) {
            container.innerHTML = `
                <div class="text-center py-12 text-slate-500">
                    <i class="fas fa-folder-open text-4xl mb-4"></i>
                    <p class="text-lg">No ${status} projects found</p>
                    <p class="text-sm mt-2">Check back later for updates!</p>
                </div>
            `;
            return;
        }

        projects.forEach(project => {
            const projectCard = this.createFullProjectCard(project, status);
            container.appendChild(projectCard);
        });
    }

    renderAllProjects() {
        this.renderProjectsList('currentProjectsList', this.getCurrentProjects(), 'current');
        this.renderProjectsList('pastProjectsList', this.getPastProjects(), 'past');
    }
}

// Projects page functionality
class ProjectsPage {
    constructor() {
        this.projectsManager = new ProjectsManager(projectsData);
        this.init();
    }

    init() {
        this.initTabs();
        this.projectsManager.renderAllProjects();
    }

    initTabs() {
        const currentTab = document.getElementById('currentTab');
        const pastTab = document.getElementById('pastTab');
        const currentPanel = document.getElementById('currentProjectsTab');
        const pastPanel = document.getElementById('pastProjectsTab');

        if (!currentTab || !pastTab || !currentPanel || !pastPanel) {
            console.warn('Tab elements not found');
            return;
        }

        const switchToTab = (activeTab, activePanel, inactiveTab, inactivePanel) => {
            // Update tab buttons
            activeTab.classList.add('active');
            activeTab.classList.remove('text-slate-600', 'hover:text-slate-900', 'border-transparent', 'hover:border-slate-300');
            activeTab.classList.add('text-slate-900', 'border-sardine-blue');

            inactiveTab.classList.remove('active');
            inactiveTab.classList.remove('text-slate-900', 'border-sardine-blue');
            inactiveTab.classList.add('text-slate-600', 'hover:text-slate-900', 'border-transparent', 'hover:border-slate-300');

            // Update panels
            activePanel.classList.remove('hidden');
            activePanel.classList.add('active');
            inactivePanel.classList.add('hidden');
            inactivePanel.classList.remove('active');
        };

        currentTab.addEventListener('click', () => {
            switchToTab(currentTab, currentPanel, pastTab, pastPanel);
        });

        pastTab.addEventListener('click', () => {
            switchToTab(pastTab, pastPanel, currentTab, currentPanel);
        });
    }
}