// Team management functionality for SARDINE Lab website

class TeamManager {
    constructor(teamData, countryDatabase) {
        this.teamData = teamData;
        this.countryDatabase = countryDatabase;
    }

    // Get current team organized by categories
    getCurrentTeam() {
        // Fallback implementation
        const current = {};
        Object.keys(this.teamData).forEach(role => {
            if (this.teamData[role] && Array.isArray(this.teamData[role])) {
                current[role] = this.teamData[role].filter(member => 
                    !member.graduation_year || member.graduation_year === 'current'
                );
            }
        });
        return current;
    }

    // Get alumni
    getAlumni() {
        // Fallback implementation
        const alumni = [];
        Object.keys(this.teamData).forEach(role => {
            if (this.teamData[role] && Array.isArray(this.teamData[role])) {
                const graduated = this.teamData[role].filter(member => 
                    member.graduation_year && member.graduation_year !== 'current'
                );
                alumni.push(...graduated);
            }
        });
        return alumni.sort((a, b) => a.name.localeCompare(b.name));
    }

    // Main render function for team section
    renderTeamSection(containerId) {
        const teamContainer = document.getElementById(containerId);
        if (!teamContainer) return;
        
        const currentTeam = this.getCurrentTeam();
        teamContainer.innerHTML = '';

        // Render each team category
        this.renderTeamCategory(teamContainer, 'Faculty Researchers', currentTeam.faculties, 'large');
        this.renderTeamCategory(teamContainer, 'Post-docs', currentTeam.postdocs, 'medium');
        this.renderTeamCategory(teamContainer, 'PhD Students', currentTeam.phds, 'small');
        this.renderTeamCategory(teamContainer, 'Research Collaborators', currentTeam.researchers, 'small');
        this.renderTeamCategory(teamContainer, 'MSc Students', currentTeam.mscs, 'tiny');
        
        // Render alumni
        const alumni = this.getAlumni();
        if (alumni && alumni.length > 0) {
            const alumniSection = this.createAlumniSection('Alumni', alumni);
            teamContainer.appendChild(alumniSection);
        }

        // Initialize world map after team is rendered
        setTimeout(() => this.renderWorldMap(), 100);
    }

    renderTeamCategory(container, title, members, size) {
        if (members && members.length > 0) {
            // Sort members (except faculties)
            if (title !== 'Faculties') {
                members = members.sort((a, b) => a.name.localeCompare(b.name));
            }
            const section = this.createTeamSection(title, members, size);
            container.appendChild(section);
        }
    }

    createTeamSection(title, members, size) {
        const section = document.createElement('div');
        section.innerHTML = `
            <h3 class="text-2xl font-bold text-center mb-8 text-sardine-blue">${title}</h3>
            <div class="grid ${this.getGridClasses(size, members.length)} gap-6" id="${title.toLowerCase().replace(/\s+/g, '-')}-grid">
            </div>
        `;

        const grid = section.querySelector(`#${title.toLowerCase().replace(/\s+/g, '-')}-grid`);
        
        members.forEach(member => {
            const memberCard = this.createMemberCard(member, size);
            grid.appendChild(memberCard);
        });

        return section;
    }

    createAlumniSection(title, alumni) {
        const section = document.createElement('div');
        section.innerHTML = `
            <h3 class="text-2xl font-bold text-center mb-8 text-sardine-blue">${title}</h3>
            <div class="bg-white p-6 rounded-lg shadow-lg border border-slate-200">
                <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm" id="alumni-grid">
                </div>
            </div>
        `;

        const grid = section.querySelector('#alumni-grid');
        
        alumni.forEach(member => {
            const alumniItem = document.createElement('div');
            alumniItem.className = 'flex items-center space-x-3';
            alumniItem.innerHTML = `
                <div class="w-12 h-12 bg-slate-300 rounded-full flex items-center justify-center text-slate-500">
                    <img class="rounded-full" src="${member.image}" alt="${member.name}">
                </div>
                <div>
                    <p class="font-medium"><a href="${member.links.website ? member.links.website : '#'}" class="hover:text-sardine-blue transition-colors" target="_blank" rel="noopener">${member.name}</a></p>
                    <p class="text-xs text-slate-500">${member.previous_position} from ${member.start_year} to ${member.graduation_year}</p>
                    <p class="text-xs text-slate-500">${member.position}</p>
                </div>
            `;
            grid.appendChild(alumniItem);
        });

        return section;
    }

    createMemberCard(member, size) {
        const card = document.createElement('div');
        card.className = 'team-card';

        const sizeClasses = {
            large: { avatar: 'w-32 h-32', name: 'text-lg', spacing: 'p-6' },
            medium: { avatar: 'w-24 h-24', name: 'text-lg', spacing: 'p-6' },
            small: { avatar: 'w-20 h-20', name: 'text-base', spacing: 'p-6' },
            tiny: { avatar: 'w-16 h-16', name: 'text-sm', spacing: 'p-4' }
        };

        const classes = sizeClasses[size];
        
        const links = Object.entries(member.links || {}).map(([type, url]) => {
            const icons = {
                website: 'fas fa-globe',
                github: 'fab fa-github',
                linkedin: 'fab fa-linkedin',
                scholar: 'fas fa-graduation-cap'
            };
            
            return `<a href="${url}" class="text-slate-400 hover:text-sardine-blue transition-colors" target="_blank" rel="noopener">
                <i class="${icons[type]} ${size === 'tiny' ? 'text-xs' : 'text-sm'}"></i>
            </a>`;
        }).join('');

        const advisorInfo = member.advisor ? 
            `<p class="text-xs text-slate-500 mb-2">Advised by ${member.advisor} ${member.co_advisor ? `and ${member.co_advisor}` : ''}</p>` : `<p class="text-xs text-slate-500 mb-2">${member.position}</p>`;

        card.innerHTML = `
            <div class="${classes.avatar} bg-slate-300 rounded-full mx-auto mb-4 flex items-center justify-center text-slate-500">
                <img class="rounded-full" src="${member.image}" alt="${member.name}">
            </div>
            <h4 class="font-semibold ${classes.name} mb-1">${member.name}</h4>
            ${advisorInfo}
            ${member.research_interests ? `<p class="text-sm text-slate-600 mb-3">${member.research_interests[0]}</p>` : ''}
            ${member.current_position ? `<p class="text-sm text-slate-600 mb-3">${member.current_position}</p>` : ''}
            <div class="flex justify-center space-x-2">
                ${links}
            </div>
        `;

        return card;
    }

    getGridClasses(size, count) {
        if (size === 'large') {
            return 'md:grid-cols-2 lg:grid-cols-3';
        } else if (size === 'medium') {
            return 'md:grid-cols-2 lg:grid-cols-4';
        } else if (size === 'small') {
            return 'md:grid-cols-2 lg:grid-cols-4';
        } else {
            return 'md:grid-cols-3 lg:grid-cols-5';
        }
    }

    // World map functionality
    renderWorldMap() {
        const mapImg = document.getElementById('worldMap');
        const pinsWrap = document.getElementById('pins');
        
        if (!mapImg || !pinsWrap || !this.teamData || !this.countryDatabase) {
            console.warn('World map requirements not met:', {
                mapImg: !!mapImg,
                pinsWrap: !!pinsWrap,
                teamData: !!this.teamData,
                countryDatabase: !!this.countryDatabase
            });
            return;
        }

        // Collect all team members with location data
        const allMembers = [];
        Object.keys(this.teamData).forEach(role => {
            if (this.teamData[role] && Array.isArray(this.teamData[role])) {
                this.teamData[role].forEach(member => {
                    if (member && member.country) {
                        allMembers.push({
                            ...member,
                            role: role
                        });
                    }
                });
            }
        });

        // Geographic projection function
        const project = (lat, lon, width, height) => {
            const mapLeft = -170;
            const mapRight = 180;
            const mapTop = 80;
            const mapBottom = -60;
            
            lat = Math.max(mapBottom, Math.min(mapTop, lat));
            lon = Math.max(mapLeft, Math.min(mapRight, lon));

            const x = ((lon - mapLeft) / (mapRight - mapLeft)) * width;
            const y = ((mapTop - lat) / (mapTop - mapBottom)) * height;
            
            return { x, y };
        };

        const placePins = () => {
            if (!mapImg || !pinsWrap) return;
            
            // Clear loading indicator
            pinsWrap.innerHTML = '';
            
            const rect = pinsWrap.getBoundingClientRect();
            const width = rect.width;
            const height = rect.height;

            // Group members by country
            const countryGroups = {};
            allMembers.forEach(member => {
                const country = member.country;
                if (!countryGroups[country]) {
                    countryGroups[country] = {
                        country: country,
                        members: []
                    };
                }
                countryGroups[country].members.push(member);
            });

            // Create tooltip element
            const tooltip = document.createElement('div');
            tooltip.className = 'map-tooltip';
            tooltip.style.display = 'none';
            document.body.appendChild(tooltip);

            // Create pins for each country
            Object.values(countryGroups).forEach((group) => {
                const country = group.country;
                
                // Get coordinates and color from database
                const coordinates = this.countryDatabase.coordinates[country];
                const color = this.countryDatabase.colors[country] || '#ef4444';
                
                if (!coordinates) {
                    console.warn(`No coordinates found for country: ${country}`);
                    return;
                }
                
                const [lat, lon] = coordinates;
                const { x, y } = project(lat, lon, width, height);
                
                if (x < 0 || x > width || y < 0 || y > height) {
                    console.log(`${country} is outside map bounds: x=${x}, y=${y}`);
                    return;
                }
                
                const pin = this.createCountryPin(group, x, y, color, tooltip);
                pinsWrap.appendChild(pin);
            });

            console.log(`World map rendered: ${Object.keys(countryGroups).length} countries with pins`);
        };

        // Initialize pins when map loads
        if (mapImg.complete) {
            setTimeout(placePins, 100);
        } else {
            mapImg.addEventListener('load', () => {
                setTimeout(placePins, 100);
            });
        }

        // Re-place pins on window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(placePins, 250);
        });
    }

    createCountryPin(group, x, y, color, tooltip) {
        const pin = document.createElement('div');
        pin.className = 'absolute';
        pin.style.left = `${x}px`;
        pin.style.top = `${y}px`;
        pin.style.transform = 'translate(-50%, -50%)';
        pin.style.zIndex = '10';
        
        pin.innerHTML = `
            <div class="map-pin" style="background-color: ${color};"></div>
        `;

        // Add hover events
        pin.addEventListener('mouseenter', (e) => {
            this.showTooltip(group, tooltip, e);
        });

        pin.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });
        
        return pin;
    }

    showTooltip(group, tooltip, event) {
        const memberCount = group.members.length;
        
        // Group members by city
        const cityGroups = {};
        group.members.forEach(member => {
            const city = member.city || 'Unknown City';
            if (!cityGroups[city]) {
                cityGroups[city] = [];
            }
            cityGroups[city].push(member);
        });

        // Sort each city's members
        Object.keys(cityGroups).forEach(city => {
            cityGroups[city].sort((a, b) => a.name.localeCompare(b.name));
        });

        // Build tooltip content
        let tooltipContent = `
            <div class="country-name">${group.country} (${memberCount})</div>
            <div class="member-list">
        `;

        Object.keys(cityGroups).forEach(city => {
            cityGroups[city].forEach(member => {
                tooltipContent += `<div class="member-item">${member.name} (${city})</div>`;
            });
        });

        tooltipContent += '</div>';
        
        tooltip.innerHTML = tooltipContent;
        tooltip.style.display = 'block';
        
        // Position tooltip
        this.positionTooltip(tooltip, event);
        
        // Update position on mouse move
        const mouseMoveHandler = (e) => this.positionTooltip(tooltip, e);
        document.addEventListener('mousemove', mouseMoveHandler);
        
        // Clean up on mouse leave
        const mouseLeaveHandler = () => {
            document.removeEventListener('mousemove', mouseMoveHandler);
            document.removeEventListener('mouseleave', mouseLeaveHandler);
        };
        document.addEventListener('mouseleave', mouseLeaveHandler, { once: true });
    }

    positionTooltip(tooltip, event) {
        const tooltipRect = tooltip.getBoundingClientRect();
        let left = event.clientX + 15;
        let top = event.clientY - 10;
        
        // Adjust if tooltip would go off screen
        if (left + tooltipRect.width > window.innerWidth) {
            left = event.clientX - tooltipRect.width - 15;
        }
        if (top + tooltipRect.height > window.innerHeight) {
            top = event.clientY - tooltipRect.height - 10;
        }
        if (left < 0) left = 10;
        if (top < 0) top = 10;
        
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
    }
}

// Photo slider functionality for group photos
class PhotoSliderManager {
    constructor(groupPhotos) {
        this.groupPhotos = groupPhotos;
        this.currentPhotoIndex = 0;
    }

    init() {
        if (!this.groupPhotos) {
            console.warn('Group photos not available.');
            return;
        }

        this.generateThumbnails();
        this.updatePhoto(0, true);
        this.updateCounter();
        this.addSwipeSupport();
        this.setupKeyboardNavigation();
    }

    generateThumbnails() {
        const thumbnailTrack = document.getElementById('thumbnailTrack');
        if (!thumbnailTrack) return;
        
        thumbnailTrack.innerHTML = '';
        
        this.groupPhotos.forEach((photo, index) => {
            const thumbnail = document.createElement('div');
            thumbnail.className = `thumbnail ${index === 0 ? 'active' : ''}`;
            thumbnail.onclick = () => this.updatePhoto(index, false);
            
            thumbnail.innerHTML = `
                <img src="assets/figs/${photo.filename}" alt="Group photo ${photo.year}" loading="lazy">
            `;
            
            thumbnailTrack.appendChild(thumbnail);
        });
    }

    updatePhoto(index, fromInit = false) {
        this.currentPhotoIndex = index;
        const photo = this.groupPhotos[index];
        
        // Update main image
        const mainImg = document.getElementById('currentPhoto');
        if (mainImg) {
            mainImg.src = `assets/figs/${photo.filename}`;
            mainImg.alt = `SARDINE Lab Group Photo - ${photo.year}`;
        }
        
        // Update caption
        const photoDate = document.getElementById('photoDate');
        const photoDescription = document.getElementById('photoDescription');
        if (photoDate) photoDate.textContent = photo.year;
        if (photoDescription) photoDescription.textContent = photo.description;
        
        // Update thumbnails
        document.querySelectorAll('.thumbnail').forEach((thumb, i) => {
            thumb.classList.toggle('active', i === index);
        });
        
        this.updateCounter();

        // Scroll to thumbnail if user action
        if (!fromInit) {
            this.scrollThumbnailIntoView(index);
        }
    }

    changePhoto(direction) {
        const newIndex = this.currentPhotoIndex + direction;
        
        if (newIndex >= 0 && newIndex < this.groupPhotos.length) {
            this.updatePhoto(newIndex, false);
        } else if (newIndex < 0) {
            this.updatePhoto(this.groupPhotos.length - 1, false); // Loop to last
        } else {
            this.updatePhoto(0, false); // Loop to first
        }
    }

    updateCounter() {
        const currentIndexEl = document.getElementById('currentIndex');
        const totalPhotosEl = document.getElementById('totalPhotos');
        if (currentIndexEl) currentIndexEl.textContent = this.currentPhotoIndex + 1;
        if (totalPhotosEl) totalPhotosEl.textContent = this.groupPhotos.length;
    }

    scrollThumbnailIntoView(index) {
        const thumbnailTrack = document.getElementById('thumbnailTrack');
        const thumbnail = thumbnailTrack?.children[index];
        
        if (thumbnail) {
            thumbnail.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest',
                inline: 'center'
            });
        }
    }

    addSwipeSupport() {
        const photoContainer = document.querySelector('.main-photo-container');
        if (!photoContainer) return;
        
        let startX = 0;
        let startY = 0;
        
        photoContainer.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        });
        
        photoContainer.addEventListener('touchend', (e) => {
            const endX = e.changedTouches[0].clientX;
            const endY = e.changedTouches[0].clientY;
            
            const deltaX = startX - endX;
            const deltaY = startY - endY;
            
            // Only trigger if horizontal swipe is more significant than vertical
            if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 50) {
                if (deltaX > 0) {
                    this.changePhoto(1); // Swipe left - next photo
                } else {
                    this.changePhoto(-1); // Swipe right - previous photo
                }
            }
        });
    }

    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                this.changePhoto(-1);
            } else if (e.key === 'ArrowRight') {
                this.changePhoto(1);
            }
        });
    }
}

// Make changePhoto function globally available for HTML onclick events
window.changePhoto = function(direction) {
    if (window.photoSlider) {
        window.photoSlider.changePhoto(direction);
    }
};