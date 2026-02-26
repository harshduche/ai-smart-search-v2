// Visual Search Engine - Frontend JavaScript

const API_BASE = '';  // Same origin

// State
let currentTab = 'text';
let selectedImage = null;
let multimodalImage = null;
let ingestVideoFile = null;

// DOM Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const resultsGrid = document.getElementById('results-grid');
const resultsCount = document.getElementById('results-count');
const searchTime = document.getElementById('search-time');
const loadingOverlay = document.getElementById('loading-overlay');
const modal = document.getElementById('video-modal');
const modalOpenVideoBtn = document.getElementById('modal-open-video-btn');
const modalVideoHint = document.getElementById('modal-video-hint');
const ingestModal = document.getElementById('ingest-modal');
const globalProgress = document.getElementById('global-upload-progress');
const globalProgressText = document.getElementById('global-progress-text');
const globalProgressEta = document.getElementById('global-progress-eta');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSearchForms();
    initImageUpload();
    initIngestUpload();
    initIngestModal();
    initRTSPIngest();
    initModal();
    initExampleQueries();
    checkModelStatus();
});

// Tab Navigation
function initTabs() {
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;

            // Update active states
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`${tab}-tab`).classList.add('active');
            currentTab = tab;
        });
    });
}

// Search Forms
function initSearchForms() {
    // Text Search
    document.getElementById('text-search-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = document.getElementById('text-query').value;
        await performTextSearch(query);
    });

    // Image Search
    document.getElementById('image-search-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log('Image search form submitted', { selectedImage });
        if (selectedImage) {
            await performImageSearch(selectedImage);
        } else {
            console.warn('No image selected for image search');
        }
    });

    // Multimodal Search
    document.getElementById('multimodal-search-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = document.getElementById('multimodal-query').value;
        console.log('Multimodal search form submitted', { multimodalImage, query });
        if (multimodalImage && query) {
            await performMultimodalSearch(query, multimodalImage);
        } else {
            console.warn('Missing image or query for multimodal search', { multimodalImage, query });
        }
    });

    // OCR Search
    document.getElementById('ocr-search-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = document.getElementById('ocr-query').value;
        await performOCRSearch(query);
    });

    // Ingest Video
    const ingestForm = document.getElementById('ingest-video-form');
    if (ingestForm) {
        ingestForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await performVideoIngest();
        });
    }
}

// Image Upload
function initImageUpload() {
    // Image search upload
    const imageUploadArea = document.getElementById('image-upload-area');
    const imageFile = document.getElementById('image-file');
    const imagePreview = document.getElementById('image-preview');
    const imageSearchBtn = document.getElementById('image-search-btn');

    console.log('Image upload initialized', { imageUploadArea, imageFile, imagePreview, imageSearchBtn });

    setupUpload(imageUploadArea, imageFile, imagePreview, (file) => {
        console.log('Image selected:', file.name, file.type, file.size);
        selectedImage = file;
        imageSearchBtn.disabled = false;
        console.log('Image search button enabled');
    });

    // Multimodal upload
    const multimodalUploadArea = document.getElementById('multimodal-upload-area');
    const multimodalFile = document.getElementById('multimodal-file');
    const multimodalPreview = document.getElementById('multimodal-preview');
    const multimodalSearchBtn = document.getElementById('multimodal-search-btn');

    setupUpload(multimodalUploadArea, multimodalFile, multimodalPreview, (file) => {
        multimodalImage = file;
        updateMultimodalBtn();
    });

    document.getElementById('multimodal-query').addEventListener('input', updateMultimodalBtn);
}

// Ingest video upload
function initIngestUpload() {
    const ingestUploadArea = document.getElementById('ingest-upload-area');
    const ingestFileInput = document.getElementById('ingest-video-file');
    const ingestInfo = document.getElementById('ingest-video-info');
    const ingestName = document.getElementById('ingest-video-name');
    const ingestSubmitBtn = document.getElementById('ingest-submit-btn');
    const ingestModeSelect = document.getElementById('ingest-mode');
    const semanticOptionsRow = document.getElementById('semantic-options-row');

    if (!ingestUploadArea || !ingestFileInput) return;

    const onSelect = (file) => {
        ingestVideoFile = file;
        if (ingestInfo && ingestName) {
            ingestInfo.hidden = false;
            ingestName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
        }
        if (ingestSubmitBtn) {
            ingestSubmitBtn.disabled = false;
        }
    };

    // For video, we don't need a visual preview; reuse setup pattern without preview
    ingestUploadArea.addEventListener('click', () => ingestFileInput.click());

    ingestUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        ingestUploadArea.style.borderColor = 'var(--primary)';
    });

    ingestUploadArea.addEventListener('dragleave', () => {
        ingestUploadArea.style.borderColor = '';
    });

    ingestUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        ingestUploadArea.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file && file.type && file.type.startsWith('video/')) {
            onSelect(file);
        } else {
            showError('Please upload a valid video file.');
        }
    });

    ingestFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            if (!file.type || !file.type.startsWith('video/')) {
                showError('Please upload a valid video file.');
                return;
            }
            onSelect(file);
        }
    });

    if (ingestModeSelect && semanticOptionsRow) {
        ingestModeSelect.addEventListener('change', () => {
            const mode = ingestModeSelect.value;
            semanticOptionsRow.style.display = mode === 'semantic' ? '' : 'none';
        });
    }
}

// Ingest modal open/close
function initIngestModal() {
    const openBtn = document.getElementById('open-ingest-modal');
    const closeBtn = ingestModal ? ingestModal.querySelector('.ingest-close') : null;

    if (openBtn && ingestModal) {
        openBtn.addEventListener('click', () => {
            ingestModal.classList.add('active');
        });
    }

    if (closeBtn && ingestModal) {
        closeBtn.addEventListener('click', () => {
            ingestModal.classList.remove('active');
        });
    }

    if (ingestModal) {
        ingestModal.addEventListener('click', (e) => {
            if (e.target === ingestModal) {
                ingestModal.classList.remove('active');
            }
        });
    }
}

// RTSP Stream ingestion
function initRTSPIngest() {
    const openBtn = document.getElementById('open-rtsp-modal');
    const modal = document.getElementById('rtsp-modal');
    const closeBtn = modal ? modal.querySelector('.close') : null;
    const form = document.getElementById('rtsp-form');

    if (!openBtn || !modal || !form) return;

    // Open modal
    openBtn.addEventListener('click', () => {
        modal.style.display = 'block';
    });

    // Close modal
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }

    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // Submit form
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await performRTSPIngest();
    });
}

function setupUpload(area, input, preview, onSelect) {
    area.addEventListener('click', () => input.click());

    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.style.borderColor = 'var(--primary)';
    });

    area.addEventListener('dragleave', () => {
        area.style.borderColor = '';
    });

    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageSelect(file, area, preview, onSelect);
        }
    });

    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageSelect(file, area, preview, onSelect);
        }
    });
}

function handleImageSelect(file, area, preview, onSelect) {
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.hidden = false;
        area.querySelector('.upload-placeholder').hidden = true;
        area.classList.add('has-image');
        onSelect(file);
    };
    reader.readAsDataURL(file);
}

function updateMultimodalBtn() {
    const btn = document.getElementById('multimodal-search-btn');
    const query = document.getElementById('multimodal-query').value;
    btn.disabled = !multimodalImage || !query.trim();
}

// Example Queries
function initExampleQueries() {
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.dataset.query;
            if (btn.classList.contains('ocr')) {
                document.getElementById('ocr-query').value = query;
            } else {
                document.getElementById('text-query').value = query;
            }
        });
    });
}

// Search Functions
async function performTextSearch(query) {
    const organizationId = getOrganizationId();
    if (!organizationId) return;
    const filters = getFilters();
    const topK = parseInt(document.getElementById('filter-results').value);
    const useReranker = getUseReranker();

    showLoading();
    try {
        const response = await fetch(`${API_BASE}/search/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                organization_id: organizationId,
                query,
                top_k: topK,
                filters: filters,
                use_reranker: useReranker
            })
        });

        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Search error:', error);
        showError('Search failed. Please try again.');
    } finally {
        hideLoading();
    }
}

async function performImageSearch(imageFile) {
    const organizationId = getOrganizationId();
    if (!organizationId) return;
    console.log('performImageSearch called with:', imageFile);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('organization_id', organizationId);
    formData.append('top_k', document.getElementById('filter-results').value);
    formData.append('use_reranker', getUseReranker());

    const filters = getFilters();
    if (filters) {
        if (filters.zone) formData.append('zone', filters.zone);
        if (filters.is_night !== undefined) formData.append('is_night', filters.is_night);
    }

    console.log('Sending image search request to:', `${API_BASE}/search/image`);
    console.log('FormData entries:', Array.from(formData.entries()));

    showLoading();
    try {
        const response = await fetch(`${API_BASE}/search/image`, {
            method: 'POST',
            body: formData
        });

        console.log('Image search response status:', response.status);
        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        console.log('Image search results:', data);
        displayResults(data);
    } catch (error) {
        console.error('Search error:', error);
        showError('Search failed. Please try again.');
    } finally {
        hideLoading();
    }
}

async function performMultimodalSearch(query, imageFile) {
    const organizationId = getOrganizationId();
    if (!organizationId) return;
    console.log('performMultimodalSearch called with:', { query, imageFile });

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('organization_id', organizationId);
    formData.append('query', query);
    formData.append('top_k', document.getElementById('filter-results').value);
    formData.append('use_reranker', getUseReranker());

    const filters = getFilters();
    if (filters) {
        if (filters.zone) formData.append('zone', filters.zone);
        if (filters.is_night !== undefined) formData.append('is_night', filters.is_night);
    }

    console.log('Sending multimodal search request to:', `${API_BASE}/search/multimodal`);
    console.log('FormData entries:', Array.from(formData.entries()).map(([k, v]) => [k, typeof v === 'object' ? v.name : v]));

    showLoading();
    try {
        const response = await fetch(`${API_BASE}/search/multimodal`, {
            method: 'POST',
            body: formData
        });

        console.log('Multimodal search response status:', response.status);
        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        console.log('Multimodal search results:', data);
        displayResults(data);
    } catch (error) {
        console.error('Search error:', error);
        showError('Search failed. Please try again.');
    } finally {
        hideLoading();
    }
}

async function performOCRSearch(text) {
    const organizationId = getOrganizationId();
    if (!organizationId) return;
    const filters = getFilters();
    const topK = parseInt(document.getElementById('filter-results').value);
    const useReranker = getUseReranker();

    showLoading();
    try {
        const response = await fetch(`${API_BASE}/search/ocr`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                organization_id: organizationId,
                text,
                top_k: topK,
                filters: filters,
                use_reranker: useReranker
            })
        });

        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Search error:', error);
        showError('Search failed. Please try again.');
    } finally {
        hideLoading();
    }
}

async function performVideoIngest() {
    const statusEl = document.getElementById('ingest-status');
    const ingestSubmitBtn = document.getElementById('ingest-submit-btn');

    if (! ingestVideoFile) {
        showError('Please select a video file to ingest.');
        return;
    }

    const zone = document.getElementById('ingest-zone').value.trim();
    const mode = document.getElementById('ingest-mode').value;
    const clipDuration = parseFloat(document.getElementById('clip-duration').value || '4.0');
    const maxFramesPerClip = parseInt(document.getElementById('max-frames-per-clip').value || '32', 10);

    const formData = new FormData();
    formData.append('file', ingestVideoFile);
    if (zone) formData.append('zone', zone);
    formData.append('semantic_video', mode === 'semantic');
    if (mode === 'semantic') {
        formData.append('clip_duration', clipDuration);
        formData.append('max_frames_per_clip', maxFramesPerClip);
    }

    if (statusEl) {
        statusEl.textContent = 'Uploading and ingesting video...';
        statusEl.classList.remove('error');
    }

    if (ingestSubmitBtn) ingestSubmitBtn.disabled = true;

    // Reset global progress bar
    if (globalProgress && globalProgressText && globalProgressEta) {
        globalProgress.hidden = false;
        setGlobalProgress(0, 'Uploading video...', null);
    }

    try {
        // Use XMLHttpRequest to get reliable upload progress events
        await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            const startTime = Date.now();

            xhr.open('POST', `${API_BASE}/ingest/upload`);

            xhr.upload.onprogress = (event) => {
                if (!event.lengthComputable) return;
                const percent = Math.round((event.loaded / event.total) * 100);

                // Estimate remaining time based on upload speed
                const elapsedMs = Date.now() - startTime;
                let etaSeconds = null;
                if (elapsedMs > 0 && percent > 0 && percent < 100) {
                    const totalSeconds = (elapsedMs / 1000) / (percent / 100);
                    etaSeconds = Math.max(0, totalSeconds - elapsedMs / 1000);
                }

                setGlobalProgress(
                    percent,
                    percent < 100 ? 'Uploading video...' : 'Upload complete, ingesting...',
                    etaSeconds
                );
            };

            xhr.onreadystatechange = () => {
                if (xhr.readyState === XMLHttpRequest.DONE) {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const data = JSON.parse(xhr.responseText);
                            if (statusEl) {
                                statusEl.textContent = `Ingestion completed: ` +
                                    `${data.videos_processed} video(s), ` +
                                    `${data.embeddings_generated} embeddings, ` +
                                    `${data.vectors_stored} vectors.`;
                            }
                            setGlobalProgress(100, 'Ingestion completed', 0);
                            resolve();
                        } catch (e) {
                            reject(e);
                        }
                    } else {
                        reject(new Error(xhr.responseText || `HTTP ${xhr.status}`));
                    }
                }
            };

            xhr.onerror = () => reject(new Error('Network error during upload'));

            xhr.send(formData);
        });
    } catch (error) {
        console.error('Video ingest error:', error);
        if (statusEl) {
            statusEl.textContent = 'Ingestion failed. See console for details.';
            statusEl.classList.add('error');
        }
    } finally {
        if (ingestSubmitBtn) ingestSubmitBtn.disabled = false;
        // Keep global progress visible briefly, then hide
        if (globalProgress) {
            setTimeout(() => {
                globalProgress.hidden = true;
            }, 3000);
        }
    }
}

async function performRTSPIngest() {
    const statusDiv = document.getElementById('rtsp-status');

    // Collect form data
    const rtspUrl = document.getElementById('rtsp-url').value.trim();
    const zone = document.getElementById('rtsp-zone').value.trim();
    const durationSeconds = document.getElementById('rtsp-duration').value;
    const maxFrames = document.getElementById('rtsp-max-frames').value;
    const clipDuration = parseFloat(document.getElementById('rtsp-clip-duration').value || '4.0');
    const maxFramesPerClip = parseInt(document.getElementById('rtsp-max-frames-per-clip').value || '32', 10);
    const cameraId = document.getElementById('rtsp-camera-id').value.trim();
    const siteId = document.getElementById('rtsp-site-id').value.trim();

    // Validation: at least one termination condition
    if (!durationSeconds && !maxFrames) {
        statusDiv.innerHTML = '<span style="color: #f87171;">⚠️ Please specify either duration or max frames</span>';
        return;
    }

    if (!rtspUrl || !rtspUrl.startsWith('rtsp://')) {
        statusDiv.innerHTML = '<span style="color: #f87171;">⚠️ Invalid RTSP URL (must start with rtsp://)</span>';
        return;
    }

    // Build request payload
    const payload = {
        rtsp_url: rtspUrl,
        zone: zone || null,
        duration_seconds: durationSeconds ? parseFloat(durationSeconds) : null,
        max_frames: maxFrames ? parseInt(maxFrames, 10) : null,
        use_semantic_clips: true,  // Always true for RTSP
        clip_duration: clipDuration,
        max_frames_per_clip: maxFramesPerClip,
        reconnect_on_failure: true,
        camera_id: cameraId || null,
        site_id: siteId || null,
    };

    // Show loading state
    statusDiv.innerHTML = '<span style="color: #60a5fa;">🔄 Connecting to RTSP stream...</span>';
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/ingest/rtsp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Ingestion failed');
        }

        // Success
        statusDiv.innerHTML = `
            <span style="color: #4ade80;">
                ✅ Stream ingestion complete!<br>
                Frames captured: ${result.frames_extracted}<br>
                Clips ingested: ${result.embeddings_generated}<br>
                Duration: ${result.duration_seconds?.toFixed(1)}s
            </span>
        `;

        // Clear form
        document.getElementById('rtsp-form').reset();

    } catch (error) {
        statusDiv.innerHTML = `<span style="color: #f87171;">❌ Error: ${error.message}</span>`;
    } finally {
        setLoading(false);
    }
}

function setGlobalProgress(percent, message, etaSeconds) {
    if (!globalProgress || !globalProgressText || !globalProgressEta) return;

    globalProgress.hidden = false;
    const bar = globalProgress.querySelector('.global-progress-bar');
    if (bar) {
        const inner = bar.querySelector('::after');
        // We can't manipulate ::after directly; instead, set width via CSS variable if desired.
        bar.style.setProperty('--progress', `${percent}%`);
        // Fallback: set width via inline style on bar itself
        bar.style.background = `linear-gradient(90deg, var(--primary) ${percent}%, var(--bg-card) ${percent}%)`;
    }

    globalProgressText.textContent = message || 'Uploading...';

    if (typeof etaSeconds === 'number' && !Number.isNaN(etaSeconds) && etaSeconds > 0) {
        const rounded = Math.ceil(etaSeconds);
        globalProgressEta.textContent = `~${rounded}s remaining`;
    } else if (percent >= 100) {
        globalProgressEta.textContent = '';
    } else {
        globalProgressEta.textContent = '';
    }
}

// Get organization ID
function getOrganizationId() {
    const orgId = document.getElementById('filter-org').value.trim();
    if (!orgId) {
        showError('Please enter an Organization ID in the filters section.');
        hideLoading();
        return null;
    }
    return orgId;
}

// Get current filter values
function getFilters() {
    const zone = document.getElementById('filter-zone').value;
    const time = document.getElementById('filter-time').value;

    const filters = {};
    if (zone) filters.zone = zone;
    if (time === 'night') filters.is_night = true;
    if (time === 'day') filters.is_night = false;

    return Object.keys(filters).length > 0 ? filters : null;
}

// Get reranker setting
function getUseReranker() {
    return document.getElementById('use-reranker').checked;
}

// Display Results
function displayResults(data) {
    resultsCount.textContent = `${data.total_results} results`;
    searchTime.textContent = `${data.search_time_ms.toFixed(0)}ms`;

    // Debug: Log first result to see what fields are available
    if (data.results.length > 0) {
        console.log('First search result:', data.results[0]);
    }

    if (data.results.length === 0) {
        resultsGrid.innerHTML = `
            <div class="no-results">
                <p>No results found. Try a different query or adjust filters.</p>
            </div>
        `;
        return;
    }

    resultsGrid.innerHTML = data.results.map((result, index) => {
        const isClip = result.source_type === 'video_clip' ||
            (typeof result.clip_start_seconds === 'number' && typeof result.clip_end_seconds === 'number');

        let primaryLabel;
        let secondaryLabelParts = [];

        if (isClip) {
            const clipIndex = typeof result.clip_index === 'number'
                ? result.clip_index
                : result.frame_number;

            const startSec = typeof result.clip_start_seconds === 'number'
                ? result.clip_start_seconds
                : result.seconds_offset;
            const endSec = typeof result.clip_end_seconds === 'number'
                ? result.clip_end_seconds
                : undefined;

            let duration = null;
            if (typeof startSec === 'number' && typeof endSec === 'number') {
                duration = Math.max(0, endSec - startSec);
            }

            primaryLabel = `Clip ${clipIndex}`;

            if (duration !== null && !Number.isNaN(duration)) {
                secondaryLabelParts.push(`${duration.toFixed(1)}s`);
            }
            if (typeof result.num_frames === 'number') {
                secondaryLabelParts.push(`${result.num_frames} frames`);
            }
            secondaryLabelParts.push(result.zone || 'Unknown');
        } else {
            primaryLabel = `Frame ${result.frame_number}`;
            secondaryLabelParts.push(result.zone || 'Unknown');
        }

        const secondaryLabel = secondaryLabelParts.filter(Boolean).join(' · ');

        return `
        <div class="result-card" data-index="${index}" onclick="showResultDetail(${index})">
            <div class="result-thumbnail">
                <img src="${getThumbnailUrl(result.thumbnail_path)}"
                     alt="${isClip ? primaryLabel : `Frame ${result.frame_number}`}"
                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%23334155%22 width=%22100%22 height=%22100%22/><text x=%2250%22 y=%2250%22 fill=%22%2394a3b8%22 text-anchor=%22middle%22 dy=%22.3em%22>No Image</text></svg>'">
                <span class="result-score">${(result.score * 100).toFixed(1)}%</span>
            </div>
            <div class="result-info">
                <div class="result-title">${result.source_file}</div>
                <div class="result-meta">
                    <span>${primaryLabel}</span>
                    <span>${secondaryLabel}</span>
                </div>
                ${result.is_night !== undefined ? `
                    <span class="result-badge ${result.is_night ? 'night' : 'day'}">
                        ${result.is_night ? 'Night' : 'Day'}
                    </span>
                ` : ''}
            </div>
        </div>
        `;
    }).join('');

    // Store results for modal
    window.currentResults = data.results;
}

function getThumbnailUrl(path) {
    if (!path) return '';
    // Convert local path to URL
    const filename = path.split('/').pop();
    const videoDir = path.split('/').slice(-2, -1)[0];
    return `/thumbnails/${videoDir}/${filename}`;
}

function getFrameUrl(path) {
    if (!path) return '';
    // Convert local path to URL for full-quality frame
    const filename = path.split('/').pop();
    const videoDir = path.split('/').slice(-2, -1)[0];
    return `/frames/${videoDir}/${filename}`;
}

function getVideoUrlForResult(result) {
    if (!result) return null;

    // Prefer explicit video_path if provided
    let filename = null;
    if (result.video_path) {
        const parts = result.video_path.split(/[\\/]/);
        const last = parts[parts.length - 1];
        if (last && last.includes('.')) {
            filename = last;
        }
    }

    // Fallback to source_file if it looks like a video file
    if (!filename && result.source_file) {
        const lower = result.source_file.toLowerCase();
        if (lower.endsWith('.mp4') || lower.endsWith('.avi') || lower.endsWith('.mov') || lower.endsWith('.mkv')) {
            filename = result.source_file;
        }
    }

    if (!filename) return null;
    return `/raw/${filename}`;
}

// Modal
function initModal() {
    const closeBtn = modal.querySelector('.modal-close');

    closeBtn.addEventListener('click', () => {
        modal.classList.remove('active');
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            modal.classList.remove('active');
        }
    });

    if (modalOpenVideoBtn) {
        modalOpenVideoBtn.addEventListener('click', () => {
            const result = window.currentSelectedResult;
            if (!result) return;

            const videoUrl = getVideoUrlForResult(result);
            if (!videoUrl) {
                alert('No video available for this frame.');
                return;
            }

            // For semantic clips, use clip_start_seconds; otherwise use seconds_offset
            let seconds = null;
            if (typeof result.clip_start_seconds === 'number') {
                seconds = result.clip_start_seconds;
            } else if (typeof result.seconds_offset === 'number') {
                seconds = result.seconds_offset;
            }

            // Debug info
            const debugInfo = `Debug Info:
clip_start_seconds: ${result.clip_start_seconds} (type: ${typeof result.clip_start_seconds})
seconds_offset: ${result.seconds_offset} (type: ${typeof result.seconds_offset})
Final seconds: ${seconds}

Will open URL with ${seconds !== null ? `#t=${seconds.toFixed(1)}` : 'NO TIMESTAMP'}`;

            console.log(debugInfo);
            alert(debugInfo);

            // Construct full URL with timestamp
            const baseUrl = videoUrl.startsWith('http') ? videoUrl : `${window.location.origin}${videoUrl}`;
            const finalUrl = seconds !== null ? `${baseUrl}#t=${seconds.toFixed(1)}` : baseUrl;

            console.log('Final URL:', finalUrl);
            window.open(finalUrl, '_blank');
        });
    }
}

window.showResultDetail = function(index) {
    const result = window.currentResults[index];
    if (!result) return;

    window.currentSelectedResult = result;

    const isClip = result.source_type === 'video_clip' ||
        (typeof result.clip_start_seconds === 'number' && typeof result.clip_end_seconds === 'number');

    // Use high-quality frame image if available, otherwise fall back to thumbnail
    const imageUrl = result.frame_path
        ? getFrameUrl(result.frame_path)
        : getThumbnailUrl(result.thumbnail_path);
    document.getElementById('modal-image').src = imageUrl;
    document.getElementById('modal-title').textContent = isClip ? 'Clip Details' : 'Frame Details';
    document.getElementById('modal-source').textContent = result.source_file;
    document.getElementById('modal-frame').textContent = result.frame_number;
    document.getElementById('modal-timestamp').textContent = result.timestamp || 'Unknown';
    document.getElementById('modal-zone').textContent = result.zone || 'Unknown';
    document.getElementById('modal-score').textContent = `${(result.score * 100).toFixed(1)}%`;

    const frameLabelEl = document.getElementById('modal-frame-label');
    const clipRangeRow = document.getElementById('modal-clip-range-row');
    const clipRangeEl = document.getElementById('modal-clip-range');
    const clipMetaRow = document.getElementById('modal-clip-meta-row');
    const clipMetaEl = document.getElementById('modal-clip-meta');

    if (isClip) {
        frameLabelEl.textContent = 'Clip index:';

        const clipIndex = typeof result.clip_index === 'number'
            ? result.clip_index
            : result.frame_number;

        const startTs = result.clip_start_timestamp || result.timestamp || '';
        const endTs = result.clip_end_timestamp || '';

        const startSec = typeof result.clip_start_seconds === 'number'
            ? result.clip_start_seconds
            : result.seconds_offset;
        const endSec = typeof result.clip_end_seconds === 'number'
            ? result.clip_end_seconds
            : undefined;

        let duration = null;
        if (typeof startSec === 'number' && typeof endSec === 'number') {
            duration = Math.max(0, endSec - startSec);
        }

        document.getElementById('modal-frame').textContent = clipIndex;

        if (startTs || endTs) {
            clipRangeRow.style.display = '';
            if (startTs && endTs) {
                clipRangeEl.textContent = `${startTs} → ${endTs}`;
            } else {
                clipRangeEl.textContent = startTs || endTs;
            }
        } else {
            clipRangeRow.style.display = 'none';
            clipRangeEl.textContent = '';
        }

        const metaParts = [];
        if (duration !== null && !Number.isNaN(duration)) {
            metaParts.push(`Duration: ${duration.toFixed(1)}s`);
        }
        if (typeof result.num_frames === 'number') {
            metaParts.push(`Frames: ${result.num_frames}`);
        }

        if (metaParts.length > 0) {
            clipMetaRow.style.display = '';
            clipMetaEl.textContent = metaParts.join(' · ');
        } else {
            clipMetaRow.style.display = 'none';
            clipMetaEl.textContent = '';
        }
    } else {
        frameLabelEl.textContent = 'Frame:';
        clipRangeRow.style.display = 'none';
        clipRangeEl.textContent = '';
        clipMetaRow.style.display = 'none';
        clipMetaEl.textContent = '';
    }

    if (modalOpenVideoBtn && modalVideoHint) {
        const videoUrl = getVideoUrlForResult(result);
        const hasTime = typeof result.clip_start_seconds === 'number' || typeof result.seconds_offset === 'number';

        if (videoUrl) {
            modalOpenVideoBtn.disabled = false;
            modalVideoHint.textContent = hasTime
                ? 'Opens the source video in a new tab at this frame\'s timestamp.'
                : 'Opens the source video in a new tab.';
        } else {
            modalOpenVideoBtn.disabled = true;
            modalVideoHint.textContent = 'No video source available for this frame.';
        }
    }

    modal.classList.add('active');
};

// Loading
function showLoading() {
    loadingOverlay.hidden = false;
}

function hideLoading() {
    loadingOverlay.hidden = true;
}

// Error handling
function showError(message) {
    resultsGrid.innerHTML = `
        <div class="no-results" style="color: var(--danger);">
            <p>${message}</p>
        </div>
    `;
}

// Check model status and show warnings
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) return;

        const data = await response.json();
        const stats = data.search_service || {};

        const warnings = [];
        if (stats.using_mock_embeddings) {
            warnings.push('⚠️ Using mock embeddings (model not loaded)');
        }
        if (stats.using_mock_reranker) {
            warnings.push('⚠️ Using mock reranker (model not loaded)');
        }

        if (warnings.length > 0) {
            showWarningBanner(warnings.join(' | '));
        }
    } catch (error) {
        console.error('Failed to check model status:', error);
    }
}

function showWarningBanner(message) {
    const banner = document.createElement('div');
    banner.className = 'warning-banner';
    banner.innerHTML = `
        <div class="warning-content">
            <span>${message}</span>
            <button class="warning-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    document.querySelector('.container').insertBefore(banner, document.querySelector('main'));
}
