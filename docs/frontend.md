# Frontend Documentation

This document covers the web dashboard UI, its features, structure, and customization options.

---

## Overview

The frontend is a **single-page application** built with vanilla HTML, CSS, and JavaScript -- no build tools or frameworks required. It is served by FastAPI as static files at `/static/index.html`.

**Access URL:** `http://localhost:8000/static/index.html`

```
frontend/
├── index.html    # Main HTML page (358 lines)
├── styles.css    # Stylesheet
└── app.js        # Frontend JavaScript (1027 lines)
```

---

## Features

### Search Interface

The dashboard provides four search modes via a tabbed interface:

| Tab | Description | Input |
|-----|-------------|-------|
| **Text Search** | Natural language queries | Text input + example buttons |
| **Image Search** | Visual similarity search | Image upload (click or drag-and-drop) |
| **Multimodal** | Combined text + image | Image upload + text input |
| **OCR Search** | Find text in footage | Text input + example buttons |

### Search Filters

All search modes share a common filter panel:

| Filter | Options | Description |
|--------|---------|-------------|
| **Zone** | All Zones, Main Gate, Perimeter, Parking, Warehouse | Location-based filtering |
| **Time of Day** | Any Time, Daytime, Nighttime | Day/night filtering |
| **Results** | 10, 20, 50, 100 | Number of results to return |
| **Use Reranker** | Toggle | Enable Qwen3-VL reranking for better results |

### Results Display

Search results appear in a responsive grid layout. Each result card shows:

- Thumbnail image (224x224)
- Similarity score (percentage)
- Source file name
- Frame number or clip index
- Zone/location
- Timestamp

### Result Detail Modal

Clicking a result opens a modal with:

- Full-size frame or thumbnail
- Source file name
- Frame number / clip range
- Timestamp
- Zone
- Similarity score
- "Open video at this time" button (opens source video at timestamp)

### Video Ingestion Modal

The "Ingest Video" button opens a modal for uploading videos:

- **File upload** -- Drag-and-drop or click to select (MP4, AVI, MOV, MKV)
- **Zone** -- Optional zone identifier
- **Ingestion Mode** -- Per-frame or Semantic clips
- **Clip Duration** -- For semantic mode (2-20 seconds)
- **Max Frames per Clip** -- Memory control (4-128 frames)
- **Upload progress bar** -- Shows upload and processing progress with ETA

### RTSP Stream Ingestion Modal

The "RTSP Stream" button opens a modal for live camera ingestion:

- **RTSP URL** -- Stream URL (rtsp://host:port/path)
- **Zone** -- Location identifier
- **Capture Duration** -- How long to capture (seconds)
- **Max Frames** -- Alternative termination by frame count
- **Clip Duration / Max Frames per Clip** -- Semantic clip settings
- **Camera Metadata** (collapsible) -- Camera ID and Site ID

---

## Architecture

### API Communication

The frontend communicates with the FastAPI backend using the same origin (no CORS issues). The `API_BASE` is set to `''` (empty string) for same-origin requests.

```javascript
const API_BASE = '';  // Same origin

// Example: Text search
const response = await fetch(`${API_BASE}/search/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k, filters })
});
```

### State Management

Simple state variables manage the UI:

```javascript
let currentTab = 'text';         // Active search tab
let selectedImage = null;        // Image for image search
let multimodalImage = null;      // Image for multimodal search
let ingestVideoFile = null;      // Video for ingestion
```

### Initialization

On `DOMContentLoaded`, the app initializes:

1. **Tab navigation** -- Click handlers for search mode tabs
2. **Search forms** -- Submit handlers for all four search forms
3. **Image upload** -- Drag-and-drop and click handlers for image inputs
4. **Ingest upload** -- Video upload with progress tracking
5. **RTSP ingest** -- RTSP stream ingestion form
6. **Modal** -- Result detail modal (click to open, close button, escape key)
7. **Example queries** -- Pre-filled query buttons
8. **Health check** -- Checks model status on load

---

## Key Functions

### Search Functions

| Function | Description |
|----------|-------------|
| `performTextSearch(query)` | Sends text query to `/search/text` |
| `performImageSearch(imageFile)` | Sends image to `/search/image` as FormData |
| `performMultimodalSearch(imageFile, query)` | Sends image + text to `/search/multimodal` |
| `performOCRSearch(text)` | Sends text to `/search/ocr` |

### Display Functions

| Function | Description |
|----------|-------------|
| `displayResults(data)` | Renders search results in the grid |
| `createResultCard(result)` | Creates a single result card element |
| `showModal(result)` | Opens the result detail modal |
| `getFilters()` | Collects current filter values |

### Utility Functions

| Function | Description |
|----------|-------------|
| `showLoading()` / `hideLoading()` | Toggle the loading overlay spinner |
| `checkModelStatus()` | Calls `/health` to check if models are loaded |
| `formatTime(seconds)` | Formats seconds to `HH:MM:SS` |

### Ingestion Functions

| Function | Description |
|----------|-------------|
| `handleVideoIngest(form)` | Uploads video and tracks progress |
| `handleRTSPIngest(form)` | Submits RTSP stream ingestion request |

---

## UI Components

### Tab System

```html
<div class="search-tabs">
    <button class="tab-btn active" data-tab="text">Text Search</button>
    <button class="tab-btn" data-tab="image">Image Search</button>
    <button class="tab-btn" data-tab="multimodal">Multimodal</button>
    <button class="tab-btn" data-tab="ocr">OCR Search</button>
</div>
```

Tabs are activated by adding/removing the `active` class on both the button and corresponding content div.

### Image Upload (Drag-and-Drop)

The image upload areas support both click-to-select and drag-and-drop:

```html
<div class="upload-area" id="image-upload-area">
    <input type="file" id="image-file" accept="image/*" hidden>
    <div class="upload-placeholder">...</div>
    <img id="image-preview" src="" alt="Preview" hidden>
</div>
```

When an image is selected, a preview is displayed in place of the upload placeholder.

### Results Grid

Results are displayed in a responsive grid layout:

```html
<div id="results-grid" class="results-grid">
    <!-- Populated dynamically by JavaScript -->
</div>
```

Each result card is generated by `createResultCard()` and includes the thumbnail, metadata, and a click handler to open the modal.

### Loading Overlay

A full-screen overlay with a spinner is shown during search operations:

```html
<div id="loading-overlay" class="loading-overlay" hidden>
    <div class="spinner"></div>
    <p>Searching...</p>
</div>
```

### Upload Progress Bar

Video upload progress is displayed globally in the header:

```html
<div id="global-upload-progress" class="global-progress" hidden>
    <div class="global-progress-bar"></div>
    <div class="global-progress-info">
        <span id="global-progress-text">Uploading...</span>
        <span id="global-progress-eta"></span>
    </div>
</div>
```

---

## Modals

### Result Detail Modal (`video-modal`)

Displays detailed information about a search result:
- Full-size image viewer
- Metadata panel (source, frame, timestamp, zone, score)
- Semantic clip information (if applicable)
- "Open video at this time" button for video results

### Ingest Video Modal (`ingest-modal`)

Video upload and ingestion interface:
- File upload area with drag-and-drop
- Zone and ingestion mode configuration
- Semantic clip options (shown when semantic mode selected)
- Submit button with status messages

### RTSP Stream Modal (`rtsp-modal`)

RTSP camera ingestion interface:
- RTSP URL input
- Zone configuration
- Duration/frame count termination
- Clip settings
- Collapsible camera metadata section

---

## Example Queries

Pre-configured example query buttons for quick testing:

### Text Search Examples
- "White sedan on zebra crossing"
- "Truck moving through farms"
- "Nighttime footage from main gate"
- "Dog running on road"

### OCR Search Examples
- License plate: "7829"
- Sign: "DANGER"
- Sign: "RESTRICTED"

---

## Customization

### Adding Search Filters

1. Add a new `<select>` or `<input>` in the `filters-grid` div in `index.html`
2. Update `getFilters()` in `app.js` to include the new filter value
3. The backend already supports many filter types (see API Reference)

### Changing Default Values

Edit the HTML elements directly:
- Default results count: `<option value="20" selected>20</option>`
- Reranker default: `<input type="checkbox" id="use-reranker" />`
- Clip duration: `<input ... value="4.0">`

### Adding New Search Modes

1. Add a new tab button and content section in `index.html`
2. Create a new `perform*Search()` function in `app.js`
3. Add the corresponding API endpoint in `api/routes/search.py`

### Styling

All styles are in `frontend/styles.css`. The design uses CSS custom properties for theming. Key classes:
- `.container` -- Main page wrapper
- `.search-section` -- Search form area
- `.results-grid` -- Results display grid
- `.result-card` -- Individual result card
- `.modal` -- Modal overlay
- `.modal-content` -- Modal content box
- `.search-btn` -- Primary action button
- `.secondary-btn` -- Secondary action button

---

## Static File Serving

The frontend and data assets are served by FastAPI's `StaticFiles` middleware:

| Mount Point | Directory | Content |
|-------------|-----------|---------|
| `/static/` | `frontend/` | HTML, CSS, JS |
| `/thumbnails/` | `data/thumbnails/` | 224x224 thumbnail images |
| `/frames/` | `data/frames/` | Full-resolution frames |
| `/raw/` | `data/raw/` | Source video files |

Result URLs in search responses are automatically converted from filesystem paths to these serving URLs by the `path_to_url()` function in `api/routes/search.py`.
