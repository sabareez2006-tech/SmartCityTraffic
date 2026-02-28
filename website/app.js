/**
 * SmartCity Traffic — Interactive Dashboard
 * Handles UI interactions, parameter collection, simulation execution,
 * and results display.
 */

// ============================================
// CONSTANTS & STATE
// ============================================
const ZONE_TYPES = ['residential', 'commercial', 'industrial', 'suburban', 'downtown'];
const ZONE_ICONS = { residential: '🏠', commercial: '🏢', industrial: '🏭', suburban: '🌳', downtown: '🏙️' };
const ZONE_COLORS = {
    residential: '#6366f1',
    commercial: '#f59e0b',
    industrial: '#10b981',
    suburban: '#a855f7',
    downtown: '#f43f5e'
};

const DEFAULT_CONFIG = {
    gridRows: 5, gridCols: 5, cityArea: 20,
    simHours: 24, timeStep: 15, numDays: 7, baseRequests: 50,
    flRounds: 20, localEpochs: 3, learningRate: 0.001, batchSize: 16,
    hiddenSize: 64, numLayers: 2,
    compressionRatio: 0.3, compressionThreshold: 0.01,
    corrThreshold: 0.85, minClients: 3,
    numTaxis: 100, taxiSpeed: 30, maxWait: 7, randomSeed: 42
};

let isRunning = false;
let currentViz = 'training_loss';
let simulationResults = null;

// ============================================
// DOM ELEMENTS
// ============================================
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCityGrid();
    initSliders();
    initGridInputListeners();
    initRunButton();
    initResetButton();
    initClearTerminal();
    initVizTabs();
    initScrollAnimations();
    loadExistingResults();
});

// ============================================
// NAVIGATION
// ============================================
function initNavigation() {
    const navbar = $('navbar');
    const menuBtn = $('mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');

    // Scroll effect
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
        updateActiveNavLink();
    });

    // Mobile menu
    if (menuBtn) {
        menuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('open');
        });
    }

    // Smooth scroll for nav links
    $$('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            navLinks.classList.remove('open');
        });
    });
}

function updateActiveNavLink() {
    const sections = ['hero', 'configure', 'results', 'visualizations', 'about'];
    const scrollPos = window.scrollY + 200;

    let current = 'hero';
    sections.forEach(id => {
        const section = $(id);
        if (section && section.offsetTop <= scrollPos) {
            current = id;
        }
    });

    $$('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.section === current);
    });
}

// ============================================
// CITY GRID VISUALIZATION
// ============================================
function initCityGrid() {
    buildGrid(5, 5);
}

function buildGrid(rows, cols) {
    const grid = $('city-grid-viz');
    if (!grid) return;

    grid.innerHTML = '';
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    const total = rows * cols;
    for (let i = 0; i < total; i++) {
        const cell = document.createElement('div');
        const type = ZONE_TYPES[i % ZONE_TYPES.length];
        cell.className = `grid-cell ${type}`;

        const icon = document.createElement('span');
        icon.className = 'cell-icon';
        icon.textContent = ZONE_ICONS[type];

        const label = document.createElement('span');
        label.className = 'cell-label';
        label.textContent = i;

        cell.appendChild(icon);
        cell.appendChild(label);

        // Hover tooltip
        cell.title = `Zone ${i} — ${type.charAt(0).toUpperCase() + type.slice(1)}`;

        // Staggered animation
        cell.style.animation = `fadeInUp 0.4s var(--ease-out) ${i * 0.03}s both`;

        grid.appendChild(cell);
    }

    // Update stats
    const zonesEl = $('total-zones');
    if (zonesEl) zonesEl.textContent = total;

    const statZones = $('stat-zones');
    if (statZones) statZones.textContent = total;
}

function initGridInputListeners() {
    const rowInput = $('grid-rows');
    const colInput = $('grid-cols');

    [rowInput, colInput].forEach(input => {
        if (input) {
            input.addEventListener('change', () => {
                const rows = parseInt(rowInput.value) || 5;
                const cols = parseInt(colInput.value) || 5;
                buildGrid(rows, cols);
            });
        }
    });

    // Update fleet stats
    const taxiInput = $('num-taxis');
    if (taxiInput) {
        taxiInput.addEventListener('change', () => {
            const statFleet = $('stat-fleet');
            if (statFleet) statFleet.textContent = taxiInput.value;
        });
    }

    const roundsInput = $('fl-rounds');
    if (roundsInput) {
        roundsInput.addEventListener('change', () => {
            const statRounds = $('stat-rounds');
            if (statRounds) statRounds.textContent = roundsInput.value;
        });
    }
}

// ============================================
// SLIDERS
// ============================================
function initSliders() {
    const compressionSlider = $('compression-ratio');
    const compressionVal = $('compression-ratio-val');
    if (compressionSlider && compressionVal) {
        compressionSlider.addEventListener('input', () => {
            compressionVal.textContent = Math.round(compressionSlider.value * 100) + '%';
        });
    }

    const corrSlider = $('corr-threshold');
    const corrVal = $('corr-threshold-val');
    if (corrSlider && corrVal) {
        corrSlider.addEventListener('input', () => {
            corrVal.textContent = Math.round(corrSlider.value * 100) + '%';
        });
    }
}

// ============================================
// CONFIGURATION COLLECTION
// ============================================
function getConfig() {
    return {
        grid_rows: parseInt($('grid-rows')?.value) || 5,
        grid_cols: parseInt($('grid-cols')?.value) || 5,
        city_area: parseFloat($('city-area')?.value) || 20,
        sim_hours: parseInt($('sim-hours')?.value) || 24,
        time_step: parseInt($('time-step')?.value) || 15,
        num_days: parseInt($('num-days')?.value) || 7,
        base_requests: parseInt($('base-requests')?.value) || 50,
        fl_rounds: parseInt($('fl-rounds')?.value) || 20,
        local_epochs: parseInt($('local-epochs')?.value) || 3,
        learning_rate: parseFloat($('learning-rate')?.value) || 0.001,
        batch_size: parseInt($('batch-size')?.value) || 16,
        hidden_size: parseInt($('hidden-size')?.value) || 64,
        num_layers: parseInt($('num-layers')?.value) || 2,
        compression_ratio: parseFloat($('compression-ratio')?.value) || 0.3,
        compression_threshold: parseFloat($('compression-threshold')?.value) || 0.01,
        corr_threshold: parseFloat($('corr-threshold')?.value) || 0.85,
        min_clients: parseInt($('min-clients')?.value) || 3,
        num_taxis: parseInt($('num-taxis')?.value) || 100,
        taxi_speed: parseInt($('taxi-speed')?.value) || 30,
        max_wait: parseInt($('max-wait')?.value) || 7,
        random_seed: parseInt($('random-seed')?.value) || 42
    };
}

// ============================================
// SIMULATION EXECUTION
// ============================================
function initRunButton() {
    const btn = $('run-simulation-btn');
    if (btn) {
        btn.addEventListener('click', runSimulation);
    }
}

async function runSimulation() {
    if (isRunning) return;
    isRunning = true;

    const btn = $('run-simulation-btn');
    btn.classList.add('running');

    const terminal = $('terminal-output');
    terminal.innerHTML = '';

    const config = getConfig();

    addTerminalLine('header', '═══════════════════════════════════════════');
    addTerminalLine('header', '  Federated Learning Traffic Simulation');
    addTerminalLine('header', '═══════════════════════════════════════════');
    addTerminalLine('info', `Configuration loaded: ${config.grid_rows}×${config.grid_cols} grid, ${config.fl_rounds} FL rounds`);

    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Stream the response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    try {
                        const msg = JSON.parse(data);
                        handleStreamMessage(msg);
                    } catch {
                        // Not JSON, skip
                    }
                }
            }
        }

        addTerminalLine('success', '✅ Simulation completed successfully!');
        showToast('Simulation completed!', 'success');

    } catch (err) {
        addTerminalLine('error', `❌ Error: ${err.message}`);
        addTerminalLine('info', 'Tip: Make sure the Flask backend is running (python app.py)');
        showToast('Simulation failed. Check if backend is running.', 'error');

        // Try to load existing results
        loadExistingResults();
    }

    isRunning = false;
    btn.classList.remove('running');
}

function handleStreamMessage(msg) {
    switch (msg.type) {
        case 'log':
            addTerminalLine(msg.level || '', msg.message);
            break;
        case 'progress':
            addTerminalLine('info', `Progress: ${msg.message}`);
            break;
        case 'results':
            simulationResults = msg.data;
            displayResults(msg.data);
            break;
        case 'complete':
            if (msg.data) {
                simulationResults = msg.data;
                displayResults(msg.data);
            }
            break;
        case 'error':
            addTerminalLine('error', msg.message);
            break;
    }
}

// ============================================
// TERMINAL
// ============================================
function addTerminalLine(type, text) {
    const terminal = $('terminal-output');
    if (!terminal) return;

    // Remove typing cursor from previous line
    const typingEl = terminal.querySelector('.typing');
    if (typingEl) typingEl.classList.remove('typing');

    const line = document.createElement('div');
    line.className = 'terminal-line';

    const prompt = document.createElement('span');
    prompt.className = 'terminal-prompt';
    prompt.textContent = type === 'error' ? '✗' : type === 'success' ? '✓' : '$';

    const textEl = document.createElement('span');
    textEl.className = `terminal-text ${type} typing`;
    textEl.textContent = text;

    line.appendChild(prompt);
    line.appendChild(textEl);
    terminal.appendChild(line);

    // Auto-scroll
    terminal.scrollTop = terminal.scrollHeight;

    // Remove typing effect after a delay
    setTimeout(() => textEl.classList.remove('typing'), 600);
}

function initClearTerminal() {
    const btn = $('clear-terminal');
    if (btn) {
        btn.addEventListener('click', () => {
            const terminal = $('terminal-output');
            terminal.innerHTML = '<div class="terminal-line"><span class="terminal-prompt">$</span><span class="terminal-text typing">Terminal cleared.</span></div>';
        });
    }
}

// ============================================
// RESULTS DISPLAY
// ============================================
function displayResults(data) {
    if (!data) return;

    const fl = data.fl_metrics || {};
    const dynamic = data.dynamic_metrics || {};
    const staticM = data.static_metrics || {};
    const compression = data.compression_stats || {};

    // Update metric cards with animation
    animateValue('res-loss', fl.final_loss, 6);
    animateValue('res-mae', fl.mae, 4);
    animateValue('res-rmse', fl.rmse, 4);
    animateValue('res-savings', compression.savings || '—');
    animateValue('res-service-dyn', dynamic.service_rate ? `${(dynamic.service_rate * 100).toFixed(1)}%` : dynamic.service_rate_display || '—');
    animateValue('res-wait-dyn', dynamic.avg_wait_time != null ? `${dynamic.avg_wait_time.toFixed(1)}min` : '—');
    animateValue('res-rides-dyn', dynamic.total_rides_completed != null ? dynamic.total_rides_completed.toLocaleString() : '—');
    animateValue('res-util-dyn', dynamic.fleet_utilization ? `${(dynamic.fleet_utilization * 100).toFixed(1)}%` : dynamic.fleet_utilization_display || '—');

    // Update comparison table
    buildComparisonTable(dynamic, staticM);

    // Update viz tabs
    if (data.plots) {
        loadVisualization(currentViz);
    }
}

function animateValue(id, value, decimals = 0) {
    const el = $(id);
    if (!el) return;

    if (typeof value === 'number') {
        el.textContent = value.toFixed(decimals);
    } else {
        el.textContent = value;
    }

    // Flash animation
    el.style.transform = 'scale(1.15)';
    el.style.transition = 'transform 0.3s ease';
    setTimeout(() => {
        el.style.transform = 'scale(1)';
    }, 300);
}

function buildComparisonTable(dynamic, staticM) {
    const container = $('comparison-table-container');
    const tbody = $('comparison-tbody');
    if (!container || !tbody) return;

    if (!dynamic || !staticM || Object.keys(dynamic).length === 0) return;

    container.style.display = 'block';
    tbody.innerHTML = '';

    const metrics = [
        { label: 'Service Rate', dynKey: 'service_rate', statKey: 'service_rate', format: 'percent' },
        { label: 'Avg Wait Time', dynKey: 'avg_wait_time', statKey: 'avg_wait_time', format: 'time' },
        { label: 'Fleet Utilization', dynKey: 'fleet_utilization', statKey: 'fleet_utilization', format: 'percent' },
        { label: 'Completed Rides', dynKey: 'total_rides_completed', statKey: 'total_rides_completed', format: 'number' },
        { label: 'Expired Requests', dynKey: 'expired_requests', statKey: 'expired_requests', format: 'number' }
    ];

    metrics.forEach(m => {
        const dynVal = dynamic[m.dynKey];
        const statVal = staticM[m.statKey];
        if (dynVal == null && statVal == null) return;

        const tr = document.createElement('tr');

        const formatVal = (v, fmt) => {
            if (v == null) return '—';
            if (fmt === 'percent') return `${(v * 100).toFixed(1)}%`;
            if (fmt === 'time') return `${v.toFixed(1)} min`;
            if (fmt === 'number') return v.toLocaleString();
            return v;
        };

        let improvement = '';
        let isNeg = false;
        if (dynVal != null && statVal != null && statVal !== 0) {
            const diff = ((dynVal - statVal) / Math.abs(statVal) * 100);
            const sign = diff > 0 ? '+' : '';
            improvement = `${sign}${diff.toFixed(1)}%`;
            // For wait time and expired requests, negative is good
            if (m.label === 'Avg Wait Time' || m.label === 'Expired Requests') {
                isNeg = diff > 0;
            } else {
                isNeg = diff < 0;
            }
        }

        tr.innerHTML = `
            <td>${m.label}</td>
            <td>${formatVal(dynVal, m.format)}</td>
            <td>${formatVal(statVal, m.format)}</td>
            <td class="${isNeg ? 'negative' : ''}">${improvement}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ============================================
// LOAD EXISTING RESULTS
// ============================================
async function loadExistingResults() {
    try {
        const resp = await fetch('/api/results');
        if (resp.ok) {
            const data = await resp.json();
            if (data && data.fl_metrics) {
                simulationResults = data;
                displayResults(data);
                addTerminalLine('info', 'Loaded previous simulation results.');
            }
        }
    } catch (e) {
        // Backend not running — try to load images directly
        loadExistingPlots();
    }
}

function loadExistingPlots() {
    // Try to load existing images from results folder
    const img = $('viz-image');
    const placeholder = $('viz-placeholder');
    if (!img) return;

    // Attempt to load the current viz
    const testImg = new Image();
    testImg.onload = () => {
        img.src = testImg.src;
        img.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';
    };
    testImg.onerror = () => {
        // Keep placeholder
    };
    testImg.src = `../results/${currentViz}.png`;
}

// ============================================
// VISUALIZATIONS
// ============================================
function initVizTabs() {
    $$('.viz-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.viz-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentViz = tab.dataset.viz;
            loadVisualization(currentViz);
        });
    });
}

function loadVisualization(vizName) {
    const img = $('viz-image');
    const placeholder = $('viz-placeholder');
    if (!img) return;

    // Try API first, then direct file
    const testImg = new Image();
    testImg.onload = () => {
        img.src = testImg.src;
        img.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';
    };
    testImg.onerror = () => {
        // Try direct path
        const testImg2 = new Image();
        testImg2.onload = () => {
            img.src = testImg2.src;
            img.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
        };
        testImg2.onerror = () => {
            img.style.display = 'none';
            if (placeholder) placeholder.style.display = 'flex';
        };
        testImg2.src = `../results/${vizName}.png`;
    };
    testImg.src = `/api/plot/${vizName}`;
}

// ============================================
// RESET
// ============================================
function initResetButton() {
    const btn = $('reset-btn');
    if (btn) {
        btn.addEventListener('click', resetDefaults);
    }
}

function resetDefaults() {
    const mapping = {
        'grid-rows': DEFAULT_CONFIG.gridRows,
        'grid-cols': DEFAULT_CONFIG.gridCols,
        'city-area': DEFAULT_CONFIG.cityArea,
        'sim-hours': DEFAULT_CONFIG.simHours,
        'time-step': DEFAULT_CONFIG.timeStep,
        'num-days': DEFAULT_CONFIG.numDays,
        'base-requests': DEFAULT_CONFIG.baseRequests,
        'fl-rounds': DEFAULT_CONFIG.flRounds,
        'local-epochs': DEFAULT_CONFIG.localEpochs,
        'learning-rate': DEFAULT_CONFIG.learningRate,
        'batch-size': DEFAULT_CONFIG.batchSize,
        'hidden-size': DEFAULT_CONFIG.hiddenSize,
        'num-layers': DEFAULT_CONFIG.numLayers,
        'compression-ratio': DEFAULT_CONFIG.compressionRatio,
        'compression-threshold': DEFAULT_CONFIG.compressionThreshold,
        'corr-threshold': DEFAULT_CONFIG.corrThreshold,
        'min-clients': DEFAULT_CONFIG.minClients,
        'num-taxis': DEFAULT_CONFIG.numTaxis,
        'taxi-speed': DEFAULT_CONFIG.taxiSpeed,
        'max-wait': DEFAULT_CONFIG.maxWait,
        'random-seed': DEFAULT_CONFIG.randomSeed
    };

    for (const [id, val] of Object.entries(mapping)) {
        const el = $(id);
        if (el) el.value = val;
    }

    // Update slider labels
    $('compression-ratio-val').textContent = '30%';
    $('corr-threshold-val').textContent = '85%';

    // Rebuild grid
    buildGrid(5, 5);

    // Reset stats
    $('stat-zones').textContent = '25';
    $('stat-fleet').textContent = '100';
    $('stat-rounds').textContent = '20';

    showToast('Parameters reset to defaults', 'info');
}

// ============================================
// TOAST NOTIFICATIONS
// ============================================
function showToast(message, type = 'info') {
    const toast = $('toast');
    const icon = $('toast-icon');
    const msg = $('toast-message');
    if (!toast) return;

    const icons = {
        success: '✓',
        error: '✗',
        info: 'ℹ',
        warning: '⚠'
    };

    toast.className = `toast ${type}`;
    icon.textContent = icons[type] || 'ℹ';
    msg.textContent = message;

    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// ============================================
// SCROLL ANIMATIONS
// ============================================
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeInUp 0.6s var(--ease-out) forwards';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    // Observe cards
    setTimeout(() => {
        $$('.config-card, .result-card, .about-card').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            observer.observe(el);
        });
    }, 500);
}
