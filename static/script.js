document.addEventListener('DOMContentLoaded', () => {
  // --- UI CORE ELEMENTS ---
  const splashScreen = document.getElementById('splash-screen');
  const themeToggle = document.getElementById('theme-toggle');
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const preview = document.getElementById('preview');
  const analyzeBtn = document.getElementById('analyze-btn');
  const processingOverlay = document.getElementById('processing-overlay');
  const dynamicProgress = document.getElementById('dynamic-progress');
  const progressPercent = document.getElementById('progress-percent');
  const progressStatus = document.getElementById('progress-status');
  const errorMessage = document.getElementById('error-message');
  const resultCard = document.getElementById('result-card');
  const resultEmpty = document.getElementById('result-empty');
  
  // --- STATE ---
  const allowed = ['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'];
  let selectedFile = null;
  let currentPreviewUrl = null;
  let currentXAIView = 'basic';
  let currentRegionView = 'all';
  let focusRegionExpanded = false;

  // --- THEME ENGINE ---
  const setTheme = (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('authentix-theme', theme);
    if (themeToggle) {
        themeToggle.textContent = theme === 'dark' ? '🌙' : '☀️';
    }
  };
  const currentTheme = localStorage.getItem('authentix-theme') || 'dark';
  setTheme(currentTheme);
  
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const newTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      setTheme(newTheme);
    });
  }

  // --- SPLASH SCREEN ANIMATION ---
  if (splashScreen) {
    const canvas = document.getElementById('splash-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        function resize() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resize);
        resize();

        const particles = [];
        const numParticles = 35;
        for (let i = 0; i < numParticles; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 1.5,
                vy: (Math.random() - 0.5) * 1.5,
                radius: Math.random() * 2 + 1
            });
        }

        function drawParticles() {
            if (!splashScreen.parentNode || splashScreen.style.display === 'none') return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < numParticles; i++) {
                let p = particles[i];
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(0, 198, 255, 0.8)';
                ctx.fill();
                for (let j = i + 1; j < numParticles; j++) {
                    let p2 = particles[j];
                    let dx = p.x - p2.x;
                    let dy = p.y - p2.y;
                    let dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 90) {
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgba(0, 198, 255, ${1 - dist / 90})`;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                }
            }
            if (!splashScreen.classList.contains('fade-out')) {
                requestAnimationFrame(drawParticles);
            }
        }
        drawParticles();
    }

    const logs = ["Initializing neural engine...", "Loading detection models...", "Analyzing facial artifacts...", "Deepfake probability engine ready..."];
    const logContainer = document.querySelector('.log-line');
    let currentLog = 0;
    if (logContainer) {
        setTimeout(() => {
            let logInterval = setInterval(() => {
                if (currentLog < logs.length) {
                    logContainer.innerHTML += `<div>> ${logs[currentLog]}</div>`;
                    currentLog++;
                } else clearInterval(logInterval);
            }, 300);
        }, 3500);
    }
    setTimeout(() => {
        splashScreen.classList.add('fade-out');
        setTimeout(() => { splashScreen.style.display = 'none'; }, 500);
    }, 4700);
  }

  // --- HELPERS ---
  function setError(msg = '') { errorMessage.textContent = msg; }
  function escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function uniqueEvidenceLines(lines) {
    const seen = new Set();
    return (Array.isArray(lines) ? lines : [])
      .map((line) => String(line ?? '').trim())
      .filter(Boolean)
      .filter((line) => {
        const key = line.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
        if (!key || seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }

  function buildKeyEvidencePoints(result, faceForensics, faceswapAnalysis) {
    const selected = uniqueEvidenceLines([
      ...(Array.isArray(result.reasons) ? result.reasons : []),
      faceswapAnalysis.summary,
      faceForensics.summary,
      faceForensics.landmark_integrity?.summary,
      result.explanation,
    ]).slice(0, 3);

    const fallback = uniqueEvidenceLines([
      'Face evidence remains the primary driver of this forensic decision.',
      'Regional consistency and landmark geometry were cross-checked for this run.',
      'Explainability outputs were generated to support manual review.',
    ]);

    fallback.forEach((point) => {
      if (selected.length < 3) {
        selected.push(point);
      }
    });

    while (selected.length < 3) {
      selected.push('Manual review remains advisable when facial evidence is limited.');
    }

    return selected.slice(0, 3);
  }

  function describeConfidence(score) {
    if (score >= 82) {
      return { label: 'Very High Signal', note: 'Dense forensic evidence supports this classification.' };
    }
    if (score >= 68) {
      return { label: 'High Signal', note: 'Multiple evidence streams are aligned in the same direction.' };
    }
    if (score >= 52) {
      return { label: 'Review Needed', note: 'The result is actionable, with moderate forensic support behind it.' };
    }
    if (score >= 35) {
      return { label: 'Mixed Signal', note: 'Some cues are present, though the evidence remains more limited.' };
    }
    return { label: 'Low Signal', note: 'Only light forensic support was observed in this run.' };
  }

  function buildForensicsHighlights(faceForensics) {
    const topRegion = Array.isArray(faceForensics.top_regions) && faceForensics.top_regions.length
      ? faceForensics.top_regions[0]
      : null;
    const authenticity = Number(faceForensics.face_authenticity_score || 0);
    const primaryPoint = topRegion
      ? `Primary focus: ${topRegion.label} scored ${Number(topRegion.score || 0).toFixed(1)}%. Face authenticity is ${authenticity.toFixed(1)}% for this scan.`
      : `Face authenticity is ${authenticity.toFixed(1)}% for this scan.`;

    let reliabilityPoint = 'Landmark and crop quality checks stayed available for this run.';
    if (!faceForensics.face_detected) {
      reliabilityPoint = 'Guided face alignment was used, so treat region evidence as review support.';
    } else if (faceForensics.landmark_integrity?.fallback) {
      reliabilityPoint = 'Guided landmark anchors stayed active for the eye, mouth, and contour checks.';
    } else if (faceForensics.face_quality?.quality_score && Number(faceForensics.face_quality.quality_score) < 45) {
      reliabilityPoint = 'The face crop is usable, but lower image quality reduces how strongly each region can be trusted.';
    }

    return [primaryPoint, reliabilityPoint];
  }
  
  function validateFile(file) {
    const ext = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : '';
    if (!allowed.includes(ext)) return 'Unsupported file type. Choose JPG, PNG, MP4, MOV.';
    return null;
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function clearSelectedMedia() {
    selectedFile = null;
    if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = null;
    fileInput.value = '';
    preview.innerHTML = '';
    preview.classList.add('hidden');
    analyzeBtn.disabled = true;
    setError('');
  }

  function renderPreview(file) {
    preview.innerHTML = '';
    if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = URL.createObjectURL(file);
    const isVideo = ['mp4', 'mov', 'avi'].some(ext => file.name.toLowerCase().endsWith(ext));
    const size = formatFileSize(file.size);

    preview.innerHTML = `
      <div class="preview-card" style="background: var(--panel); border: 1px solid var(--border); box-shadow: var(--card-shadow); border-radius: 14px; overflow: hidden;">
        <div class="preview-media-container" style="background: #000; overflow: hidden; border-bottom: 1px solid var(--border);">
          ${isVideo ? `<video src="${currentPreviewUrl}" controls style="width:100%; max-height: 400px;"></video>` : `<img src="${currentPreviewUrl}" style="width:100%; max-height: 400px; object-fit: contain;">`}
        </div>
        <div style="padding: 1.2rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 800; font-size: 0.95rem; color: var(--text);">${file.name}</div>
                <div style="font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem;">${size} | ${isVideo ? 'VIDEO' : 'IMAGE'}</div>
            </div>
            <button id="remove-media-btn" class="remove-btn">Remove</button>
        </div>
      </div>
    `;
    preview.classList.remove('hidden');
    document.getElementById('remove-media-btn').addEventListener('click', clearSelectedMedia);
    preview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function buildXAICard(report, index) {
    const unavailable = report.status !== 'available';
    const titleTone = unavailable ? 'var(--amber)' : 'var(--cyan)';
    const statusPill = unavailable
      ? '<span class="xai-status-pill unavailable">Unavailable</span>'
      : '<span class="xai-status-pill available">Available</span>';
    const isBasic = report.basic ? 'true' : 'false';

    return `
      <article class="xai-report-card" data-index="${index}" data-basic="${isBasic}">
        <div class="xai-report-head">
          <div>
            <div class="xai-report-method" style="color: ${titleTone};">${escapeHtml(report.method)}</div>
            <div class="xai-report-mode">${report.basic ? 'Basic View' : 'Advanced View'}</div>
          </div>
          ${statusPill}
        </div>
        <div class="xai-report-media">
          <img src="/${report.image_url}" alt="${escapeHtml(report.method)}" class="xai-report-image">
        </div>
        <div class="xai-report-copy-wrap">
          <p class="xai-report-copy">
            ${escapeHtml(unavailable ? 'This XAI output could not be generated for this run.' : (report.short_explanation || report.description || report.message))}
          </p>
        </div>
      </article>
    `;
  }

  function buildFaceRegionCard(region) {
    const score = Number(region.score || 0);
    const tone = score >= 70 ? 'var(--red)' : (score >= 45 ? 'var(--amber)' : 'var(--green)');
    const status = region.status || (score >= 70 ? 'Suspicious' : (score >= 45 ? 'Review' : 'Stable'));
    return `
      <article class="face-region-card" data-region-key="${escapeHtml(region.key || '')}">
        <div class="face-region-card-head">
          <div>
            <div class="face-region-name">${escapeHtml(region.label)}</div>
            <div class="face-region-focus">${escapeHtml(region.focus || '')}</div>
          </div>
          <div class="face-region-chip" style="color:${tone}; border-color:${tone};">${escapeHtml(status)}</div>
        </div>
        <div class="face-region-score-row">
          <div class="face-region-score" style="color:${tone};">${score.toFixed(1)}%</div>
          <div class="face-region-weight">Weight ${Number(region.importance || 0).toFixed(2)}</div>
        </div>
        <div class="face-region-bar"><span style="width:${Math.min(score, 100)}%; background:${tone}; color:${tone};"></span></div>
        ${region.image_url ? `<img src="/${region.image_url}" alt="${escapeHtml(region.label)}" class="face-region-thumb">` : ''}
        <p class="face-region-copy">${escapeHtml(region.explanation || '')}</p>
      </article>
    `;
  }

  function buildFaceSwapMetricCard(label, value, copy, tone = 'var(--cyan)') {
    return `
      <article class="faceswap-metric-card">
        <div class="faceswap-metric-label">${escapeHtml(label)}</div>
        <div class="faceswap-metric-value" style="color:${tone};">${escapeHtml(value)}</div>
        <p class="faceswap-metric-copy">${escapeHtml(copy || '')}</p>
      </article>
    `;
  }

  function buildFaceSwapArtifactCard(label, caption, imageUrl) {
    if (!imageUrl) {
      return `
        <article class="faceswap-artifact-card">
          <div class="faceswap-artifact-head">
            <div class="faceswap-artifact-label">${escapeHtml(label)}</div>
            <div class="faceswap-artifact-caption">${escapeHtml(caption)}</div>
          </div>
          <div class="faceswap-artifact-empty">This face-swap output could not be generated for this run.</div>
        </article>
      `;
    }
    return `
      <article class="faceswap-artifact-card">
        <div class="faceswap-artifact-head">
          <div class="faceswap-artifact-label">${escapeHtml(label)}</div>
          <div class="faceswap-artifact-caption">${escapeHtml(caption)}</div>
        </div>
        <img src="/${imageUrl}" alt="${escapeHtml(label)}" class="faceswap-artifact-image">
      </article>
    `;
  }

  function applyXAIView(view) {
    currentXAIView = view;
    const cards = document.querySelectorAll('.xai-report-card');
    const basicBtn = document.getElementById('xai-basic-btn');
    const advBtn = document.getElementById('xai-adv-btn');

    cards.forEach((card) => {
      const isBasic = card.dataset.basic === 'true';
      const keepVisible = view === 'advanced' ? true : isBasic;
      card.classList.toggle('hidden', !keepVisible);
    });

    if (basicBtn && advBtn) {
      basicBtn.style.background = view === 'basic' ? 'var(--cyan)' : 'transparent';
      basicBtn.style.color = view === 'basic' ? '#03111d' : 'var(--muted)';
      advBtn.style.background = view === 'advanced' ? 'var(--cyan)' : 'transparent';
      advBtn.style.color = view === 'advanced' ? '#03111d' : 'var(--muted)';
    }
  }

  function applyRegionView(view) {
    currentRegionView = view;
    const regionGrid = document.getElementById('forensics-region-grid');
    const regionCards = document.querySelectorAll('.face-region-card');
    const regionSelect = document.getElementById('region-view-select');

    regionCards.forEach((card) => {
      const showCard = view === 'all' || card.dataset.regionKey === view;
      card.classList.toggle('hidden', !showCard);
    });

    regionGrid?.classList.toggle('forensics-region-grid--single', view !== 'all');
    if (regionSelect && regionSelect.value !== view) {
      regionSelect.value = view;
    }
  }

  function setFocusRegionVisibility(expanded) {
    focusRegionExpanded = expanded;
    const focusView = document.getElementById('forensics-focus-view');
    const focusToggle = document.getElementById('focus-region-toggle');

    focusView?.classList.toggle('hidden', !expanded);
    if (focusToggle) {
      focusToggle.textContent = expanded ? 'Hide Focus Region View' : 'Show Focus Region View';
      focusToggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    }
  }

  // --- RESULT RENDERING (Task 1 & 2) ---
  function renderResult(result) {
    resultEmpty.classList.add('hidden');
    resultCard.classList.remove('hidden');

    const verdict = String(result.verdict || '').toUpperCase();
    const reviewNote = result.review_note || '';
    const evidenceStrength = result.evidence_strength || 'moderate';
    let statusColor = verdict === 'DEEPFAKE' ? 'var(--red)' : 'var(--green)';
    const evidenceTone = evidenceStrength === 'strong' ? statusColor : (evidenceStrength === 'good' ? 'var(--cyan)' : 'var(--muted)');

    const previewImg = currentPreviewUrl || `/${result.heatmap_url}`;
    const xaiReports = (Array.isArray(result.xai_advanced_reports) && result.xai_advanced_reports.length
      ? result.xai_advanced_reports
      : (Array.isArray(result.xai_reports) ? result.xai_reports : [])).slice(0, 6);
    const xaiContext = result.xai_context || {};
    const faceForensics = result.face_forensics || {};
    const faceswapAnalysis = result.faceswap_analysis || {};
    const faceswapArtifacts = faceswapAnalysis.artifacts || {};
    const modelStatus = String(result.model_status || 'heuristic_only');
    const referenceDatasets = Array.isArray(result.reference_datasets) ? result.reference_datasets : [];
    const calibrationMode = String(result.calibration_mode || 'labeled_reference');
    let confidenceScore = Math.min(100, Math.max(0, Number(result.confidence) || 0));
    if (verdict === 'REAL' && confidenceScore < 50) {
      confidenceScore = 100 - confidenceScore;
    }
    const confidenceProfile = describeConfidence(confidenceScore);
    const runtimeLabel = modelStatus === 'checkpoint_loaded' ? 'Checkpoint-loaded model' : 'Heuristic-only model';
    const runtimeNote = modelStatus === 'checkpoint_loaded'
      ? 'A trained checkpoint contributed to this verdict.'
      : 'No trained checkpoint was loaded, so this result is transparently marked as heuristic-plus-calibration.';
    const calibrationSummary = referenceDatasets.length
      ? `Calibration: ${calibrationMode.replaceAll('_', ' ')} using ${referenceDatasets.join(', ')}.`
      : 'Calibration datasets were not available for this run.';
    const keyEvidencePoints = buildKeyEvidencePoints(result, faceForensics, faceswapAnalysis);
    const forensicsHighlights = buildForensicsHighlights(faceForensics);
    const xaiGalleryHTML = xaiReports.map(buildXAICard).join('');
    const xaiMetaLine = result.file_type === 'VIDEO'
      ? `
          <div class="xai-meta-chip">Frame ${escapeHtml(xaiContext.frame_number ?? 'N/A')}</div>
          <div class="xai-meta-chip">Timestamp ${escapeHtml(xaiContext.timestamp_label ?? 'N/A')}</div>
        `
      : '<div class="xai-meta-chip">Primary image frame</div>';
    const suspiciousFrameHTML = Array.isArray(result.suspicious_frames) && result.suspicious_frames.length
      ? `
          <div class="panel" style="margin: 0;">
            <h3 style="color: var(--cyan); font-size: 0.8rem; text-transform: uppercase;">Key Suspicious Frames</h3>
            <div style="display: grid; gap: 0.75rem;">
              ${result.suspicious_frames.map((frame) => `
                <div style="display: flex; justify-content: space-between; gap: 1rem; padding: 0.9rem 1rem; border: 1px solid var(--border); border-radius: 12px; background: rgba(var(--panel-rgb), 0.18);">
                  <div>
                    <div style="font-weight: 700; color: var(--text);">Frame ${escapeHtml(frame.frame_number)}</div>
                    <div style="font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem;">${escapeHtml(frame.timestamp)} • ${escapeHtml(frame.explanation)}</div>
                  </div>
                  <div style="font-weight: 800; color: var(--cyan);">${Number(frame.confidence || 0).toFixed(1)}%</div>
                </div>
              `).join('')}
            </div>
          </div>
        `
      : '';
    const faceswapScore = Number(faceswapAnalysis.faceswap_score || 0);
    const faceswapTone = faceswapScore >= 62 ? 'var(--red)' : (faceswapScore >= 42 ? 'var(--amber)' : 'var(--cyan)');
    const faceswapSummary = faceswapAnalysis.summary || 'This face-swap output could not be generated for this run.';
    const faceswapExplanationList = Array.isArray(faceswapAnalysis.explanations) && faceswapAnalysis.explanations.length
      ? faceswapAnalysis.explanations.slice(0, 3)
      : ['This face-swap output could not be generated for this run.'];
    const faceswapRegionList = Array.isArray(faceswapAnalysis.suspicious_regions) && faceswapAnalysis.suspicious_regions.length
      ? faceswapAnalysis.suspicious_regions
      : [];
    const strongestFaceSwapFrame = faceswapAnalysis.strongest_frame || result.strongest_frame || null;
    const faceswapArtifactsAvailable = [
      ['Identity Overlay', 'Regional identity drift across the aligned face.', faceswapArtifacts.identity_overlay_url],
      ['Boundary Overlay', 'Forehead, cheek, and jawline seam analysis.', faceswapArtifacts.boundary_overlay_url],
      ['Landmark Overlay', 'Geometry integrity evidence for face-swap detection.', faceswapArtifacts.landmark_overlay_url],
    ].filter((artifact) => artifact[2]);
    const faceswapArtifactSection = faceswapArtifactsAvailable.length
      ? `
          <div class="faceswap-artifact-grid">
            ${faceswapArtifactsAvailable.map(([label, caption, imageUrl]) => buildFaceSwapArtifactCard(label, caption, imageUrl)).join('')}
          </div>
        `
      : '';
    const faceSwapScoreNote = strongestFaceSwapFrame
      ? `Frame ${strongestFaceSwapFrame.frame_number || 'N/A'} at ${strongestFaceSwapFrame.timestamp || 'N/A'}`
      : (faceswapAnalysis.available === false ? 'Limited dedicated coverage' : 'Dedicated signal');
    const regionOptions = Array.isArray(faceForensics.region_grid_ordered) ? faceForensics.region_grid_ordered : [];
    const defaultRegionView = (Array.isArray(faceForensics.top_regions) && faceForensics.top_regions[0]?.key && regionOptions.some((region) => region.key === faceForensics.top_regions[0].key))
      ? faceForensics.top_regions[0].key
      : 'all';
    currentRegionView = defaultRegionView;
    focusRegionExpanded = false;
    const faceSwapSection = `
      <section class="faceswap-shell">
        <div class="faceswap-header">
          <div>
            <div class="faceswap-kicker">Dedicated Manipulation Typing</div>
            <h3 class="faceswap-title">Face Swap Forensic Evidence</h3>
            <p class="faceswap-subtitle">Identity consistency, boundary blending, landmark mismatch, and regional facial evidence fused for dedicated face-swap detection.</p>
          </div>
          <div class="faceswap-score-badge">
            <span>Face Swap Score</span>
            <strong style="color:${faceswapTone};">${faceswapScore.toFixed(1)}%</strong>
            <small>${escapeHtml(faceSwapScoreNote)}</small>
          </div>
        </div>
        <div class="faceswap-summary-card">
          <div class="faceswap-summary-label">Summary</div>
          <div class="faceswap-summary-copy">${escapeHtml(faceswapSummary)}</div>
        </div>
        <div class="faceswap-metric-grid">
          ${buildFaceSwapMetricCard('Identity Consistency', `${Number(faceswapAnalysis.identity_consistency_score || 0).toFixed(1)}%`, faceswapAnalysis.embedding_source || 'Embedding source unavailable.')}
          ${buildFaceSwapMetricCard('Boundary Anomaly', `${Number(faceswapAnalysis.boundary_anomaly_score || 0).toFixed(1)}%`, 'Jawline, forehead, and cheek seam continuity.')}
          ${buildFaceSwapMetricCard('Landmark Mismatch', `${Number(faceswapAnalysis.landmark_mismatch_score || 0).toFixed(1)}%`, 'Eye, mouth, symmetry, and contour geometry checks.')}
          ${buildFaceSwapMetricCard('Texture Mismatch', `${Number(faceswapAnalysis.texture_mismatch_score || 0).toFixed(1)}%`, 'Central-face versus outer-boundary texture comparison.')}
          ${buildFaceSwapMetricCard('Region Consensus', `${Number(faceswapAnalysis.region_consensus_score || 0).toFixed(1)}%`, 'Agreement between the most suspicious face zones.')}
          ${buildFaceSwapMetricCard('Temporal Drift', `${Number(faceswapAnalysis.temporal_identity_drift_score || 0).toFixed(1)}%`, strongestFaceSwapFrame ? `Strongest frame: ${strongestFaceSwapFrame.frame_number || 'N/A'} at ${strongestFaceSwapFrame.timestamp || 'N/A'}` : 'Temporal identity drift was not available.')}
        </div>
        ${faceswapArtifactSection}
        <div class="faceswap-bottom-grid">
          <article class="faceswap-list-card">
            <div class="faceswap-list-label">Suspicious Face Regions</div>
            ${
              faceswapRegionList.length
                ? `<div class="faceswap-region-list">${faceswapRegionList.map((region) => `
                    <div class="faceswap-region-pill">
                      <strong>${escapeHtml(region.label)}</strong>
                      <span>${Number(region.score || 0).toFixed(1)}%</span>
                      <small>${escapeHtml(region.explanation || '')}</small>
                    </div>
                  `).join('')}</div>`
                : '<div class="faceswap-empty-copy">No specific face region was strongly flagged for face swapping in this run.</div>'
            }
          </article>
          <article class="faceswap-list-card">
            <div class="faceswap-list-label">Analyst Notes</div>
            <ul class="faceswap-explanation-list">
              ${faceswapExplanationList.map((entry) => `<li>${escapeHtml(entry)}</li>`).join('')}
            </ul>
            ${
              strongestFaceSwapFrame
                ? `<div class="faceswap-strongest-frame">Strongest frame: Frame ${escapeHtml(strongestFaceSwapFrame.frame_number || 'N/A')} at ${escapeHtml(strongestFaceSwapFrame.timestamp || 'N/A')}</div>`
                : '<div class="faceswap-strongest-frame">Strongest frame metadata not available for this run.</div>'
            }
          </article>
        </div>
      </section>
    `;
    const facialRegionSection = faceForensics && faceForensics.region_grid_ordered && faceForensics.region_grid_ordered.length
      ? `
          <section class="facial-forensics-shell">
            <div class="facial-forensics-header">
              <div>
                <div class="forensics-kicker">Face-Centric Forensics</div>
                <h3 class="forensics-title">Facial Region Forensic Analysis</h3>
                <p class="forensics-subtitle">Face-aligned regional forensic analysis with landmark-guided anomaly scoring.</p>
              </div>
              <div class="forensics-score-badge">
                <span>Face Authenticity Score</span>
                <strong>${Number(faceForensics.face_authenticity_score || 0).toFixed(1)}%</strong>
              </div>
            </div>
            <div class="forensics-insight-grid">
              ${forensicsHighlights.map((point, index) => `
                <article class="forensics-insight-card">
                  <div class="forensics-insight-label">Point ${index + 1}</div>
                  <strong class="forensics-insight-copy">${escapeHtml(point)}</strong>
                </article>
              `).join('')}
            </div>
            <div class="forensics-focus-toggle-row">
              <button type="button" id="focus-region-toggle" class="forensics-focus-toggle" aria-expanded="false">Show Focus Region View</button>
            </div>
            <div class="forensics-evidence-grid">
              <article class="forensics-evidence-card">
                <div class="forensics-evidence-head">
                  <div>
                    <div class="forensics-evidence-label">Aligned Face</div>
                    <div class="forensics-evidence-caption">Normalized facial crop for region comparison.</div>
                  </div>
                </div>
                <img src="/${faceForensics.aligned_face_url}" alt="Aligned face" class="forensics-evidence-image">
              </article>
              <article class="forensics-evidence-card">
                <div class="forensics-evidence-head">
                  <div>
                    <div class="forensics-evidence-label">3x3 Region Overlay</div>
                    <div class="forensics-evidence-caption">Regional anomaly map across the aligned face.</div>
                  </div>
                </div>
                <img src="/${faceForensics.grid_overlay_url}" alt="Face grid overlay" class="forensics-evidence-image">
              </article>
              <article class="forensics-evidence-card">
                <div class="forensics-evidence-head">
                  <div>
                    <div class="forensics-evidence-label">Landmark Overlay</div>
                    <div class="forensics-evidence-caption">Geometry anchors for eye, mouth, and contour checks.</div>
                  </div>
                </div>
                <img src="/${faceForensics.landmark_overlay_url}" alt="Landmark overlay" class="forensics-evidence-image">
                <div class="forensics-evidence-note">${faceForensics.face_detected ? `Detector: ${escapeHtml(faceForensics.detector || 'N/A')}` : escapeHtml(faceForensics.fallback_reason || 'Fallback crop used.')}</div>
              </article>
            </div>
            <div id="forensics-focus-view" class="forensics-focus-view hidden">
              <div class="forensics-focus-controls">
                <label for="region-view-select" class="forensics-region-toolbar-label">Region</label>
                <select id="region-view-select" class="region-view-select">
                  <option value="all">All Regions</option>
                  ${regionOptions.map((region) => `<option value="${escapeHtml(region.key || '')}">${escapeHtml(region.label)}</option>`).join('')}
                </select>
              </div>
              <div id="forensics-region-grid" class="forensics-region-grid">
                ${faceForensics.region_grid_ordered.map(buildFaceRegionCard).join('')}
              </div>
            </div>
          </section>
        `
      : '';
    const xaiFocusPoint = xaiContext.selection_reason || 'Explainability outputs highlight the strongest anomaly concentration from this analysis run.';

    resultCard.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
            <div class="panel" style="margin:0;">
                <h3 style="color: var(--cyan); font-size: 0.8rem; text-transform: uppercase;">Analyzed Media</h3>
                <div style="width:100%; aspect-ratio:16/9; background:#000; border-radius: 8px; overflow:hidden;">
                    ${result.file_type === 'VIDEO' ? `<video src="${previewImg}" controls style="width:100%; height:100%;"></video>` : `<img src="${previewImg}" style="width:100%; height:100%; object-fit:contain;">`}
                </div>
            </div>
            <div class="panel" style="margin:0;">
                <h3 style="color: ${statusColor}; font-size: 0.8rem; text-transform: uppercase;">Forensic Attention</h3>
                <div style="width:100%; aspect-ratio:16/9; background:#000; border-radius: 8px; overflow:hidden;">
                    <img src="/${result.heatmap_url}" style="width:100%; height:100%; object-fit:contain;">
                </div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem;">
            <div class="panel verdict-panel" style="margin:0;">
                <div>
                    <div class="verdict-eyebrow">Verdict</div>
                    <div class="verdict-title" style="color: ${statusColor};">${verdict}</div>
                    <div class="verdict-strength" style="color: ${evidenceTone};">${escapeHtml(evidenceStrength)} evidence</div>
                    <div class="verdict-review-note">${escapeHtml(runtimeLabel)}</div>
                    ${reviewNote ? `<div class="verdict-review-note">${escapeHtml(reviewNote)}</div>` : ''}
                    <div class="verdict-copy">${escapeHtml(result.explanation || '')}</div>
                </div>
                <div class="confidence-spotlight">
                    <div class="confidence-spotlight-head">
                        <div>
                            <div class="confidence-label">Confidence Level</div>
                            <div class="confidence-band">${escapeHtml(confidenceProfile.label)}</div>
                        </div>
                        <div class="confidence-value" style="color: ${statusColor};">${confidenceScore.toFixed(1)}%</div>
                    </div>
                    <div class="confidence-meter-track">
                        <div class="confidence-meter-fill" style="width: ${confidenceScore}%; background: ${statusColor}; color: ${statusColor};"></div>
                        <div class="confidence-meter-marker" style="left: clamp(10px, ${confidenceScore}%, calc(100% - 10px)); border-color: ${statusColor}; color: ${statusColor};"></div>
                    </div>
                    <div class="confidence-scale">
                        <span>Low</span>
                        <span>Review</span>
                        <span>High</span>
                    </div>
                    <div class="confidence-caption">${escapeHtml(confidenceProfile.note)}</div>
                    <div class="confidence-caption" style="margin-top:0.65rem;">${escapeHtml(runtimeNote)}</div>
                </div>
            </div>
            <div class="panel evidence-panel" style="margin:0;">
                <h3 class="evidence-panel-title">Evidentiary Details</h3>
                <div style="padding: 0.8rem 0.95rem; margin-bottom: 1rem; border: 1px solid var(--border); border-radius: 12px; background: rgba(var(--panel-rgb), 0.18);">
                    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08rem; color: var(--cyan);">Runtime & Calibration</div>
                    <div style="font-size: 0.9rem; color: var(--text); margin-top: 0.35rem; font-weight: 700;">${escapeHtml(runtimeLabel)}</div>
                    <div style="font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem;">${escapeHtml(calibrationSummary)}</div>
                </div>
                <div class="evidence-point-grid">
                    ${keyEvidencePoints.map((point, index) => `
                      <article class="evidence-point-card">
                        <span class="evidence-point-index">${String(index + 1).padStart(2, '0')}</span>
                        <strong class="evidence-point-text">${escapeHtml(point)}</strong>
                      </article>
                    `).join('')}
                </div>
            </div>
            ${suspiciousFrameHTML}
        </div>

        ${faceSwapSection}
        ${facialRegionSection}
        <div class="panel xai-shell" style="margin-top: 2rem;">
            <div class="xai-shell-head">
                <div>
                    <div class="xai-shell-title-row">
                        <div class="xai-shell-dot"></div>
                        <h3 class="xai-shell-title">Explainable AI Evidence Matrix</h3>
                    </div>
                    <div class="xai-meta-row">
                      ${xaiMetaLine}
                    </div>
                </div>
                <div class="xai-toggle-wrap">
                    <button id="xai-basic-btn" class="xai-toggle-btn">BASIC</button>
                    <button id="xai-adv-btn" class="xai-toggle-btn">ADVANCED</button>
                </div>
            </div>
            <div class="xai-focus-point">
              <strong>${escapeHtml(xaiFocusPoint)}</strong>
            </div>
            <div id="xai-gallery" class="xai-gallery">
                ${xaiGalleryHTML}
            </div>
        </div>
    `;

    document.getElementById('xai-basic-btn')?.addEventListener('click', () => applyXAIView('basic'));
    document.getElementById('xai-adv-btn')?.addEventListener('click', () => applyXAIView('advanced'));
    document.getElementById('focus-region-toggle')?.addEventListener('click', () => setFocusRegionVisibility(!focusRegionExpanded));
    document.getElementById('region-view-select')?.addEventListener('change', (event) => applyRegionView(event.target.value || 'all'));
    applyXAIView(currentXAIView);
    applyRegionView(currentRegionView);
    setFocusRegionVisibility(focusRegionExpanded);
    resultCard.scrollIntoView({ behavior: 'smooth' });
  }

  // --- ANALYSIS LOGIC ---
  async function analyze() {
    if (!selectedFile) return;
    
    setError('');
    resultCard.classList.add('hidden');
    resultEmpty.classList.remove('hidden');
    processingOverlay.classList.remove('hidden');
    
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const jobId = 'JOB-' + Math.random().toString(36).substr(2, 9).toUpperCase();
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mode', mode);
    formData.append('job_id', jobId);

    // Initial Progress
    progressPercent.textContent = '10%';
    dynamicProgress.style.width = '10%';
    progressStatus.textContent = '> Initializing neural uplink...';

    const poll = setInterval(async () => {
        try {
            const r = await fetch(`/api/progress/${jobId}`);
            const d = await r.json();
            if (d.success && d.progress) {
                progressPercent.textContent = d.progress.percentage + '%';
                dynamicProgress.style.width = d.progress.percentage + '%';
                progressStatus.textContent = '> ' + d.progress.message;
                if (d.progress.complete) clearInterval(poll);
            }
        } catch (e) {}
    }, 600);

    try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        clearInterval(poll);

        if (data.success) {
            progressPercent.textContent = '100%';
            dynamicProgress.style.width = '100%';
            progressStatus.textContent = '> Scan Matrix Complete.';
            currentXAIView = 'basic';
            setTimeout(() => {
                processingOverlay.classList.add('hidden');
                renderResult(data.result);
            }, 500);
        } else {
            setError(data.error || 'Deep analysis failed.');
            processingOverlay.classList.add('hidden');
        }
    } catch (e) {
        clearInterval(poll);
        setError('Server connection lost.');
        processingOverlay.classList.add('hidden');
    }
  }

  dropZone.addEventListener('click', (e) => {
    if (e.target !== fileInput) fileInput.click();
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        const file = e.target.files[0];
        const err = validateFile(file);
        if (err) setError(err);
        else { setError(''); selectedFile = file; renderPreview(file); analyzeBtn.disabled = false; }
    }
  });

  ['dragenter', 'dragover'].forEach(e => {
    dropZone.addEventListener(e, (ev) => { ev.preventDefault(); dropZone.classList.add('drag-over'); });
  });
  ['dragleave', 'drop'].forEach(e => {
    dropZone.addEventListener(e, (ev) => { ev.preventDefault(); dropZone.classList.remove('drag-over'); });
  });
  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) {
        const file = e.dataTransfer.files[0];
        const err = validateFile(file);
        if (err) setError(err);
        else { setError(''); selectedFile = file; renderPreview(file); analyzeBtn.disabled = false; }
    }
  });

  analyzeBtn.addEventListener('click', analyze);
});
