/**
 * LiverCare AI - Frontend JavaScript Application Module
 */

document.addEventListener('DOMContentLoaded', () => {

    // ---------------------------------------------------------
    // 1. MOBILE NAVIGATION TOGGLE
    // ---------------------------------------------------------
    const hamburgerBtn = document.getElementById('hamburger-btn');
    const mobileNav = document.getElementById('mobile-nav');
    const mobileLinks = document.querySelectorAll('.mobile-link');

    if (hamburgerBtn && mobileNav) {
        hamburgerBtn.addEventListener('click', () => {
            mobileNav.classList.toggle('active');
        });

        mobileLinks.forEach(link => {
            link.addEventListener('click', () => {
                mobileNav.classList.remove('active');
            });
        });
    }

    // ---------------------------------------------------------
    // 2. TOOLTIP ACCESSIBILITY FOR MOBILE & DESKTOP
    // ---------------------------------------------------------
    const tooltipTriggers = document.querySelectorAll('.tooltip-trigger');

    tooltipTriggers.forEach(trigger => {
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            // Toggle current active state
            const isActive = trigger.classList.contains('active');
            tooltipTriggers.forEach(t => t.classList.remove('active'));
            if (!isActive) {
                trigger.classList.add('active');
            }
        });
    });

    document.addEventListener('click', () => {
        tooltipTriggers.forEach(t => t.classList.remove('active'));
    });

    // ---------------------------------------------------------
    // 3. QUICK TEST PRESETS
    // ---------------------------------------------------------
    const presetHealthyBtn = document.getElementById('preset-healthy');
    const presetHighRiskBtn = document.getElementById('preset-highrisk');
    const presetCustomBtn = document.getElementById('preset-custom');

    const form = document.getElementById('prediction-form');
    const outputSection = document.getElementById('output-section');
    const alertBanner = document.getElementById('alert-banner');

    const healthyProfile = {
        age: 25,
        gender: 'Female',
        total_bilirubin: 0.7,
        direct_bilirubin: 0.2,
        alkaline_phosphatase: 160,
        alt: 18,
        ast: 15,
        total_proteins: 7.2,
        albumin: 4.1,
        ag_ratio: 1.30
    };

    const highRiskProfile = {
        age: 62,
        gender: 'Male',
        total_bilirubin: 10.9,
        direct_bilirubin: 5.5,
        alkaline_phosphatase: 699,
        alt: 64,
        ast: 100,
        total_proteins: 7.5,
        albumin: 3.2,
        ag_ratio: 0.74
    };

    function populateForm(profile) {
        for (const [key, value] of Object.entries(profile)) {
            const input = document.getElementById(key);
            if (input) {
                input.value = value;
            }
        }
        hideAlert();
        outputSection.classList.add('hidden');
    }

    if (presetHealthyBtn) {
        presetHealthyBtn.addEventListener('click', () => {
            populateForm(healthyProfile);
            showAlert("Loaded Healthy / Low-Risk Profile into form.", "info");
            scrollToSection('prediction');
        });
    }

    if (presetHighRiskBtn) {
        presetHighRiskBtn.addEventListener('click', () => {
            populateForm(highRiskProfile);
            showAlert("Loaded High-Risk Profile into form.", "info");
            scrollToSection('prediction');
        });
    }

    if (presetCustomBtn) {
        presetCustomBtn.addEventListener('click', () => {
            if (form) form.reset();
            hideAlert();
            outputSection.classList.add('hidden');
            scrollToSection('prediction');
        });
    }

    // Smooth scroll helper
    function scrollToSection(sectionId) {
        const elem = document.getElementById(sectionId);
        if (elem) {
            elem.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function showAlert(message, type = "error") {
        if (!alertBanner) return;
        alertBanner.textContent = message;
        alertBanner.className = `alert-banner ${type === 'error' ? 'alert-error' : 'alert-info'}`;
        alertBanner.classList.remove('hidden');
    }

    function hideAlert() {
        if (alertBanner) alertBanner.classList.add('hidden');
    }

    // ---------------------------------------------------------
    // 4. PREDICTION FORM SUBMISSION (AJAX TO BACKEND ML MODEL)
    // ---------------------------------------------------------
    const btnSubmit = document.getElementById('btn-submit');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideAlert();

            const formData = new FormData(form);
            const payload = {};
            formData.forEach((val, key) => payload[key] = val);

            // UI Loading State
            btnSubmit.disabled = true;
            btnSubmit.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing Model Prediction...`;

            try {
                const response = await fetch('/predict_form', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (!response.ok || data.status === 'error') {
                    showAlert(data.error || "Prediction processing error.", "error");
                    return;
                }

                renderResult(data);

            } catch (err) {
                showAlert(`Server communication error: ${err.message}`, "error");
            } finally {
                btnSubmit.disabled = false;
                btnSubmit.innerHTML = `Predict Liver Disease`;
            }
        });
    }

    // ---------------------------------------------------------
    // 5. RENDER PREDICTION OUTPUT
    // ---------------------------------------------------------
    function renderResult(data) {
        const outputBadge = document.getElementById('output-badge');
        const riskTag = document.getElementById('risk-tag');
        const probDisplay = document.getElementById('prob-display');
        const progressBarFill = document.getElementById('progress-bar-fill');
        const explanationText = document.getElementById('output-explanation-text');

        const prob = data.probability; // e.g. 96.94
        const isPositive = data.prediction_class === 1;
        const riskCat = data.risk_category; // "High", "Moderate", "Low"

        // 1. Result Badge
        if (isPositive) {
            outputBadge.className = 'result-badge positive';
            outputBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Liver Disease Positive`;
            explanationText.textContent = "The model predicts that this patient belongs to the positive liver-disease class based on the provided clinical parameters.";
        } else {
            outputBadge.className = 'result-badge negative';
            outputBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Liver Disease Negative`;
            explanationText.textContent = "The model predicts that this patient belongs to the negative liver-disease class based on the provided clinical parameters.";
        }

        // 2. Risk Category Tag
        riskTag.textContent = riskCat.toUpperCase();
        riskTag.className = `risk-tag ${riskCat.toLowerCase()}`;

        // 3. Probability & Animated Progress Bar
        probDisplay.textContent = `${prob.toFixed(2)}%`;

        progressBarFill.className = 'progress-fill';
        if (prob > 70) {
            progressBarFill.classList.add('fill-high');
        } else if (prob >= 30) {
            progressBarFill.classList.add('fill-moderate');
        } else {
            progressBarFill.classList.add('fill-low');
        }

        // Reveal section & animate progress bar fill
        outputSection.classList.remove('hidden');
        setTimeout(() => {
            progressBarFill.style.width = `${Math.min(Math.max(prob, 5), 100)}%`;
        }, 50);

        scrollToSection('output-section');
    }

    // ---------------------------------------------------------
    // 6. RESET PREDICTION BUTTON
    // ---------------------------------------------------------
    const btnResetPrediction = document.getElementById('btn-reset-prediction');
    if (btnResetPrediction) {
        btnResetPrediction.addEventListener('click', () => {
            outputSection.classList.add('hidden');
            scrollToSection('prediction');
        });
    }

});
