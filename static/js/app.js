// LoanPredict Pro - Enhanced JavaScript Application
'use strict';

class LoanPredictApp {
    constructor() {
        this.version = '2.0.0';
        this.apiEndpoint = '/api';
        this.initialized = false;
        
        // Bind methods
        this.init = this.init.bind(this);
        this.initializeComponents = this.initializeComponents.bind(this);
        this.setupEventListeners = this.setupEventListeners.bind(this);
        
        // Auto-initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', this.init);
        } else {
            this.init();
        }
    }
    
    init() {
        if (this.initialized) return;
        
        console.log(`🚀 Inicializando LoanPredict Pro v${this.version}`);
        
        try {
            this.initializeComponents();
            this.setupEventListeners();
            this.initializeFormValidation();
            this.initializeCharts();
            this.initializeDataTables();
            this.setupProgressiveEnhancement();
            
            this.initialized = true;
            console.log('✅ LoanPredict Pro inicializado correctamente');
        } catch (error) {
            console.error('❌ Error inicializando LoanPredict Pro:', error);
            this.showNotification('Error inicializando la aplicación', 'warning');
        }
    }
    
    initializeComponents() {
        try {
            // Initialize tooltips with enhanced options
            this.initializeTooltips();
            
            // Initialize popovers
            this.initializePopovers();
            
            // Setup loading states
            this.initializeLoadingStates();
            
            // Initialize file upload components
            this.initializeFileUpload();
            
            // Setup real-time calculations (solo si estamos en la página de predict)
            if (document.getElementById('loanForm')) {
                this.initializeCalculators();
            }
        } catch (error) {
            console.error('Error en initializeComponents:', error);
        }
    }
    
    initializeTooltips() {
        try {
            const tooltipElements = document.querySelectorAll('[data-bs-toggle="tooltip"]');
            tooltipElements.forEach(element => {
                if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
                    new bootstrap.Tooltip(element, {
                        boundary: 'viewport',
                        customClass: 'custom-tooltip',
                        delay: { show: 500, hide: 100 },
                        html: true
                    });
                }
            });
        } catch (error) {
            console.error('Error inicializando tooltips:', error);
        }
    }
    
    initializePopovers() {
        try {
            const popoverElements = document.querySelectorAll('[data-bs-toggle="popover"]');
            popoverElements.forEach(element => {
                if (typeof bootstrap !== 'undefined' && bootstrap.Popover) {
                    new bootstrap.Popover(element, {
                        boundary: 'viewport',
                        customClass: 'custom-popover',
                        trigger: 'hover focus',
                        html: true
                    });
                }
            });
        } catch (error) {
            console.error('Error inicializando popovers:', error);
        }
    }
    
    initializeCalculators() {
        try {
            // Solo inicializar si estamos en la página correcta
            const applicantIncomeInput = document.getElementById('applicant_income');
            const loanAmountInput = document.getElementById('loan_amount');
            const loanTermInput = document.getElementById('loan_term');
            const coapplicantIncomeInput = document.getElementById('coapplicant_income');
            
            if (applicantIncomeInput && loanAmountInput) {
                const inputs = [applicantIncomeInput, loanAmountInput, loanTermInput, coapplicantIncomeInput].filter(input => input !== null);
                
                inputs.forEach(input => {
                    input.addEventListener('input', () => {
                        this.updateLoanCalculations();
                        this.validateLoanRatios();
                    });
                });
                
                console.log('✅ Calculadoras inicializadas');
            }
        } catch (error) {
            console.error('Error inicializando calculadoras:', error);
        }
    }
    
    initializeFormValidation() {
        try {
            const forms = document.querySelectorAll('.needs-validation');
            
            forms.forEach(form => {
                // Real-time validation
                this.setupRealtimeValidation(form);
                
                // Enhanced submit handling
                form.addEventListener('submit', (event) => {
                    if (!form.checkValidity()) {
                        event.preventDefault();
                        event.stopPropagation();
                        
                        // Focus first invalid field with smooth scroll
                        const firstInvalid = form.querySelector(':invalid');
                        if (firstInvalid) {
                            firstInvalid.scrollIntoView({ 
                                behavior: 'smooth', 
                                block: 'center' 
                            });
                            setTimeout(() => firstInvalid.focus(), 300);
                        }
                        
                        this.showNotification('Por favor, corrige los errores en el formulario', 'warning');
                    }
                    
                    form.classList.add('was-validated');
                });
            });
            
            // Setup loan form specific validation
            const loanForm = document.getElementById('loanForm');
            if (loanForm) {
                this.setupLoanFormValidation(loanForm);
            }
        } catch (error) {
            console.error('Error inicializando validación de formularios:', error);
        }
    }
    
    setupRealtimeValidation(form) {
        try {
            const inputs = form.querySelectorAll('input, select, textarea');
            
            inputs.forEach(input => {
                // Validate on blur
                input.addEventListener('blur', () => {
                    this.validateField(input);
                });
                
                // Validate on input for certain fields
                if (input.type === 'number' || input.type === 'email') {
                    input.addEventListener('input', 
                        this.debounce(() => this.validateField(input), 500)
                    );
                }
            });
        } catch (error) {
            console.error('Error en setupRealtimeValidation:', error);
        }
    }
    
    validateField(input) {
        try {
            const isValid = input.checkValidity();
            
            // Remove existing feedback
            const existingFeedback = input.parentNode.querySelector('.invalid-feedback, .valid-feedback');
            if (existingFeedback) {
                existingFeedback.remove();
            }
            
            // Add appropriate classes
            input.classList.remove('is-valid', 'is-invalid');
            input.classList.add(isValid ? 'is-valid' : 'is-invalid');
            
            // Add custom feedback for specific validations
            if (!isValid) {
                const feedback = document.createElement('div');
                feedback.className = 'invalid-feedback';
                feedback.textContent = this.getCustomValidationMessage(input);
                input.parentNode.appendChild(feedback);
            } else if (input.value.trim() !== '') {
                const feedback = document.createElement('div');
                feedback.className = 'valid-feedback';
                feedback.innerHTML = '<i class="fas fa-check"></i> Válido';
                input.parentNode.appendChild(feedback);
            }
            
            return isValid;
        } catch (error) {
            console.error('Error validando campo:', error);
            return false;
        }
    }
    
    getCustomValidationMessage(input) {
        if (input.validity.valueMissing) {
            return 'Este campo es obligatorio';
        }
        if (input.validity.typeMismatch) {
            return 'Por favor, ingresa un valor válido';
        }
        if (input.validity.rangeUnderflow) {
            return `El valor debe ser mayor o igual a ${input.min}`;
        }
        if (input.validity.rangeOverflow) {
            return `El valor debe ser menor o igual a ${input.max}`;
        }
        if (input.validity.patternMismatch) {
            return 'El formato ingresado no es válido';
        }
        return input.validationMessage;
    }
    
    setupLoanFormValidation(form) {
        try {
            const applicantIncome = document.getElementById('applicant_income');
            const coapplicantIncome = document.getElementById('coapplicant_income');
            const loanAmount = document.getElementById('loan_amount');
            const loanTerm = document.getElementById('loan_term');
            
            if (applicantIncome && loanAmount) {
                [applicantIncome, coapplicantIncome, loanAmount, loanTerm].forEach(input => {
                    if (input) {
                        input.addEventListener('input', () => {
                            this.updateLoanCalculations();
                            this.validateLoanRatios();
                        });
                    }
                });
            }
        } catch (error) {
            console.error('Error en setupLoanFormValidation:', error);
        }
    }
    
    updateLoanCalculations() {
        try {
            const applicantIncome = parseFloat(document.getElementById('applicant_income')?.value) || 0;
            const coapplicantIncome = parseFloat(document.getElementById('coapplicant_income')?.value) || 0;
            const loanAmount = parseFloat(document.getElementById('loan_amount')?.value) || 0;
            const loanTerm = parseFloat(document.getElementById('loan_term')?.value) || 360;
            
            if (applicantIncome > 0 && loanAmount > 0) {
                const totalIncome = applicantIncome + coapplicantIncome;
                const debtRatio = loanAmount / applicantIncome;
                const monthlyPayment = loanAmount / loanTerm;
                const paymentRatio = (monthlyPayment / applicantIncome) * 100;
                
                // Update UI with calculations
                this.displayCalculations({
                    totalIncome,
                    debtRatio,
                    monthlyPayment,
                    paymentRatio
                });
            }
        } catch (error) {
            console.error('Error actualizando cálculos:', error);
        }
    }
    
    displayCalculations(calculations) {
        try {
            let calculationsDiv = document.getElementById('calculations');
            let debtRatioInfo = document.getElementById('debt-ratio-info');
            
            if (!calculationsDiv || !debtRatioInfo) return;
            
            const { totalIncome, debtRatio, monthlyPayment, paymentRatio } = calculations;
            
            // Determine risk level
            let ratioClass = 'text-success';
            let ratioText = 'Excelente';
            let riskIcon = 'fa-check-circle';
            
            if (debtRatio > 5) {
                ratioClass = 'text-danger';
                ratioText = 'Alto riesgo';
                riskIcon = 'fa-exclamation-triangle';
            } else if (debtRatio > 3) {
                ratioClass = 'text-warning';
                ratioText = 'Moderado';
                riskIcon = 'fa-exclamation-circle';
            }
            
            calculationsDiv.innerHTML = `
                <div class="row g-3">
                    <div class="col-md-3">
                        <div class="calculation-item">
                            <i class="fas fa-users text-primary mb-1"></i>
                            <small class="d-block text-muted">Ingreso Total</small>
                            <span class="fw-bold">$${this.formatNumber(totalIncome)}</span>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="calculation-item">
                            <i class="fas ${riskIcon} ${ratioClass} mb-1"></i>
                            <small class="d-block text-muted">Ratio Deuda/Ingreso</small>
                            <span class="${ratioClass} fw-bold">${debtRatio.toFixed(2)} (${ratioText})</span>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="calculation-item">
                            <i class="fas fa-calendar-alt text-info mb-1"></i>
                            <small class="d-block text-muted">Pago Mensual Est.</small>
                            <span class="fw-bold">$${this.formatNumber(monthlyPayment)}</span>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="calculation-item">
                            <i class="fas fa-percentage text-warning mb-1"></i>
                            <small class="d-block text-muted">% del Ingreso</small>
                            <span class="fw-bold">${paymentRatio.toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            `;
            
            debtRatioInfo.style.display = 'block';
            debtRatioInfo.className = `alert alert-${debtRatio > 5 ? 'danger' : debtRatio > 3 ? 'warning' : 'success'}`;
        } catch (error) {
            console.error('Error mostrando cálculos:', error);
        }
    }
    
    validateLoanRatios() {
        try {
            const applicantIncome = parseFloat(document.getElementById('applicant_income')?.value) || 0;
            const loanAmount = parseFloat(document.getElementById('loan_amount')?.value) || 0;
            
            if (applicantIncome > 0 && loanAmount > 0) {
                const debtRatio = loanAmount / applicantIncome;
                
                // Add custom validation
                const loanAmountInput = document.getElementById('loan_amount');
                if (loanAmountInput) {
                    if (debtRatio > 10) {
                        loanAmountInput.setCustomValidity('El ratio deuda-ingreso es excesivamente alto');
                    } else {
                        loanAmountInput.setCustomValidity('');
                    }
                }
            }
        } catch (error) {
            console.error('Error validando ratios:', error);
        }
    }
    
    initializeCharts() {
        try {
            // Enhanced chart configuration
            this.chartConfig = {
                responsive: true,
                displayModeBar: false,
                displaylogo: false,
                modeBarButtonsToRemove: [
                    'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
                    'hoverClosestCartesian', 'hoverCompareCartesian'
                ],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'loan_prediction_chart',
                    height: 500,
                    width: 700,
                    scale: 1
                }
            };
            
            // Apply enhanced styling to existing charts
            const charts = document.querySelectorAll('[id$="Chart"]');
            charts.forEach(chart => {
                if (chart.innerHTML.trim()) {
                    // Add resize observer for responsive charts
                    this.observeChartResize(chart);
                }
            });
        } catch (error) {
            console.error('Error inicializando gráficos:', error);
        }
    }
    
    observeChartResize(chartElement) {
        try {
            if (!window.ResizeObserver) return;
            
            const resizeObserver = new ResizeObserver(entries => {
                for (let entry of entries) {
                    if (entry.target === chartElement) {
                        // Debounce resize
                        clearTimeout(chartElement._resizeTimeout);
                        chartElement._resizeTimeout = setTimeout(() => {
                            if (window.Plotly) {
                                Plotly.Plots.resize(chartElement);
                            }
                        }, 100);
                    }
                }
            });
            
            resizeObserver.observe(chartElement);
        } catch (error) {
            console.error('Error observando resize de gráfico:', error);
        }
    }
    
    initializeDataTables() {
        try {
            const tables = document.querySelectorAll('table[id$="Table"]');
            
            tables.forEach(table => {
                if (typeof $ !== 'undefined' && $.fn.DataTable && !$.fn.DataTable.isDataTable(table)) {
                    const config = this.getDataTableConfig(table);
                    $(table).DataTable(config);
                }
            });
        } catch (error) {
            console.error('Error inicializando DataTables:', error);
        }
    }
    
    getDataTableConfig(table) {
        const baseConfig = {
            responsive: true,
            pageLength: 25,
            lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "Todos"]],
            order: [[0, 'asc']],
            language: {
                url: '//cdn.datatables.net/plug-ins/1.13.7/i18n/es-ES.json'
            },
            dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                 '<"row"<"col-sm-12"tr>>' +
                 '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',
            drawCallback: function() {
                // Re-initialize tooltips for new rows
                this.initializeTooltips();
            }.bind(this)
        };
        
        // Table-specific configurations
        if (table.id === 'resultsTable') {
            baseConfig.columnDefs = [
                {
                    targets: [3, 4], // Probability columns
                    render: function(data, type, row) {
                        if (type === 'display' && typeof data === 'number') {
                            const percentage = (data * 100).toFixed(1);
                            const color = data > 0.8 ? 'success' : data > 0.6 ? 'warning' : 'danger';
                            return `
                                <div class="progress" style="height: 20px;">
                                    <div class="progress-bar bg-${color}" 
                                         role="progressbar" 
                                         style="width: ${percentage}%"
                                         data-bs-toggle="tooltip"
                                         title="${percentage}%">
                                        ${percentage}%
                                    </div>
                                </div>
                            `;
                        }
                        return data;
                    }
                }
            ];
        }
        
        return baseConfig;
    }
    
    initializeFileUpload() {
        try {
            const fileInputs = document.querySelectorAll('input[type="file"]');
            
            fileInputs.forEach(input => {
                input.addEventListener('change', (event) => {
                    this.handleFileUpload(event);
                });
                
                // Add drag and drop functionality
                this.setupDragAndDrop(input);
            });
        } catch (error) {
            console.error('Error inicializando file upload:', error);
        }
    }
    
    handleFileUpload(event) {
        try {
            const file = event.target.files[0];
            if (!file) return;
            
            // Validate file
            const validation = this.validateFile(file);
            if (!validation.valid) {
                this.showNotification(validation.message, 'error');
                event.target.value = '';
                return;
            }
            
            // Show file info
            this.displayFileInfo(file, event.target);
            
            // Preview CSV if possible
            if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                this.previewCSV(file);
            }
        } catch (error) {
            console.error('Error manejando file upload:', error);
        }
    }
    
    validateFile(file) {
        try {
            const maxSize = 16 * 1024 * 1024; // 16MB
            const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];
            const allowedExtensions = ['.csv'];
            
            if (file.size > maxSize) {
                return {
                    valid: false,
                    message: `El archivo es demasiado grande. Máximo permitido: ${this.formatFileSize(maxSize)}`
                };
            }
            
            const isValidType = allowedTypes.includes(file.type) || 
                               allowedExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
            
            if (!isValidType) {
                return {
                    valid: false,
                    message: 'Solo se permiten archivos CSV'
                };
            }
            
            return { valid: true };
        } catch (error) {
            console.error('Error validando archivo:', error);
            return { valid: false, message: 'Error validando archivo' };
        }
    }
    
    displayFileInfo(file, input) {
        try {
            const container = input.closest('.mb-3') || input.parentNode;
            let infoDiv = container.querySelector('.file-info');
            
            if (!infoDiv) {
                infoDiv = document.createElement('div');
                infoDiv.className = 'file-info mt-2';
                container.appendChild(infoDiv);
            }
            
            infoDiv.innerHTML = `
                <div class="alert alert-info d-flex align-items-center">
                    <i class="fas fa-file-csv fa-2x me-3 text-primary"></i>
                    <div class="flex-grow-1">
                        <strong>${file.name}</strong><br>
                        <small class="text-muted">
                            ${this.formatFileSize(file.size)} • 
                            ${new Date(file.lastModified).toLocaleDateString()} • 
                            ${file.type || 'CSV'}
                        </small>
                    </div>
                    <button type="button" class="btn btn-outline-danger btn-sm" onclick="this.removeFile()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            
            // Add remove functionality
            infoDiv.querySelector('button').removeFile = () => {
                input.value = '';
                infoDiv.remove();
            };
        } catch (error) {
            console.error('Error mostrando info del archivo:', error);
        }
    }
    
    setupDragAndDrop(input) {
        try {
            const dropZone = input.closest('.mb-3') || input.parentNode;
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, this.preventDefaults);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => {
                    dropZone.classList.add('drag-over');
                });
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => {
                    dropZone.classList.remove('drag-over');
                });
            });
            
            dropZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files.length > 0) {
                    input.files = files;
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            });
        } catch (error) {
            console.error('Error configurando drag and drop:', error);
        }
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    previewCSV(file) {
        try {
            const reader = new FileReader();
            reader.onload = (e) => {
                const csv = e.target.result;
                const lines = csv.split('\n').slice(0, 6); // First 5 rows + header
                
                if (lines.length < 2) return;
                
                // Create preview table
                const headers = lines[0].split(',');
                const preview = `
                    <div class="csv-preview mt-3">
                        <h6><i class="fas fa-eye me-2"></i>Vista Previa del Archivo</h6>
                        <div class="table-responsive">
                            <table class="table table-sm table-striped">
                                <thead class="table-dark">
                                    <tr>${headers.map(h => `<th>${h.trim()}</th>`).join('')}</tr>
                                </thead>
                                <tbody>
                                    ${lines.slice(1, 6).map(line => 
                                        `<tr>${line.split(',').map(cell => `<td>${cell.trim()}</td>`).join('')}</tr>`
                                    ).join('')}
                                </tbody>
                            </table>
                        </div>
                        <small class="text-muted">
                            <i class="fas fa-info-circle me-1"></i>
                            Mostrando las primeras 5 filas de ${lines.length - 1} registros totales
                        </small>
                    </div>
                `;
                
                const container = document.querySelector('.file-info')?.parentNode;
                if (container) {
                    const existingPreview = container.querySelector('.csv-preview');
                    if (existingPreview) existingPreview.remove();
                    
                    container.insertAdjacentHTML('beforeend', preview);
                }
            };
            
            reader.readAsText(file);
        } catch (error) {
            console.error('Error previsualizando CSV:', error);
        }
    }
    
    initializeLoadingStates() {
        try {
            const forms = document.querySelectorAll('form');
            
            forms.forEach(form => {
                form.addEventListener('submit', (event) => {
                    if (form.checkValidity()) {
                        this.setFormLoading(form, true);
                        
                        // Reset loading state after timeout (fallback)
                        setTimeout(() => {
                            this.setFormLoading(form, false);
                        }, 30000);
                    }
                });
            });
        } catch (error) {
            console.error('Error inicializando loading states:', error);
        }
    }
    
    setFormLoading(form, loading) {
        try {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (!submitBtn) return;
            
            if (loading) {
                submitBtn.dataset.originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = `
                    <div class="spinner-border spinner-border-sm me-2" role="status">
                        <span class="visually-hidden">Cargando...</span>
                    </div>
                    Procesando...
                `;
                submitBtn.disabled = true;
            } else {
                submitBtn.innerHTML = submitBtn.dataset.originalText || submitBtn.innerHTML;
                submitBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error configurando loading state:', error);
        }
    }
    
    setupEventListeners() {
        try {
            // Enhanced search functionality
            this.setupGlobalSearch();
            
            // Keyboard shortcuts
            this.setupKeyboardShortcuts();
            
            // Performance monitoring
            this.setupPerformanceMonitoring();
        } catch (error) {
            console.error('Error configurando event listeners:', error);
        }
    }
    
    setupGlobalSearch() {
        try {
            const searchInputs = document.querySelectorAll('input[type="search"], .dataTables_filter input');
            
            searchInputs.forEach(input => {
                input.addEventListener('input', this.debounce((e) => {
                    this.handleGlobalSearch(e.target.value);
                }, 300));
            });
        } catch (error) {
            console.error('Error configurando búsqueda global:', error);
        }
    }
    
    handleGlobalSearch(query) {
        try {
            if (query.length < 2) return;
            
            // Highlight search terms in visible content
            this.highlightSearchTerms(query);
        } catch (error) {
            console.error('Error manejando búsqueda:', error);
        }
    }
    
    highlightSearchTerms(query) {
        try {
            // Simple highlighting implementation
            const elements = document.querySelectorAll('td, .card-text, .card-title');
            
            elements.forEach(element => {
                if (element.dataset.originalText) {
                    element.innerHTML = element.dataset.originalText;
                } else {
                    element.dataset.originalText = element.innerHTML;
                }
                
                if (query) {
                    const regex = new RegExp(`(${query})`, 'gi');
                    element.innerHTML = element.innerHTML.replace(regex, '<mark>$1</mark>');
                }
            });
        } catch (error) {
            console.error('Error resaltando términos:', error);
        }
    }
    
    setupKeyboardShortcuts() {
        try {
            document.addEventListener('keydown', (e) => {
                // Ctrl/Cmd + K for search
                if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                    e.preventDefault();
                    const searchInput = document.querySelector('.dataTables_filter input, input[type="search"]');
                    if (searchInput) {
                        searchInput.focus();
                    }
                }
                
                // Escape to clear search
                if (e.key === 'Escape') {
                    const searchInput = document.querySelector('.dataTables_filter input:focus');
                    if (searchInput) {
                        searchInput.value = '';
                        searchInput.dispatchEvent(new Event('input'));
                    }
                }
            });
        } catch (error) {
            console.error('Error configurando shortcuts:', error);
        }
    }
    
    setupProgressiveEnhancement() {
        try {
            // Lazy load images
            this.setupLazyLoading();
            
            // Intersection observer for animations
            this.setupScrollAnimations();
            
            // Enhanced accessibility
            this.enhanceAccessibility();
        } catch (error) {
            console.error('Error configurando progressive enhancement:', error);
        }
    }
    
    setupLazyLoading() {
        try {
            const images = document.querySelectorAll('img[data-src]');
            
            if ('IntersectionObserver' in window) {
                const imageObserver = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            const img = entry.target;
                            img.src = img.dataset.src;
                            img.classList.remove('lazy');
                            imageObserver.unobserve(img);
                        }
                    });
                });
                
                images.forEach(img => imageObserver.observe(img));
            }
        } catch (error) {
            console.error('Error configurando lazy loading:', error);
        }
    }
    
    setupScrollAnimations() {
        try {
            const animateElements = document.querySelectorAll('.animate-on-scroll, .card, .stats-card');
            
            if ('IntersectionObserver' in window) {
                const animationObserver = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            entry.target.classList.add('animated', 'fadeInUp');
                            animationObserver.unobserve(entry.target);
                        }
                    });
                }, {
                    threshold: 0.1,
                    rootMargin: '0px 0px -50px 0px'
                });
                
                animateElements.forEach(el => {
                    el.style.opacity = '0';
                    el.style.transform = 'translateY(20px)';
                    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                    animationObserver.observe(el);
                });
            }
        } catch (error) {
            console.error('Error configurando animaciones:', error);
        }
    }
    
    enhanceAccessibility() {
        try {
            // Add ARIA labels to interactive elements
            const interactiveElements = document.querySelectorAll('button, a, input, select');
            
            interactiveElements.forEach(element => {
                if (!element.getAttribute('aria-label') && !element.getAttribute('aria-labelledby')) {
                    const text = element.textContent?.trim() || element.value || element.placeholder;
                    if (text) {
                        element.setAttribute('aria-label', text);
                    }
                }
            });
            
            // Add skip links
            this.addSkipLinks();
        } catch (error) {
            console.error('Error mejorando accesibilidad:', error);
        }
    }
    
    addSkipLinks() {
        try {
            const skipLink = document.createElement('a');
            skipLink.href = '#main-content';
            skipLink.textContent = 'Saltar al contenido principal';
            skipLink.className = 'skip-link position-absolute';
            skipLink.style.cssText = `
                top: -40px;
                left: 6px;
                background: var(--primary);
                color: white;
                padding: 8px;
                text-decoration: none;
                border-radius: 4px;
                z-index: 1000;
            `;
            
            skipLink.addEventListener('focus', () => {
                skipLink.style.top = '6px';
            });
            
            skipLink.addEventListener('blur', () => {
                skipLink.style.top = '-40px';
            });
            
            document.body.insertBefore(skipLink, document.body.firstChild);
        } catch (error) {
            console.error('Error agregando skip links:', error);
        }
    }
    
    setupPerformanceMonitoring() {
        try {
            if ('PerformanceObserver' in window) {
                // Monitor Largest Contentful Paint
                const lcpObserver = new PerformanceObserver((list) => {
                    const entries = list.getEntries();
                    const lastEntry = entries[entries.length - 1];
                    console.log('📊 LCP:', lastEntry.startTime);
                });
                
                lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
                
                // Monitor Cumulative Layout Shift
                const clsObserver = new PerformanceObserver((list) => {
                    let clsValue = 0;
                    for (const entry of list.getEntries()) {
                        if (!entry.hadRecentInput) {
                            clsValue += entry.value;
                        }
                    }
                    console.log('📊 CLS:', clsValue);
                });
                
                clsObserver.observe({ entryTypes: ['layout-shift'] });
            }
        } catch (error) {
            console.error('Error configurando performance monitoring:', error);
        }
    }
    
    // Utility methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    formatNumber(num) {
        try {
            return new Intl.NumberFormat('es-ES', {
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }).format(num);
        } catch (error) {
            return num.toString();
        }
    }
    
    formatCurrency(value) {
        try {
            return new Intl.NumberFormat('es-ES', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        } catch (error) {
            return `$${value}`;
        }
    }
    
    formatPercentage(value) {
        try {
            return new Intl.NumberFormat('es-ES', {
                style: 'percent',
                minimumFractionDigits: 1,
                maximumFractionDigits: 1
            }).format(value);
        } catch (error) {
            return `${(value * 100).toFixed(1)}%`;
        }
    }
    
    formatFileSize(bytes) {
        try {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        } catch (error) {
            return 'Tamaño desconocido';
        }
    }
    
    showNotification(message, type = 'info') {
        try {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show notification-alert`;
            alertDiv.style.cssText = `
                position: fixed;
                top: 100px;
                right: 20px;
                z-index: 1055;
                min-width: 300px;
                max-width: 400px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            `;
            
            const icon = {
                success: 'fa-check-circle',
                error: 'fa-exclamation-triangle',
                danger: 'fa-exclamation-triangle',
                warning: 'fa-exclamation-circle',
                info: 'fa-info-circle'
            }[type] || 'fa-info-circle';
            
            alertDiv.innerHTML = `
                <i class="fas ${icon} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            document.body.appendChild(alertDiv);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
        } catch (error) {
            console.error('Error mostrando notificación:', error);
            // Fallback to alert
            alert(message);
        }
    }
    
    // API methods
    async callAPI(endpoint, method = 'GET', data = null) {
        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };
            
            if (data) {
                options.body = JSON.stringify(data);
            }
            
            const response = await fetch(`${this.apiEndpoint}${endpoint}`, options);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            this.showNotification('Error en la comunicación con el servidor', 'error');
            throw error;
        }
    }
}

// Initialize the application
try {
    window.LoanPredictApp = new LoanPredictApp();
    window.app = window.LoanPredictApp;
} catch (error) {
    console.error('❌ Error inicializando aplicación:', error);
    document.addEventListener('DOMContentLoaded', function() {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-warning';
        alertDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            Algunas funciones avanzadas pueden no estar disponibles. La aplicación funcionará en modo básico.
        `;
        document.body.insertBefore(alertDiv, document.body.firstChild);
    });
}

// Service Worker registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('✅ SW registered:', registration);
            })
            .catch(registrationError => {
                console.log('❌ SW registration failed:', registrationError);
            });
    });
}