document.addEventListener('DOMContentLoaded', function() {
    // Elementos del DOM
    const csvFileInput = document.getElementById('csvFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clusterBtn = document.getElementById('clusterBtn');
    const cleanBtn = document.getElementById('cleanBtn');
    const sampleBtn = document.getElementById('sample-btn');
    const previewTable = document.getElementById('preview-table');
    const resultsTbody = document.getElementById('results-tbody');
    const clusterResultsTbody = document.getElementById('cluster-results-tbody');
    const clusterStats = document.getElementById('cluster-stats');
    const errorMessage = document.getElementById('error-message');
    const noData = document.getElementById('no-data');
    const noClusterData = document.getElementById('no-cluster-data');
    const resultsContainer = document.getElementById('results-container');
    const clusterResultsContainer = document.getElementById('cluster-results-container');
    const loading = document.getElementById('loading');
    const loadingCluster = document.getElementById('loading-cluster');
    const numClustersSelect = document.getElementById('num-clusters');
    const previewSection = document.getElementById('preview-section');
    
    let currentData = null;
    
    // Manejar subida de archivos
    csvFileInput.addEventListener('change', function(e) {
        if (e.target.files.length === 0) return;
        
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        // Mostrar carga
        errorMessage.textContent = '';
        loading.style.display = 'flex';
        noData.style.display = 'none';
        resultsContainer.style.display = 'none';
        clusterResultsContainer.style.display = 'none';
        previewSection.style.display = 'none';
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => Promise.reject(err));
            }
            return response.json();
        })
        .then(data => {
            loading.style.display = 'none';
            
            if (data.error) {
                errorMessage.textContent = data.error;
                if (data.required_columns) {
                    errorMessage.textContent += '. Columnas requeridas: ' + data.required_columns.join(', ');
                }
                return;
            }
            
            currentData = data;
            displayPreview(data.preview);
            document.getElementById('total-rows').textContent = data.total_rows;
            analyzeBtn.disabled = false;
            clusterBtn.disabled = false;
            previewSection.style.display = 'block';
        })
        .catch(error => {
            loading.style.display = 'none';
            errorMessage.textContent = 'Error al subir el archivo: ' + (error.error || error.message);
            console.error('Error:', error);
        });
    });
    
    // Mostrar vista previa de datos
    function displayPreview(data) {
        if (!data || data.length === 0) {
            previewTable.innerHTML = '<p>No hay datos para mostrar</p>';
            return;
        }
        
        const headers = Object.keys(data[0]);
        let html = '<table><thead><tr>';
        
        headers.forEach(header => {
            html += `<th>${header}</th>`;
        });
        
        html += '</tr></thead><tbody>';
        
        data.forEach(row => {
            html += '<tr>';
            headers.forEach(header => {
                html += `<td>${row[header]}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table>';
        previewTable.innerHTML = html;
    }

// Botón de análisis corregido
analyzeBtn.addEventListener('click', function() {
    if (!currentData || !currentData.filename) {
        errorMessage.textContent = 'No hay datos cargados para analizar';
        return;
    }

    loading.style.display = 'flex';
    resultsContainer.style.display = 'none';
    errorMessage.textContent = '';

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: currentData.filename
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => Promise.reject(err));
        }
        return response.json();
    })
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            errorMessage.textContent = data.error;
            return;
        }
        
        if (data.success && data.results) {
            displayResults(data.results, data.stats);
            resultsContainer.style.display = 'block';
            
            // Mostrar mensaje de éxito si está disponible
            if (data.message) {
                console.log('✅ ' + data.message);
            }
        } else {
            errorMessage.textContent = 'No se recibieron resultados del análisis';
        }
    })
    .catch(error => {
        loading.style.display = 'none';
        console.error('Error completo:', error);
        
        // Manejo mejorado de errores
        let errorMsg = 'Error al analizar datos';
        if (error.error) {
            errorMsg = error.error;
        } else if (error.message) {
            errorMsg = error.message;
        } else if (typeof error === 'string') {
            errorMsg = error;
        }
        
        errorMessage.textContent = errorMsg;
    });
});

// Función displayResults actualizada para manejar la nueva estructura de datos
function displayResults(results, stats) {
    resultsTbody.innerHTML = '';
    
    if (!results || results.length === 0) {
        resultsTbody.innerHTML = '<div class="table-row">No hay resultados para mostrar</div>';
        return;
    }
    
    // Mostrar cada resultado
    results.forEach(result => {
        const row = document.createElement('div');
        row.className = 'table-row';
        
        // Crear tooltip con probabilidades detalladas si están disponibles
        let tooltip = '';
        if (result.probabilities) {
            tooltip = `title="Probabilidades: Bajo ${result.probabilities.bajo}%, Medio ${result.probabilities.medio}%, Alto ${result.probabilities.alto}%"`;
        }
        
        row.innerHTML = `
            <div>${result.id}</div>
            <div style="color: ${result.color}; font-weight: bold;" ${tooltip}>
                ${result.prediction}
            </div>
            <div>${result.confidence}%</div>
            <div style="color: ${result.color}">
                ${result.risk}
            </div>
        `;
        resultsTbody.appendChild(row);
    });
    
    // Actualizar estadísticas con los nuevos campos
    if (stats) {
        // Estadísticas básicas
        document.getElementById('stat-alto').textContent = stats.alto || '0';
        document.getElementById('stat-medio').textContent = stats.medio || '0';
        document.getElementById('stat-bajo').textContent = stats.bajo || '0';
        document.getElementById('stat-confianza').textContent = 
            (stats.confianza_promedio || '0') + '%';
        
        // Estadísticas adicionales si existen elementos para mostrarlas
        if (document.getElementById('stat-total')) {
            document.getElementById('stat-total').textContent = stats.total_muestras || '0';
        }
        
        if (document.getElementById('stat-porcentaje-alto')) {
            document.getElementById('stat-porcentaje-alto').textContent = 
                (stats.porcentaje_alto || '0') + '%';
        }
        
        if (document.getElementById('stat-porcentaje-medio')) {
            document.getElementById('stat-porcentaje-medio').textContent = 
                (stats.porcentaje_medio || '0') + '%';
        }
        
        if (document.getElementById('stat-porcentaje-bajo')) {
            document.getElementById('stat-porcentaje-bajo').textContent = 
                (stats.porcentaje_bajo || '0') + '%';
        }
        
        // Mostrar resumen en consola para debugging
        console.log('📊 Estadísticas del análisis:', {
            total: stats.total_muestras,
            alto: `${stats.alto} (${stats.porcentaje_alto}%)`,
            medio: `${stats.medio} (${stats.porcentaje_medio}%)`,
            bajo: `${stats.bajo} (${stats.porcentaje_bajo}%)`,
            confianza_promedio: `${stats.confianza_promedio}%`
        });
    }
}
    
    // Botón de clustering
    clusterBtn.addEventListener('click', function() {
        if (!currentData || !currentData.filename) return;
        
        const n_clusters = parseInt(numClustersSelect.value);
        
        loadingCluster.style.display = 'flex';
        clusterResultsContainer.style.display = 'none';
        
        fetch('/cluster', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: currentData.filename,
                n_clusters: n_clusters
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => Promise.reject(err));
            }
            return response.json();
        })
        .then(data => {
            loadingCluster.style.display = 'none';
            
            if (data.error) {
                errorMessage.textContent = data.error;
                return;
            }
            
            displayClusterResults(data);
            clusterResultsContainer.style.display = 'block';
        })
        .catch(error => {
            loadingCluster.style.display = 'none';
            errorMessage.textContent = 'Error en clustering: ' + (error.error || error.message);
            console.error('Error:', error);
        });
    });
    
    // Mostrar resultados de clustering
    function displayClusterResults(data) {
        clusterResultsTbody.innerHTML = '';
        clusterStats.innerHTML = '';
        
        // Mostrar imagen del cluster
        const clusterViz = document.getElementById('cluster-viz');
        clusterViz.innerHTML = `<img src="data:image/png;base64,${data.cluster_image}" alt="Visualización de Clusters">`;
        
        // Llenar tabla de resultados
        data.results.forEach(result => {
            const row = document.createElement('div');
            row.className = 'table-row';
            
            const features = Object.entries(result.features)
                .map(([key, value]) => `${key}: ${value}`)
                .join(', ');
            
            row.innerHTML = `
                <div>${result.id}</div>
                <div>Cluster ${result.cluster}</div>
                <div>${result.distance}</div>
                <div title="${features}">${features.substring(0, 30)}...</div>
            `;
            clusterResultsTbody.appendChild(row);
        });
        
        // Mostrar estadísticas
        data.stats.forEach(stat => {
            const statCard = document.createElement('div');
            statCard.className = 'cluster-stat-card';
            
            const means = Object.entries(stat.means)
                .map(([key, value]) => `${key}: ${value}`)
                .join(', ');
            
            statCard.innerHTML = `
                <h4>Cluster ${stat.cluster}</h4>
                <p><strong>Muestras:</strong> ${stat.count}</p>
                <p><strong>Medias:</strong> ${means}</p>
            `;
            clusterStats.appendChild(statCard);
        });
    }
    
    // Botón de limpiar
    cleanBtn.addEventListener('click', function() {
        csvFileInput.value = '';
        previewTable.innerHTML = '';
        resultsTbody.innerHTML = '';
        clusterResultsTbody.innerHTML = '';
        clusterStats.innerHTML = '';
        errorMessage.textContent = '';
        document.getElementById('total-rows').textContent = '0';
        analyzeBtn.disabled = true;
        clusterBtn.disabled = true;
        resultsContainer.style.display = 'none';
        clusterResultsContainer.style.display = 'none';
        noData.style.display = 'block';
        noClusterData.style.display = 'block';
        currentData = null;
        previewSection.style.display = 'none';
    });
    
    // Botón de ejemplo
    sampleBtn.addEventListener('click', function() {
        window.location.href = '/download_sample';
    });
    
    // Manejar pestañas
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            this.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        });
    });
    
    // Cambiar número de clusters
    numClustersSelect.addEventListener('change', function() {
        if (clusterResultsContainer.style.display === 'block') {
            clusterBtn.click();
        }
    });
});