var chartInstances = {};

function renderChart(canvas) {
    var url = canvas.dataset.chartUrl;
    if (!url) return;

    fetch(url)
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var id = canvas.id;

            if (chartInstances[id]) {
                chartInstances[id].data.labels = data.labels;
                chartInstances[id].data.datasets[0].data = data.loss;
                chartInstances[id].data.datasets[1].data = data.accuracy;
                chartInstances[id].update('none');
                return;
            }

            chartInstances[id] = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [
                        {
                            label: 'Loss',
                            data: data.loss,
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231,76,60,0.1)',
                            fill: true,
                            yAxisID: 'y',
                            tension: 0.3,
                        },
                        {
                            label: 'Accuracy',
                            data: data.accuracy,
                            borderColor: '#2ecc71',
                            backgroundColor: 'rgba(46,204,113,0.1)',
                            fill: true,
                            yAxisID: 'y1',
                            tension: 0.3,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    interaction: { mode: 'index', intersect: false },
                    scales: {
                        y: { type: 'linear', position: 'left', title: { display: true, text: 'Loss' } },
                        y1: { type: 'linear', position: 'right', min: 0, max: 1, title: { display: true, text: 'Accuracy' }, grid: { drawOnChartArea: false } },
                    },
                },
            });
        })
        .catch(function() {});
}

function initCharts() {
    document.querySelectorAll('canvas[data-chart-url]').forEach(renderChart);
}

setInterval(initCharts, 3000);
document.addEventListener('DOMContentLoaded', initCharts);
document.body.addEventListener('htmx:afterSettle', function() {
    setTimeout(initCharts, 200);
});

// --- Job detail overlay (pure JS, no HTMX dependency) ---

function loadJobDetail(jobId) {
    var container = document.getElementById('job-detail-container');
    var overlay = document.getElementById('job-detail-overlay');
    if (!container || !overlay) return;

    container.innerHTML = '<p style="text-align:center;padding:2rem" aria-busy="true">Loading...</p>';
    overlay.classList.remove('hidden');

    fetch('/dashboard/partials/job/' + jobId)
        .then(function(r) { return r.text(); })
        .then(function(html) {
            container.innerHTML = html;
            // Process any htmx attributes in the new content
            if (window.htmx) htmx.process(container);
            setTimeout(initCharts, 150);
        })
        .catch(function(err) {
            container.innerHTML = '<p style="color:#c0392b;padding:1rem">Failed to load job details.</p>';
        });
}

function closeJobDetail() {
    var overlay = document.getElementById('job-detail-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
        var container = document.getElementById('job-detail-container');
        if (container) {
            container.querySelectorAll('canvas').forEach(function(c) {
                if (chartInstances[c.id]) {
                    chartInstances[c.id].destroy();
                    delete chartInstances[c.id];
                }
            });
            container.innerHTML = '';
        }
    }
}

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeJobDetail();
});
