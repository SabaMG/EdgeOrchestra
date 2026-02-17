const chartInstances = {};

function renderChart(canvas) {
    const url = canvas.dataset.chartUrl;
    if (!url) return;

    fetch(url)
        .then(r => r.json())
        .then(data => {
            const id = canvas.id;

            if (chartInstances[id]) {
                chartInstances[id].data.labels = data.labels;
                chartInstances[id].data.datasets[0].data = data.loss;
                chartInstances[id].data.datasets[1].data = data.accuracy;
                chartInstances[id].update();
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
                            yAxisID: 'y',
                            tension: 0.3,
                        },
                        {
                            label: 'Accuracy',
                            data: data.accuracy,
                            borderColor: '#2ecc71',
                            backgroundColor: 'rgba(46,204,113,0.1)',
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
        .catch(() => {});
}

function initCharts() {
    document.querySelectorAll('canvas[data-chart-url]').forEach(renderChart);
}

document.addEventListener('DOMContentLoaded', initCharts);
document.body.addEventListener('htmx:afterSwap', () => setTimeout(initCharts, 100));
