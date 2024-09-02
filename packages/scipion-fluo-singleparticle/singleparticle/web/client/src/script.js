import Chart from 'chart.js/auto';

const BASE_URL = 'http://127.0.0.1:5000/';

const ctx = document.getElementById('energy');
var labels = [];
var data = [];
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            label: 'Energy',
            data: data,
            fill: false
        }]
    },
    options: {
        scales: {
            y: {
                type: 'logarithmic'
            }
        }
    }
}
);

function fetchCSVData() {
    /** @type {HTMLSelectElement} */
    const [_, project, run, name] = document.location.pathname.split('/');
    const energies_url = BASE_URL + 'user_data/projects/' + project + '/Runs/' + run + '/extra/working_dir/energies.csv';
    fetch(energies_url)
        .then(response => {
            if (response.ok) {
                return response.text()
            }
            else if (response.status == 404) {
                return null;
            } else {
                return Promise.reject(new Error(`Failed to fetch with status: ${response.status}`))
            }
        })
        .then(csvText => {
            if (csvText) {
                updateCSVData(csvText);
            }
        })
        .catch(error => {
            console.log("Error not 404: ", error);
        })
}

function updateCSVData(csvText) {
    const new_energies = csvText.split('\n').map(line => line.trim()).map(line => line.split(',').at(-1)).filter(val => !isNaN(parseFloat(val)));
    for (let i = labels.length; i < new_energies.length; i++) {
        labels.push(i);
        data.push(parseFloat(new_energies[i]));
    }
    chart.update();
}

// Fonction de polling
function startPolling(chart, interval = 1000) {
    fetchCSVData(chart); // Chargez les données initiales
    setInterval(() => fetchCSVData(chart), interval); // Rafraîchissement périodique
}

startPolling(chart);