async function analyzeImage() {
    const fileInput = document.getElementById('imageInput');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');

    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select an X-ray image first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        document.getElementById('predClass').innerText = data.predicted_class || data.diagnosis || 'Unknown';
        document.getElementById('predConf').innerText = data.confidence ? (data.confidence * 100).toFixed(2) + '%' : '-';
        document.getElementById('rawOutput').innerText = JSON.stringify(data, null, 2);
        resultsDiv.style.display = 'block';
    } catch (err) {
        alert('Error analyzing image: ' + err.message);
    } finally {
        loadingDiv.style.display = 'none';
    }
}
