document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');
    
    const resultContainer = document.getElementById('result-container');
    const resultValue = document.getElementById('result-value');
    const resultError = document.getElementById('result-error');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Get values
        const sqft = parseFloat(document.getElementById('square_footage').value);
        const beds = parseInt(document.getElementById('bedrooms').value);
        const baths = parseFloat(document.getElementById('bathrooms').value);

        // UI Loading State
        submitBtn.disabled = true;
        btnText.classList.add('loading');
        btnLoader.classList.remove('hidden');
        
        resultContainer.classList.remove('show');
        resultValue.classList.add('hidden');
        resultError.classList.add('hidden');

        try {
            // Because on Vercel the API is in the same origin, we can just use /api/predict
            // Fallback for local testing if running a separate server, but Vercel unified URL is best.
            const apiUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
                            ? '/api/predict' 
                            : '/api/predict';

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    square_footage: sqft,
                    bedrooms: beds,
                    bathrooms: baths
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            // Format price
            const formattedPrice = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            }).format(data.price);

            // Animate Number value (Counter effect)
            animateValue(resultValue, 0, data.price, 1000);
            
            resultValue.classList.remove('hidden');

        } catch (error) {
            console.error('Prediction Error:', error);
            resultError.textContent = 'Error: ' + error.message;
            resultError.classList.remove('hidden');
        } finally {
            // Reset Loading State
            submitBtn.disabled = false;
            btnText.classList.remove('loading');
            btnLoader.classList.add('hidden');
            
            // Show result container
            resultContainer.classList.add('show');
        }
    });

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            
            // Ease out cubic
            const easeOutProgress = 1 - Math.pow(1 - progress, 3);
            const currentVal = Math.floor(easeOutProgress * (end - start) + start);
            
            obj.innerHTML = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            }).format(currentVal);
            
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
