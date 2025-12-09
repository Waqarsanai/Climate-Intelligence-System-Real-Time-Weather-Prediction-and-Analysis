
// Fetch model metrics from API
async function fetchModelMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/system/status`);
        if (!response.ok) return;
        const data = await response.json();

        // Check if we have model metrics
        if (data.model_metrics) {
            updateModelMetrics(data.model_metrics);
        } else {
            // Set default/placeholder values
            updateModelMetrics({
                R2: 0.90,
                RMSE: 0.8,
                MAE: 0.6,
                target_accuracy_min: data.target_accuracy_min || 93,
                target_accuracy_max: data.target_accuracy_max || 95
            });
        }
    } catch (error) {
        console.error('Error fetching model metrics:', error);
    }
}

