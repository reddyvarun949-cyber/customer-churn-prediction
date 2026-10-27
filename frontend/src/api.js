// All communication with the Flask backend lives in this one file.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

// Turns any failed request into an Error with a message a person can read.
async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(API_URL + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (networkError) {
    throw new Error("Cannot reach the API. Is the backend running on " + API_URL + "?");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || "Request failed with status " + response.status);
    error.fields = data.fields || {};
    throw error;
  }
  return data;
}

export function getOptions() {
  return request("/api/options");
}

// One customer -> { churn_probability, will_churn, decision, recommended_action, ... }
export function predictChurn(customer) {
  return request("/api/predict", { method: "POST", body: JSON.stringify(customer) });
}

// A list of customers -> { results: [...], summary: {...} }
export function predictBatch(customers) {
  return request("/api/predict-batch", { method: "POST", body: JSON.stringify({ customers }) });
}
