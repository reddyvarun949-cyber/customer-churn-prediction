import { useEffect, useState } from "react";
import { getOptions, predictBatch, predictChurn } from "./api.js";

// Starting values for the form (a fairly new month-to-month fibre customer).
const START_FORM = {
  tenure_months: "4",
  contract_type: "Month-to-month",
  monthly_charges: "999",
  total_charges: "3996",
  internet_service: "Fiber optic",
  tech_support: "No",
  payment_method: "Electronic check",
  paperless_billing: "Yes",
  num_complaints_6m: "1",
  avg_data_gb: "80",
  senior_citizen: "0",
  city_tier: "Tier 1",
};

const NUMBER_FIELDS = [
  { name: "tenure_months", label: "Tenure (months)", step: "1" },
  { name: "monthly_charges", label: "Monthly charges (₹)", step: "1" },
  { name: "total_charges", label: "Total charges so far (₹)", step: "0.01" },
  { name: "num_complaints_6m", label: "Complaints in last 6 months", step: "1" },
  { name: "avg_data_gb", label: "Average data use (GB / month)", step: "0.1" },
];

const SELECT_FIELDS = [
  { name: "contract_type", label: "Contract" },
  { name: "internet_service", label: "Internet service" },
  { name: "tech_support", label: "Tech support" },
  { name: "payment_method", label: "Payment method" },
  { name: "paperless_billing", label: "Paperless billing" },
  { name: "city_tier", label: "City tier" },
];

// Example input for the batch box (two customers).
const BATCH_EXAMPLE = JSON.stringify(
  [
    { customer_id: "CUST-A", ...START_FORM },
    {
      customer_id: "CUST-B", tenure_months: 60, contract_type: "Two year", monthly_charges: 700,
      total_charges: 42000, internet_service: "DSL", tech_support: "Yes",
      payment_method: "Bank transfer (automatic)", paperless_billing: "No", num_complaints_6m: 0,
      avg_data_gb: 45, senior_citizen: 0, city_tier: "Tier 2",
    },
  ],
  null,
  2
);

// Very simple CSV reader: first line = column names, values separated by commas.
// It does not handle quoted values that contain commas (none of our columns need that).
function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim() !== "");
  if (lines.length < 2) throw new Error("The CSV needs a header line and at least one customer line.");
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const customer = {};
    headers.forEach((header, i) => {
      customer[header] = (values[i] || "").trim(); // numbers stay as text; the API converts them
    });
    return customer;
  });
}

function percent(p) {
  return (p * 100).toFixed(1) + "%";
}

// ---------------- single customer ----------------
function SingleCustomer({ options }) {
  const [form, setForm] = useState(START_FORM);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [loading, setLoading] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;
    setForm({ ...form, [name]: value });
    setFieldErrors({ ...fieldErrors, [name]: undefined }); // clear the message for this field
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setResult(null);
    setError("");
    setFieldErrors({});
    setLoading(true);
    try {
      setResult(await predictChurn(form)); // the API checks every value
    } catch (err) {
      setError(err.message);
      setFieldErrors(err.fields || {});
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>One customer</h2>
      <form onSubmit={handleSubmit} noValidate>
        <div className="grid">
          {NUMBER_FIELDS.map((field) => {
            const [min, max] = options.numeric_ranges[field.name];
            return (
              <label key={field.name}>
                {field.label}
                <input
                  type="number"
                  name={field.name}
                  value={form[field.name]}
                  min={min}
                  max={max}
                  step={field.step}
                  onChange={handleChange}
                  className={fieldErrors[field.name] ? "invalid" : ""}
                />
                <span className="hint">{fieldErrors[field.name] || `Allowed: ${min} – ${max}`}</span>
              </label>
            );
          })}

          {SELECT_FIELDS.map((field) => (
            <label key={field.name}>
              {field.label}
              <select
                name={field.name}
                value={form[field.name]}
                onChange={handleChange}
                className={fieldErrors[field.name] ? "invalid" : ""}
              >
                {options.categories[field.name].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
              {fieldErrors[field.name] && <span className="hint error-text">{fieldErrors[field.name]}</span>}
            </label>
          ))}

          <label>
            Senior citizen (60+)
            <select name="senior_citizen" value={form.senior_citizen} onChange={handleChange}>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </label>
        </div>
        <p className="hint">
          Leave total charges empty only for a brand-new customer (tenure 0). Data use may be left empty.
        </p>

        <button type="submit" disabled={loading}>
          {loading ? "Predicting…" : "Predict churn"}
        </button>
      </form>

      {error && <div className="alert">{error}</div>}

      {result && (
        <div className={"result " + (result.will_churn ? "risk" : "safe")}>
          <p className="muted">Churn probability (risk score)</p>
          <p className="big">{percent(result.churn_probability)}</p>
          <p>
            <strong>Decision: {result.decision}</strong> (threshold {percent(result.threshold)})
          </p>
          <p>{result.recommended_action}</p>
          <p className="muted small">{result.note} Model: {result.model}.</p>
        </div>
      )}
    </section>
  );
}

// ---------------- batch ----------------
function BatchCustomers({ options }) {
  const [text, setText] = useState(BATCH_EXAMPLE);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Read an uploaded CSV file in the browser, convert it to JSON and show it in the text box.
  function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setText(JSON.stringify(parseCsv(String(reader.result)), null, 2));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    };
    reader.readAsText(file);
  }

  async function handleRun() {
    setResponse(null);
    setError("");
    let customers;
    try {
      customers = JSON.parse(text);
    } catch {
      setError("The text is not valid JSON. Paste a list like [ {...}, {...} ] or upload a CSV file.");
      return;
    }
    if (!Array.isArray(customers)) {
      setError("Paste a JSON list: [ {...}, {...} ].");
      return;
    }
    if (customers.length > options.max_batch) {
      setError(`At most ${options.max_batch} customers per request (you have ${customers.length}).`);
      return;
    }
    setLoading(true);
    try {
      setResponse(await predictBatch(customers));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Many customers (batch)</h2>
      <p className="muted small">
        Paste a JSON list of customers or upload a CSV whose first line has the column names (same names as the
        form fields, plus an optional <code>customer_id</code>). Up to {options.max_batch} customers.
      </p>
      <label>
        Upload CSV
        <input type="file" accept=".csv,text/csv" onChange={handleFile} />
      </label>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={10} spellCheck="false" />
      <button type="button" onClick={handleRun} disabled={loading}>
        {loading ? "Scoring…" : "Score customers"}
      </button>

      {error && <div className="alert">{error}</div>}

      {response && (
        <>
          <p>
            {response.summary.valid} of {response.summary.total} rows scored, {response.summary.invalid} with
            errors. <strong>{response.summary.to_contact} customers to contact.</strong>
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Row</th>
                  <th>Customer</th>
                  <th>Probability</th>
                  <th>Decision</th>
                  <th>Action / errors</th>
                </tr>
              </thead>
              <tbody>
                {response.results.map((r) => (
                  <tr key={r.row} className={!r.ok ? "row-error" : r.will_churn ? "row-risk" : ""}>
                    <td>{r.row}</td>
                    <td>{r.customer_id || "—"}</td>
                    <td>{r.ok ? percent(r.churn_probability) : "—"}</td>
                    <td>{r.ok ? r.decision : "Invalid"}</td>
                    <td>
                      {r.ok
                        ? r.recommended_action
                        : Object.entries(r.fields).map(([field, msg]) => (
                            <div key={field}>
                              <code>{field}</code>: {msg}
                            </div>
                          ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

// ---------------- page ----------------
function App() {
  const [options, setOptions] = useState(null); // allowed values from the API
  const [error, setError] = useState("");

  // Load the dropdown values, threshold and business assumptions once when the page opens.
  useEffect(() => {
    getOptions()
      .then(setOptions)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <main className="page">
      <header>
        <h1>Customer Churn Predictor</h1>
        <p className="muted">
          Estimate which customers are likely to cancel next month, and whether a retention offer is worth it.
        </p>
      </header>

      {error && <div className="alert">{error}</div>}
      {!options && !error && <p className="muted">Loading…</p>}

      {options && (
        <>
          <p className="muted small">
            Model: {options.model}. Contact a customer when the churn probability is at least{" "}
            {percent(options.threshold)} — chosen to maximise expected profit with an offer cost of ₹
            {options.business_assumptions.offer_cost_rupees}, a saved customer worth ₹
            {options.business_assumptions.saved_customer_value_rupees} and{" "}
            {percent(options.business_assumptions.offer_success_rate)} of churners accepting the offer.
          </p>
          <SingleCustomer options={options} />
          <BatchCustomers options={options} />
        </>
      )}
    </main>
  );
}

export default App;
