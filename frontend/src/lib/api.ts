import { HealthResponse, EdaResponse, CustomerRiskResponse, ScoreResponse, ScoreRequest, NarrativeResponse, XaiResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchWithRetry(url: string, options: RequestInit = {}, retries = 5, delayMs = 2000): Promise<Response> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, options);
      if (res.ok || res.status === 404) return res;
    } catch (err) {
      if (attempt === retries) throw err;
    }
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return fetch(url, options);
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetchWithRetry(`${API_BASE_URL}/health`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Backend not healthy");
  }
  return res.json();
}

export async function fetchEda(customerSegment?: string, dateFrom?: string, dateTo?: string): Promise<EdaResponse> {
  const url = new URL(`${API_BASE_URL}/eda`);
  if (customerSegment) url.searchParams.append("customer_segment", customerSegment);
  if (dateFrom) url.searchParams.append("date_from", dateFrom);
  if (dateTo) url.searchParams.append("date_to", dateTo);

  const res = await fetchWithRetry(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Failed to fetch EDA data");
  }
  return res.json();
}

export async function fetchCustomerRisk(customerId: string): Promise<CustomerRiskResponse> {
  const res = await fetchWithRetry(`${API_BASE_URL}/customers/${customerId}/risk`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Customer ${customerId} not found`);
  }
  return res.json();
}

export async function fetchCustomerNarrative(customerId: string): Promise<NarrativeResponse> {
  const res = await fetchWithRetry(`${API_BASE_URL}/customers/${customerId}/narrative`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Narrative unavailable for ${customerId}`);
  }
  return res.json();
}

export async function fetchCustomerXai(customerId: string): Promise<XaiResponse> {
  const res = await fetchWithRetry(`${API_BASE_URL}/customers/${customerId}/xai`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`XAI data unavailable for ${customerId}`);
  }
  return res.json();
}

export async function fetchScore(request: ScoreRequest): Promise<ScoreResponse[]> {
  const res = await fetchWithRetry(`${API_BASE_URL}/score`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    cache: "no-store"
  });
  
  if (!res.ok) {
    throw new Error("Failed to fetch score data");
  }
  return res.json();
}
