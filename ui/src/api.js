/** Minimal API client. No external deps — uses native fetch. */

const BASE = '/api';

/** POST /api/cfa-chat */
export async function cfaChat({ user_input, scenario, mode, secondary_check }) {
  const res = await fetch(`${BASE}/cfa-chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_input,
      scenario: scenario || 'auto',
      mode: mode || 'rule_only',
      secondary_check: secondary_check || false,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

/** POST /api/cfa-analyze */
export async function cfaAnalyze({ user_input, model_output, scenario, mode, secondary_check }) {
  const res = await fetch(`${BASE}/cfa-analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_input,
      model_output,
      scenario: scenario || 'healthcare',
      mode: mode || 'rule_only',
      secondary_check: secondary_check || false,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

/** GET /api/fact-schema?scenario=xxx */
export async function fetchFactSchema(scenario) {
  const res = await fetch(`${BASE}/fact-schema?scenario=${encodeURIComponent(scenario || 'healthcare')}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json();
}

/** GET /api/protected-facts?scenario=xxx */
export async function fetchProtectedFacts(scenario) {
  const res = await fetch(`${BASE}/protected-facts?scenario=${encodeURIComponent(scenario || 'healthcare')}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json();
}

/** POST /api/protected-facts */
export async function addProtectedFact({ scenario, fact }) {
  const res = await fetch(`${BASE}/protected-facts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scenario: scenario || 'healthcare',
      fact,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json();
}

/** GET /api/debug/last-llm-payload */
export async function fetchLastLlmPayload({ requestId, purpose } = {}) {
  const params = new URLSearchParams();
  if (requestId) params.set('request_id', requestId);
  if (purpose) params.set('purpose', purpose);

  const suffix = params.toString() ? `?${params.toString()}` : '';
  const res = await fetch(`${BASE}/debug/last-llm-payload${suffix}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json();
}

/** POST /api/protected-facts/import-jsonl */
export async function importConfidentialJsonl({ content, filename, replace }) {
  const res = await fetch(`${BASE}/protected-facts/import-jsonl`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content,
      filename,
      replace: Boolean(replace),
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json();
}
