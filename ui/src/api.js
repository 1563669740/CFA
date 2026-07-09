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