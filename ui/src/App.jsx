import { useState, useCallback, useEffect } from 'react';
import {
  cfaChat,
  cfaAnalyze,
  fetchFactSchema,
  fetchProtectedFacts,
  addProtectedFact,
  importConfidentialJsonl,
  fetchLastLlmPayload,
} from './api';
import DiffView from './DiffView';

/* ================================================================
   Constants
   ================================================================ */
const SCENARIOS = [
  { id: 'auto',          label: '🤖 自动识别' },
  { id: 'healthcare',    label: '🏥 医疗健康' },
  { id: 'finance',       label: '💰 金融信贷' },
  { id: 'aerospace',     label: '🛰 航天测控' },
  { id: 'meetings',      label: '📋 会议室'   },
  { id: 'confidential',  label: '🛡️ 保密库' },
];

const MODES = [
  { id: 'rule_only',     label: '模式1 · 规则' },
  { id: 'rule_plus_llm', label: '模式2/3 · 规则+LLM' },
];

const QUICK_EXAMPLES_CHAT = [
  '最近ICU病房那边急性心梗做完介入的病人，目前的抗凝方案和医保情况怎么样？',
  '上海那边半导体行业的公司，最近申请的项目贷款和流动资金怎么样了？',
  '请结合航天测控生产区的情况，判断哪些系统需要优先处置。',
];

const QUICK_EXAMPLES_ANALYZE = [
  {
    ui: '心内科ICU的那个心梗病人',
    mo: '建议继续双抗，并结合β受体阻滞剂治疗。',
  },
  {
    ui: '帮我查一下王芳的情况',
    mo: '王芳目前在心内科ICU，诊断为急性广泛前壁心肌梗死，正在使用替格瑞洛联合阿司匹林双抗治疗，医保类型为城镇职工医保。',
  },
];

/* ================================================================
   Helpers
   ================================================================ */
const RISK_CLASS = (level) => {
  if (!level || level === 'NONE') return 'safe';
  if (level === 'LOW' || level === 'MEDIUM') return 'warn';
  return 'danger';
};

const SAFE_USED_LABELS = {
  raw_answer:              '原样返回（无风险）',
  cfa_safe_answer:         'CFA 规则改写',
  secondary_safe_answer:   'LLM 改写 + 二次检测通过',
  fallback:                '兜底安全答复',
  general_answer:          '通用回答（非领域问题）',
};

const INTENT_LABELS = {
  domain_healthcare: '🏥 医疗健康',
  domain_finance:    '💰 金融信贷',
  domain_aerospace:  '🛰️ 航天测控',
  domain_meetings:   '📋 会议室',
  general_weather:   '🌤️ 天气查询',
  general_chat:      '💬 通用对话',
  ambiguous:         '❓ 意图不明',
};

const STRATEGY_LABELS = {
  cfa_gated:         'CFA 安全网关检测',
  general_answer:    '通用 AI 回答',
  weather_answer:    '天气查询回答',
  need_city_prompt:  '天气查询（需追问城市）',
};

/* ================================================================
   Output Section
   ================================================================ */
function OutputSection({ loading, error, result }) {
  if (loading) {
    return (
      <div className="output-area">
        <div className="output-empty">
          <div className="spinner" />
          <div className="hint">正在处理，请稍候...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="output-area">
        <div className="error-banner">⚠ {error}</div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="output-area">
        <div className="output-empty">
          <div className="icon">🛡️</div>
          <div className="hint">组合事实还原风险检测与防护平台</div>
          <div className="hint" style={{ fontSize: '0.8rem' }}>
            在左侧输入内容后点击「发送检测」
          </div>
        </div>
      </div>
    );
  }

  const riskCls = RISK_CLASS(result.risk_level);

  return (
    <div className="output-area">
      {/* Risk Banner */}
      {result.risk_detected ? (
        <div className={`risk-banner ${riskCls}`}>
          <span>⚠ 检测到组合事实还原风险</span>
          <span className="badge">{result.risk_level}</span>
        </div>
      ) : (
        <div className={`risk-banner safe`}>
          <span>✅ 未检测到组合事实还原风险</span>
          <span className="badge">安全</span>
        </div>
      )}

      {/* Meta Grid */}
      <div className="meta-grid">
        <div className="meta-item">
          <div className="meta-label">CFA 得分</div>
          <div className="meta-value">{result.score}{result.findings_count > 0 ? ' / 100' : ''}</div>
        </div>
        <div className="meta-item">
          <div className="meta-label">回答策略</div>
          <div className="meta-value">
            {SAFE_USED_LABELS[result.safe_answer_used] || result.safe_answer_used}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">风险发现数</div>
          <div className="meta-value">{result.findings_count}</div>
        </div>
        <div className="meta-item">
          <div className="meta-label">请求 ID</div>
          <div className="meta-value" style={{ fontSize: '0.8rem', wordBreak: 'break-all' }}>
            {result.request_id}
          </div>
        </div>
      </div>

      {/* Routing Info (intent / routed_scenario / answer_strategy) */}
      {(result.intent || result.routed_scenario || result.answer_strategy) && (
        <div className="answer-card" style={{ borderLeft: '3px solid var(--primary)', marginBottom: 12 }}>
          <div className="card-title">🧭 路由信息</div>
          <div className="card-body" style={{ fontSize: '0.85rem', display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
            {result.intent && (
              <span><strong>识别意图：</strong>{INTENT_LABELS[result.intent] || result.intent}</span>
            )}
            {result.routed_scenario && (
              <span><strong>实际路由：</strong>{result.routed_scenario}</span>
            )}
            {result.answer_strategy && (
              <span><strong>应答策略：</strong>{STRATEGY_LABELS[result.answer_strategy] || result.answer_strategy}</span>
            )}
          </div>
        </div>
      )}

      {/* Risk Detail Cards */}
      {result.findings_summary && result.findings_summary.map((f, idx) => (
        <div key={idx} className="answer-card" style={{ borderLeft: `3px solid var(${f.level === 'CRITICAL' ? '--red' : f.level === 'HIGH' ? '--red' : '--yellow'})` }}>
          <div className="card-title">
            🔍 风险发现 #{idx + 1} — {f.level} — {f.target} ({f.target_id})
          </div>
          <div className="card-body" style={{ fontSize: '0.88rem' }}>
            <p><strong>被还原信息：</strong>{f.restored}</p>
            <p><strong>关键锚点：</strong></p>
            <ul style={{ margin: '4px 0 8px 20px' }}>
              {f.key_anchors.map((a, i2) => <li key={i2}>{a}</li>)}
            </ul>
            {f.chain && f.chain.length > 0 && (
              <>
                <p><strong>还原链：</strong></p>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-dim)', fontFamily: 'monospace', marginBottom: 4 }}>
                  {f.chain.map((s) => (
                    <div key={s.field_name + s.canonical_value}>
                      {s.field_label}{s.match_symbol}{s.canonical_value}: {s.before_count} → {s.after_count}
                      {s.after_count <= 5 ? ` (${s.remaining_asset_ids.join(',')})` : ''}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      ))}

      {/* Diff Comparison View */}
      {result.raw_answer && result.answer && (
        <DiffView rawAnswer={result.raw_answer} safeAnswer={result.answer} />
      )}

      {/* Raw Answer (LLM 原始输出) */}
      {result.raw_answer && !result.answer && (
        <div className="answer-card">
          <div className="card-title">
            📝 LLM 原始输出
            <button className="copy-btn" onClick={() => copyText(result.raw_answer)}>📋 复制</button>
          </div>
          <div className="card-body">{result.raw_answer}</div>
        </div>
      )}

      {/* Safe Answer — only shown if no raw_answer for comparison */}
      {result.answer && !result.raw_answer && (
        <div className="answer-card">
          <div className="card-title">
            🔒 CFA 安全回答
            <button className="copy-btn" onClick={() => copyText(result.answer)}>📋 复制</button>
          </div>
          <div className="card-body">{result.answer}</div>
        </div>
      )}

      {/* Raw JSON */}
      <details style={{ marginBottom: 16 }}>
        <summary style={{ cursor: 'pointer', fontSize: '0.85rem', color: 'var(--text-dim)', marginBottom: 8 }}>
          查看完整响应 JSON
        </summary>
        <div className="raw-json">{JSON.stringify(result, null, 2)}</div>
      </details>
    </div>
  );
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    // brief visual feedback is nice but keep it simple
  }).catch(() => {});
}

/* ================================================================
   Confidential Import Box
   ================================================================ */
function ConfidentialImportBox({ onImported }) {
  const [fileName, setFileName] = useState('');
  const [content, setContent] = useState('');
  const [replace, setReplace] = useState(true);
  const [preview, setPreview] = useState(null);
  const [importing, setImporting] = useState(false);
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');

  const parsePreview = (text) => {
    const lines = text.split(/\r?\n/).filter((line) => line.trim());
    const categories = {};
    const levels = {};
    const samples = [];
    let valid = 0;
    let invalid = 0;

    for (const line of lines) {
      try {
        const row = JSON.parse(line);
        valid += 1;

        const category = row.category || '未分类';
        const level = row.confidential_level || 'unknown';

        categories[category] = (categories[category] || 0) + 1;
        levels[level] = (levels[level] || 0) + 1;

        if (samples.length < 5) {
          samples.push({
            fact_text: row.fact_text || '',
            category,
            confidential_level: level,
            summary: row.summary || '',
          });
        }
      } catch {
        invalid += 1;
      }
    }

    return {
      total: lines.length,
      valid,
      invalid,
      categories,
      levels,
      samples,
    };
  };

  const onFileChange = async (e) => {
    setErr('');
    setMsg('');
    setPreview(null);

    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.jsonl')) {
      setErr('请选择 .jsonl 文件');
      return;
    }

    const text = await file.text();

    setFileName(file.name);
    setContent(text);
    setPreview(parsePreview(text));
  };

  const submitImport = async () => {
    if (!content.trim()) {
      setErr('请先选择 JSONL 文件');
      return;
    }

    setImporting(true);
    setErr('');
    setMsg('');

    try {
      const data = await importConfidentialJsonl({
        content,
        filename: fileName,
        replace,
      });

      setMsg(
        `导入完成：成功 ${data.imported} 条，重复 ${data.duplicates} 条，错误 ${data.error_count} 行，当前总数 ${data.total_facts} 条`
      );

      if (onImported) {
        onImported(data);
      }
    } catch (e) {
      setErr(e.message || '导入失败');
    } finally {
      setImporting(false);
    }
  };

  return (
    <div className="import-box">
      <div className="import-title">保密库批量导入</div>

      <div className="import-desc">
        支持导入保密库-导入数据.jsonl。系统会把 fact_text、summary、keywords、confidential_level 转换为受保护事实。
      </div>

      <input
        className="file-input"
        type="file"
        accept=".jsonl,application/jsonl,text/plain"
        onChange={onFileChange}
      />

      {fileName && (
        <div className="file-name">
          已选择文件：{fileName}
        </div>
      )}

      <label className="check-line">
        <input
          type="checkbox"
          checked={replace}
          onChange={(e) => setReplace(e.target.checked)}
        />
        清空旧保密库后导入
      </label>

      {preview && (
        <div className="import-preview">
          <div>总行数：{preview.total}</div>
          <div>有效 JSON 行：{preview.valid}</div>
          <div>错误行：{preview.invalid}</div>

          <div className="preview-block">
            <b>密级统计：</b>
            {Object.entries(preview.levels).map(([k, v]) => (
              <span key={k} className="preview-pill">{k}: {v}</span>
            ))}
          </div>

          <div className="preview-block">
            <b>类别统计：</b>
            {Object.entries(preview.categories).map(([k, v]) => (
              <span key={k} className="preview-pill">{k}: {v}</span>
            ))}
          </div>

          <div className="preview-samples">
            <b>前 5 条预览：</b>
            {preview.samples.map((item, idx) => (
              <div key={idx} className="preview-sample">
                <div>{item.category} / {item.confidential_level}</div>
                <div>{item.fact_text}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {err && <div className="error-banner">⚠ {err}</div>}
      {msg && <div className="success-banner">✅ {msg}</div>}

      <button
        className="btn-send"
        onClick={submitImport}
        disabled={importing || !content.trim()}
      >
        {importing ? '正在导入...' : '开始导入保密库'}
      </button>
    </div>
  );
}

/* ================================================================
   Protected Fact Panel
   ================================================================ */
function ProtectedFactPanel({ currentScenario }) {
  const initialScenario = currentScenario === 'auto' ? 'healthcare' : currentScenario;

  const [factScenario, setFactScenario] = useState(initialScenario);
  const [schema, setSchema] = useState(null);
  const [factsInfo, setFactsInfo] = useState(null);
  const [form, setForm] = useState({});
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');

  useEffect(() => {
    if (currentScenario && currentScenario !== 'auto') {
      setFactScenario(currentScenario);
    }
  }, [currentScenario]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setErr('');
      setMsg('');
      try {
        const schemaData = await fetchFactSchema(factScenario);
        const factsData = await fetchProtectedFacts(factScenario);

        if (!cancelled) {
          setSchema(schemaData);
          setFactsInfo(factsData);
          setForm({});
        }
      } catch (e) {
        if (!cancelled) {
          setErr(e.message || '加载事实字段失败');
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, [factScenario]);

  const updateField = (name, value) => {
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const submit = async () => {
    setSaving(true);
    setErr('');
    setMsg('');

    try {
      const data = await addProtectedFact({
        scenario: factScenario,
        fact: form,
      });

      setMsg(`添加成功：${data.id}，当前事实数 ${data.count}`);
      setForm({});

      const factsData = await fetchProtectedFacts(factScenario);
      setFactsInfo(factsData);
    } catch (e) {
      setErr(e.message || '添加失败');
    } finally {
      setSaving(false);
    }
  };

  const fields = schema?.fields || [];
  const canSubmit = fields.some((f) => String(form[f.name] || '').trim());

  return (
    <div className="input-area">
      <div className="fact-title">受保护事实管理</div>
      <div className="fact-desc">
        新增事实会写入对应场景的事实池 JSON；哪些字段属于受保护字段，由策略文件 protected_fields 决定。
      </div>

      <label>业务场景</label>
      <select
        className="fact-select"
        value={factScenario}
        onChange={(e) => setFactScenario(e.target.value)}
      >
        {SCENARIOS.filter((s) => s.id !== 'auto').map((s) => (
          <option key={s.id} value={s.id}>
            {s.label}
          </option>
        ))}
      </select>

      {factScenario === 'confidential' && (
        <ConfidentialImportBox
          onImported={async () => {
            const schemaData = await fetchFactSchema('confidential');
            const factsData = await fetchProtectedFacts('confidential');
            setSchema(schemaData);
            setFactsInfo(factsData);
          }}
        />
      )}

      {factsInfo && (
        <div className="fact-count">
          当前场景已有事实：{factsInfo.count} 条
        </div>
      )}

      {err && <div className="error-banner">⚠ {err}</div>}
      {msg && <div className="success-banner">✅ {msg}</div>}

      <div className="fact-form-grid">
        {fields.map((f) => (
          <div
            key={f.name}
            className={`fact-field ${f.protected ? 'protected' : ''}`}
          >
            <label>
              {f.label}
              {f.protected && <span className="protected-badge">受保护</span>}
              {f.identifier && <span className="id-badge">锚点</span>}
            </label>

            <input
              value={form[f.name] || ''}
              placeholder={f.name === 'id' ? '不填则自动生成' : `请输入${f.label}`}
              onChange={(e) => updateField(f.name, e.target.value)}
            />
          </div>
        ))}
      </div>

      <div className="btn-row">
        <button
          className="btn-send"
          onClick={submit}
          disabled={saving || !canSubmit}
        >
          {saving ? '正在添加...' : '添加受保护事实'}
        </button>

        <button className="btn-clear" onClick={() => setForm({})}>
          清空表单
        </button>
      </div>
    </div>
  );
}

/* ================================================================
   LLM Payload Debug Panel
   ================================================================ */
function LlmPayloadPanel({ payload, error }) {
  if (!payload && !error) return null;

  return (
    <details className="llm-payload-panel">
      <summary>🧾 查看发送给 LLM 的真实请求内容</summary>

      {error && (
        <div className="error-banner">
          ⚠ {error}
        </div>
      )}

      {payload && (
        <>
          <div className="llm-payload-meta">
            <span>捕获时间：{payload.captured_at}</span>
            <span>模型：{payload.model}</span>
            <span>接口：{payload.base_url}</span>
          </div>

          <pre className="llm-payload-code">
            {JSON.stringify(payload.payload, null, 2)}
          </pre>
        </>
      )}
    </details>
  );
}

/* ================================================================
   App
   ================================================================ */
export default function App() {
  const [mode, setMode] = useState('chat');           // 'chat' | 'analyze'
  const [scenario, setScenario] = useState('auto');
  const [extractionMode, setExtractionMode] = useState('rule_only');
  const [secondaryCheck, setSecondaryCheck] = useState(false);

  // Inputs
  const [userInput, setUserInput] = useState('');
  const [modelOutput, setModelOutput] = useState('');

  // Output
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // LLM Payload debug
  const [llmPayload, setLlmPayload] = useState(null);
  const [llmPayloadError, setLlmPayloadError] = useState('');

  const handleSend = useCallback(async () => {
    if (!userInput.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);
    setLlmPayload(null);
    setLlmPayloadError('');

    try {
      let data;
      if (mode === 'chat') {
        data = await cfaChat({
          user_input: userInput,
          scenario,
          mode: extractionMode,
          secondary_check: secondaryCheck,
        });
      } else {
        if (!modelOutput.trim()) {
          throw new Error('请填写模型输出内容');
        }
        data = await cfaAnalyze({
          user_input: userInput,
          model_output: modelOutput,
          scenario,
          mode: extractionMode,
          secondary_check: secondaryCheck,
        });
      }
      setResult(data);

      // 只有对话模式会调用 LLM 生成，读取真实 payload
      if (mode === 'chat') {
        try {
          const payloadData = await fetchLastLlmPayload();
          setLlmPayload(payloadData);
        } catch (e) {
          setLlmPayloadError(e.message || '读取 LLM 请求内容失败');
        }
      }
    } catch (err) {
      setError(err.message || '请求失败');
    } finally {
      setLoading(false);
    }
  }, [mode, userInput, modelOutput, scenario, extractionMode, secondaryCheck]);

  const handleClear = useCallback(() => {
    setUserInput('');
    setModelOutput('');
    setResult(null);
    setError('');
  }, []);

  const fillQuickChat = (text) => {
    setMode('chat');
    setUserInput(text);
    setResult(null);
    setError('');
  };

  const fillQuickAnalyze = (ui, mo) => {
    setMode('analyze');
    setUserInput(ui);
    setModelOutput(mo);
    setResult(null);
    setError('');
  };

  const exportResult = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cfa-result-${result.request_id || 'export'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      {/* ================================================================
          Toolbar
          ================================================================ */}
      <div className="toolbar">
        <span className="brand">🛡 组合事实还原风险检测与防护平台</span>

        <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
          {SCENARIOS.map((s) => (
            <option key={s.id} value={s.id}>{s.label}</option>
          ))}
        </select>

        <select value={extractionMode} onChange={(e) => setExtractionMode(e.target.value)}>
          {MODES.map((m) => (
            <option key={m.id} value={m.id}>{m.label}</option>
          ))}
        </select>

        <label>
          <input
            type="checkbox"
            checked={secondaryCheck}
            onChange={(e) => setSecondaryCheck(e.target.checked)}
            disabled={extractionMode === 'rule_only'}
          />
          二次检测 (Mode 3)
        </label>

        <button onClick={handleClear} title="清空输入和结果">🗑 清空</button>
        <button onClick={exportResult} disabled={!result} title="导出结果为 JSON 文件">📥 导出</button>
      </div>

      {/* ================================================================
          Workspace
          ================================================================ */}
      <div className="workspace">
        {/* Left Panel: Input */}
        <div className="panel-left">
          {/* Tabs */}
          <div className="tabs">
            <div
              className={`tab ${mode === 'chat' ? 'active' : ''}`}
              onClick={() => setMode('chat')}
            >
              💬 对话模式
            </div>
            <div
              className={`tab ${mode === 'analyze' ? 'active' : ''}`}
              onClick={() => setMode('analyze')}
            >
              🔍 分析模式
            </div>
            <div
              className={`tab ${mode === 'facts' ? 'active' : ''}`}
              onClick={() => setMode('facts')}
            >
              📋 事实管理
            </div>
          </div>

          {/* Input Area */}
          {mode === 'facts' ? (
            <ProtectedFactPanel currentScenario={scenario} />
          ) : (
            <div className="input-area">
              <label>用户输入 / 问题</label>
              <textarea
                placeholder={mode === 'chat'
                  ? '输入你想问的问题，系统将调用 LLM 生成回答并进行 CFA 检测...'
                  : '输入用户的原始提问...'}
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                rows={mode === 'chat' ? 4 : 3}
              />
              {mode === 'chat' && (
                <div className="char-count">{userInput.length} 字符</div>
              )}

              {mode === 'analyze' && (
                <>
                  <label>模型原始输出</label>
                  <textarea
                    placeholder="粘贴大模型生成的原始回答..."
                    value={modelOutput}
                    onChange={(e) => setModelOutput(e.target.value)}
                    rows={5}
                  />
                  <div className="char-count">{modelOutput.length} 字符</div>
                </>
              )}

              <div className="btn-row">
                <button className="btn-send" onClick={handleSend} disabled={loading || !userInput.trim()}>
                  {loading ? '⏳ 处理中...' : '🚀 发送检测'}
                </button>
                <button className="btn-clear" onClick={handleClear}>清空</button>
              </div>

              {/* Quick Examples */}
              <div className="quick-examples">
                <div className="label">⚡ 快速示例</div>
                <div className="tags">
                  {mode === 'chat'
                    ? QUICK_EXAMPLES_CHAT.map((text, i) => (
                        <span key={i} className="tag" onClick={() => fillQuickChat(text)}>
                          示例 {i + 1}
                        </span>
                      ))
                    : QUICK_EXAMPLES_ANALYZE.map((ex, i) => (
                        <span key={i} className="tag" onClick={() => fillQuickAnalyze(ex.ui, ex.mo)}>
                          示例 {i + 1}
                        </span>
                      ))
                  }
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel: Output */}
        <div className="panel-right">
          <OutputSection loading={loading} error={error} result={result} />
          <LlmPayloadPanel payload={llmPayload} error={llmPayloadError} />
        </div>
      </div>
    </>
  );
}