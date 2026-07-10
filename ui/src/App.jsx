import { useState, useCallback, useEffect, useMemo } from 'react';
import {
  cfaChat,
  cfaAnalyze,
  fetchFactSchema,
  fetchProtectedFacts,
  addProtectedFact,
  importConfidentialJsonl,
  fetchLastLlmPayload,
  fetchConfidentialKb,
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
  domain_confidential: '🔒 内部敏感查询',
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

const FINDING_TYPE_LABELS = {
  direct_protected_disclosure: '直接披露受保护字段',
  indirect_asset_restoration: '组合定位到单条事实',
  indirect_protected_value_restoration: '受保护字段值收敛',
  input_hypothesis_confirmation: '用户假设被模型确认',
  direct_confidential_value_match: '本地直接命中保密值',
};

function formatNumber(value, digits = 2) {
  if (value === undefined || value === null || value === '') return '-';
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : String(value);
}

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
  const isConfidential = result.routed_scenario === 'confidential';
  const publicFindings = (!isConfidential && Array.isArray(result.findings_summary))
    ? result.findings_summary
    : [];
  const displayedRawAnswer = isConfidential
    ? (result.demo_raw_answer || result.demo_raw_answer_redacted)
    : result.raw_answer;
  const confidentialRawIsRedacted = isConfidential && !result.demo_raw_answer && Boolean(result.demo_raw_answer_redacted);
  const rawAnswerTitle = isConfidential
    ? (confidentialRawIsRedacted ? '📝 LLM 原始回答（脱敏演示）' : '📝 LLM 原始回答（未拦截）')
    : '📝 LLM 原始回答';
  const rawDiffLabel = isConfidential
    ? (confidentialRawIsRedacted
      ? '📝 LLM 原始回答（脱敏演示，不一致部分标红）'
      : '📝 LLM 原始输出（未拦截，不一致部分标红）')
    : '📝 LLM 原始输出（不一致部分标红）';
  const hasDiffView = Boolean(displayedRawAnswer && result.answer);
  const contextSummary = result.confidential_context_summary;
  const cfaEvidence = Array.isArray(result.confidential_cfa_evidence)
    ? result.confidential_cfa_evidence
    : [];

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

      {isConfidential && result.risk_detected && (
        <div className="answer-card" style={{ borderLeft: '3px solid var(--red)' }}>
          <div className="card-title">🛡️ 保密场景已拦截</div>
          <div className="card-body" style={{ fontSize: '0.88rem' }}>
            本次回答可能使保密事实被唯一还原。最终答复已由 CFA/本地闸门改写为安全响应；下方实验对照卡片可展示 CFA 还原出的受限事实。
          </div>
        </div>
      )}

      {isConfidential && contextSummary && (
        <div className="answer-card confidential-context-card">
          <div className="card-title">
            {contextSummary.kb_type === 'progressive_confidential_internal_kb'
              ? '🧩 LLM 分步加载的内部知识库摘要'
              : '🧩 LLM 携带的安全知识库摘要'}
          </div>
          <div className="context-badges">
            <span className="context-badge sanitized">
              {contextSummary.kb_type === 'progressive_confidential_internal_kb' ? '分步按需加载' : '仅脱敏摘要'}
            </span>
            <span className="context-badge not-sent">原始事实不整体发送给 LLM</span>
            <span className="context-badge backend-only">命中证据仅后端审计</span>
          </div>
          <div className="context-grid">
            <div><strong>类型：</strong>{contextSummary.kb_type}</div>
            <div><strong>记录总数：</strong>{contextSummary.total_records}</div>
            {contextSummary.kb_type === 'progressive_confidential_internal_kb' ? (
              <>
                <div><strong>目录发送条数：</strong>{contextSummary.catalog_entries_sent}</div>
                <div><strong>选择来源：</strong>{contextSummary.selection_source}</div>
                <div><strong>已选 KB 数：</strong>{contextSummary.selected_kb_count}</div>
                <div><strong>已选 content_units：</strong>{contextSummary.selected_content_unit_count}</div>
                <div><strong>阶段1状态：</strong>{contextSummary.stage1_status}</div>
                <div><strong>阶段2状态：</strong>{contextSummary.stage2_status}</div>
              </>
            ) : (
              <>
                <div><strong>相关：</strong>{contextSummary.current_query?.related ? '是' : '否'}</div>
                <div><strong>安全主题：</strong>{contextSummary.current_query?.safe_topic}</div>
                <div><strong>命中数量桶：</strong>{contextSummary.current_query?.matched_count_bucket}</div>
              </>
            )}
          </div>
          {contextSummary.coverage && (
            <div className="context-coverage">
              {Object.entries(contextSummary.coverage).map(([key, value]) => (
                <span key={key}>{key}: {value}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {isConfidential && cfaEvidence.length > 0 && (
        <div className="answer-card cfa-evidence-card">
          <div className="card-title">🧬 CFA 还原的受限事实（实验对照）</div>
          <div className="safe-note">
            该区域用于本地实验对照：展示 CFA 根据“用户输入 + LLM 原始候选回答”还原出的受限事实，便于与左侧保密库核对。
          </div>
          {cfaEvidence.map((item, idx) => (
            <div className="cfa-evidence-item" key={`${item.target_id || 'finding'}-${idx}`}>
              <div className="cfa-evidence-head">
                <span className={`evidence-level ${RISK_CLASS(item.risk_level)}`}>{item.risk_level}</span>
                <strong>{item.target_name || item.target_id || `风险发现 #${idx + 1}`}</strong>
                <span>{FINDING_TYPE_LABELS[item.finding_type] || item.finding_type}</span>
                <span>得分 {formatNumber(item.score, 1)}</span>
              </div>
              {item.restored_fact && (
                <div className="restored-fact"><strong>还原事实：</strong>{item.restored_fact}</div>
              )}
              {Array.isArray(item.restored_fields) && item.restored_fields.length > 0 && (
                <div className="evidence-fields">
                  {item.restored_fields.map((field) => (
                    <span className="field-chip" key={`${field.name}-${field.value}`}>
                      {field.label || field.name}: {field.value || '-'}
                    </span>
                  ))}
                </div>
              )}
              {item.reason && <div className="evidence-reason">{item.reason}</div>}
              {Array.isArray(item.reasoning_process) && item.reasoning_process.length > 0 && (
                <div className="evidence-reasoning">
                  <strong>CFA 推理过程：</strong>
                  <ol>
                    {item.reasoning_process.map((step, i2) => <li key={`${step}-${i2}`}>{step}</li>)}
                  </ol>
                </div>
              )}
              <div className="evidence-anchors">
                <strong>关键锚点：</strong>
                {Array.isArray(item.key_anchors) && item.key_anchors.length > 0 ? (
                  item.key_anchors.map((anchor, i2) => <span key={`${anchor}-${i2}`}>{anchor}</span>)
                ) : (
                  <span className="empty-anchor">
                    无{item.anchor_status_reason ? `——${item.anchor_status_reason}` : '——当前仅检测到受保护值泄露，尚未通过删除锚点反事实验证唯一还原。'}
                  </span>
                )}
              </div>
              <div className="evidence-stats">
                <span>输入候选：{item.input_candidate_count || 0}</span>
                <span>最终候选：{item.final_candidate_count || 0}</span>
                <span>信息增益：{formatNumber(item.information_gain_bits)} bit</span>
              </div>
              {Array.isArray(item.reduction_chain) && item.reduction_chain.length > 0 && (
                <div className="reduction-table">
                  {item.reduction_chain.map((step, i2) => (
                    <div className="reduction-row" key={`${step.field_name}-${step.canonical_value}-${i2}`}>
                      <span>{step.field_label || step.field_name}</span>
                      <span>{step.match_symbol || '='}{step.canonical_value}</span>
                      <span>{step.before_count} → {step.after_count}</span>
                      <span>{(step.remaining_asset_ids || []).join(', ')}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Risk Detail Cards */}
      {publicFindings.map((f, idx) => (
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
      {hasDiffView && (
        <DiffView
          rawAnswer={displayedRawAnswer}
          safeAnswer={result.answer}
          rawLabel={rawDiffLabel}
          safeLabel="🔒 CFA 安全回答（不一致部分标蓝）"
        />
      )}

      {isConfidential && !displayedRawAnswer && (
        <div className="answer-card" style={{ borderLeft: '3px solid var(--yellow)' }}>
          <div className="card-title">📝 LLM 原始回答未展开</div>
          <div className="card-body" style={{ fontSize: '0.88rem' }}>
            保密场景默认不返回原始回答。如需演示差异，请启用左侧“展示 LLM 原始回答”。
          </div>
        </div>
      )}

      {/* Fallback single-answer cards when diff cannot be shown */}
      {!hasDiffView && displayedRawAnswer && (
        <div className={`answer-card ${isConfidential ? 'redacted-raw-card' : ''}`}>
          <div className="card-title">
            {rawAnswerTitle}
            <button className="copy-btn" onClick={() => copyText(displayedRawAnswer)}>📋 复制</button>
          </div>
          {isConfidential && (
            <div className="safe-note">
              {confidentialRawIsRedacted
                ? '这里展示的是后端脱敏后的演示文本。'
                : '这里展示 CFA 检测前的 LLM 原始候选回答，未经过最终本地闸门替换。'}
            </div>
          )}
          <div className="card-body">{displayedRawAnswer}</div>
        </div>
      )}

      {!hasDiffView && result.answer && (
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
        {isConfidential && (
          <div className="hint" style={{ marginBottom: 8 }}>
            保密场景默认不返回原始输出；本地实验模式会按需返回 LLM 原始候选回答和 CFA 还原证据用于页面对照。
          </div>
        )}
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

function formatFactValue(value) {
  if (Array.isArray(value)) return value.join('；');
  if (value && typeof value === 'object') return JSON.stringify(value, null, 2);
  return String(value ?? '');
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
          <div>当前场景已有事实：{factsInfo.count} 条</div>
          {factsInfo.import_meta && (
            <div className="import-meta">
              <div>
                最近一次导入：成功 {factsInfo.import_meta.imported} 条，
                重复 {factsInfo.import_meta.duplicates} 条，
                错误 {factsInfo.import_meta.error_count} 行，
                导入后总数 {factsInfo.import_meta.total_facts} 条
              </div>
              <div>
                文件：{factsInfo.import_meta.filename || '未命名'}；
                时间：{factsInfo.import_meta.imported_at || '-'}
              </div>
            </div>
          )}
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
   Confidential KB Browser
   ================================================================ */
function ConfidentialKbBrowser() {
  const [schema, setSchema] = useState(null);
  const [factsInfo, setFactsInfo] = useState(null);
  const [internalKbInfo, setInternalKbInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [level, setLevel] = useState('all');
  const [maskProtected, setMaskProtected] = useState(false);
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setErr('');
      try {
        const [schemaData, factsData, kbData] = await Promise.all([
          fetchFactSchema('confidential'),
          fetchProtectedFacts('confidential'),
          fetchConfidentialKb(),
        ]);

        if (!cancelled) {
          setSchema(schemaData);
          setFactsInfo(factsData);
          setInternalKbInfo(kbData);
        }
      } catch (e) {
        if (!cancelled) setErr(e.message || '加载保密库失败');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const facts = factsInfo?.facts || [];
  const fields = schema?.fields || [];
  const protectedFields = new Set(schema?.protected_fields || []);

  const internalKbMap = useMemo(() => {
    const records = internalKbInfo?.records || [];

    return new Map(
      records.map((record) => [
        record.kb_id,
        record,
      ])
    );
  }, [internalKbInfo]);

  const categories = useMemo(() => {
    return Array.from(new Set(facts.map((f) => f.category || '未分类'))).sort();
  }, [facts]);

  const levels = useMemo(() => {
    return Array.from(new Set(facts.map((f) => f.confidential_level || 'unknown'))).sort();
  }, [facts]);

  const filteredFacts = useMemo(() => {
    const q = query.trim().toLowerCase();
    return facts.filter((fact) => {
      if (category !== 'all' && (fact.category || '未分类') !== category) return false;
      if (level !== 'all' && (fact.confidential_level || 'unknown') !== level) return false;
      if (!q) return true;
      return JSON.stringify(fact, null, 0).toLowerCase().includes(q);
    });
  }, [facts, query, category, level]);

  const visibleFields = fields.length
    ? fields
    : [
        { name: 'id', label: '事实编号', protected: false },
        { name: 'category', label: '类别', protected: false },
        { name: 'confidential_level', label: '密级', protected: true },
        { name: 'secret_summary', label: '摘要', protected: true },
        { name: 'secret_content', label: '内容', protected: true },
        { name: 'secret_keywords', label: '关键词', protected: true },
      ];

  const renderValue = (fact, field) => {
    const value = formatFactValue(fact[field.name]);
    if (!value) return <span className="dim-text">-</span>;
    if (maskProtected && (field.protected || protectedFields.has(field.name))) {
      return <span className="masked-value">[已遮罩]</span>;
    }
    return value;
  };

  return (
    <div className="input-area kb-browser">
      <div className="kb-header">
        <div>
          <div className="fact-title">保密库浏览</div>
          <div className="fact-desc">
            管理员/演示视图：此处同时展示 CFA 检测事实及其对应内部知识库。
            问答时第一阶段只发送知识库目录，第二阶段只发送模型选中的
            content_units，不会整体发送全部内部知识库。
          </div>
        </div>
        <div className="kb-stat-card">
          <span>当前事实</span>
          <strong>{factsInfo?.count ?? 0}</strong>
        </div>
        {internalKbInfo && (
          <div className="kb-stat-card" style={{ marginLeft: 8 }}>
            <span>内部知识库文档</span>
            <strong>{internalKbInfo.count ?? 0}</strong>
          </div>
        )}
      </div>

      {factsInfo?.import_meta && (
        <div className="kb-import-meta">
          <span>最近导入文件：{factsInfo.import_meta.filename || '未命名'}</span>
          <span>时间：{factsInfo.import_meta.imported_at || '-'}</span>
          <span>成功：{factsInfo.import_meta.imported}</span>
          <span>重复：{factsInfo.import_meta.duplicates}</span>
          <span>错误：{factsInfo.import_meta.error_count}</span>
        </div>
      )}

      <div className="kb-warning">
        <strong>安全边界：</strong>此页用于人工核对导入结果；外部 LLM 不携带 secret_content、secret_summary、secret_keywords、SEC ID 或还原链。
      </div>

      <div className="kb-filters">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="搜索事实内容、摘要、关键词或 ID..."
        />
        <select value={category} onChange={(e) => setCategory(e.target.value)}>
          <option value="all">全部类别</option>
          {categories.map((item) => <option key={item} value={item}>{item}</option>)}
        </select>
        <select value={level} onChange={(e) => setLevel(e.target.value)}>
          <option value="all">全部密级</option>
          {levels.map((item) => <option key={item} value={item}>{item}</option>)}
        </select>
        <label className="mask-toggle">
          <input
            type="checkbox"
            checked={maskProtected}
            onChange={(e) => setMaskProtected(e.target.checked)}
          />
          遮罩受保护字段
        </label>
      </div>

      {loading && <div className="hint">正在加载保密库...</div>}
      {err && <div className="error-banner">⚠ {err}</div>}

      {!loading && !err && (
        <div className="kb-table-wrap">
          <table className="kb-table">
            <thead>
              <tr>
                <th>展开</th>
                {visibleFields.map((field) => (
                  <th key={field.name}>{field.label || field.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredFacts.map((fact, idx) => {
                const rowKey = fact.id || `row-${idx}`;

                const relatedKb = fact.source_kb_id
                  ? internalKbMap.get(fact.source_kb_id)
                  : null;

                return (
                  <tr key={rowKey}>
                    <td>
                      <button
                        className="expand-btn"
                        onClick={() => setExpanded((prev) => ({ ...prev, [rowKey]: !prev[rowKey] }))}
                      >
                        {expanded[rowKey] ? '收起' : '展开'}
                      </button>
                      {expanded[rowKey] && (
                        <div className="kb-row-detail">
                          <div><strong>攻击改写：</strong>{formatFactValue(fact.attack_paraphrases) || '-'}</div>
                          <div><strong>负样本：</strong>{formatFactValue(fact.negative_samples) || '-'}</div>

                          <div className="related-internal-kb">
                            <div className="related-kb-title">
                              <strong>对应内部知识库：</strong>
                              {relatedKb
                                ? `${relatedKb.title}（${relatedKb.kb_id}）`
                                : '未建立关联'}
                            </div>

                            {relatedKb && (
                              <>
                                <div className="related-kb-meta">
                                  部门：
                                  {relatedKb.metadata?.department || '-'}
                                  {'；'}
                                  日期：
                                  {relatedKb.metadata?.date || '-'}
                                </div>

                                <div className="related-kb-units">
                                  {(relatedKb.content_units || []).map((unit) => (
                                    <div
                                      className="related-kb-unit"
                                      key={`${relatedKb.kb_id}-${unit.unit_id}`}
                                    >
                                      <div>
                                        <strong>
                                          {unit.unit_id || '内容单元'}
                                        </strong>
                                        {unit.role && ` · ${unit.role}`}
                                      </div>

                                      <div>{unit.text}</div>
                                    </div>
                                  ))}
                                </div>

                                <details>
                                  <summary>查看内部知识库原始结构</summary>
                                  <pre>
                                    {JSON.stringify(relatedKb, null, 2)}
                                  </pre>
                                </details>
                              </>
                            )}
                          </div>

                          <details>
                            <summary>查看保密事实原始 JSON</summary>
                            <pre>{JSON.stringify(fact, null, 2)}</pre>
                          </details>
                        </div>
                      )}
                    </td>
                    {visibleFields.map((field) => (
                      <td key={field.name} className={field.protected ? 'protected-cell' : ''}>
                        {renderValue(fact, field)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
          {filteredFacts.length === 0 && (
            <div className="output-empty" style={{ minHeight: 160 }}>
              <div className="hint">没有匹配的保密事实</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ================================================================
   LLM Payload Debug Panel
   ================================================================ */
function llmPayloadStageTitle(item) {
  const purpose = item?.ref?.purpose || item?.payload?.purpose || '';
  if (purpose === 'confidential_kb_selection') {
    return '第一阶段：KB 简介/目录选择请求';
  }
  if (purpose === 'confidential_answer_generation') {
    return '第二阶段：已选 content_units 回答生成请求';
  }
  if (purpose === 'primary_generation') {
    return '主回答生成请求';
  }
  return purpose || 'LLM 请求';
}

function LlmPayloadPanel({ items = [], confidential }) {
  const visibleItems = Array.isArray(items) ? items : [];
  if (visibleItems.length === 0 && !confidential) return null;

  return (
    <details className="llm-payload-panel">
      <summary>🧾 查看发送给 LLM 的真实请求内容</summary>

      {confidential && (
        <div className="hint" style={{ marginBottom: 8 }}>
          保密场景的调试 payload 可能包含按需加载后的模拟内部 KB content_units，
          仅用于本地实验与排查；公开回答仍由后端安全门控处理。
        </div>
      )}

      {visibleItems.length === 0 && (
        <div className="error-banner">
          ⚠ 本次请求没有捕获到对应的 LLM 请求。可能未开启 CFA_DEBUG_LLM_PAYLOAD。
        </div>
      )}

      {visibleItems.map((item, idx) => {
        const payload = item.payload;
        return (
          <div className="llm-payload-stage" key={`${item.ref?.purpose || idx}-${idx}`}>
            <div className="llm-payload-stage-title">
              <span>{llmPayloadStageTitle(item)}</span>
              {item.ref?.stage && <span className="llm-payload-stage-badge">Stage {item.ref.stage}</span>}
            </div>

            {item.error && (
              <div className="error-banner llm-payload-stage-error">
                ⚠ {item.error}
              </div>
            )}

            {payload && (
              <>
                <div className="llm-payload-meta">
                  <span>捕获时间：{payload.captured_at}</span>
                  <span>请求 ID：{payload.request_id || '-'}</span>
                  <span>调用目的：{payload.purpose || '-'}</span>
                  <span>场景：{payload.scenario || '-'}</span>
                  <span>实际路由：{payload.effective_scenario || '-'}</span>
                  <span>模式：{payload.mode || '-'}</span>
                  <span>二次检测：{payload.secondary_check ? '是' : '否'}</span>
                  <span>注入事实池：{payload.inject_fact_pool ? '是' : '否'}</span>
                  <span>安全知识库：{payload.safe_knowledge_type || '无'}</span>
                  {payload.selection_source && <span>选择来源：{payload.selection_source}</span>}
                  {payload.selected_content_unit_count !== undefined && (
                    <span>已选 units：{payload.selected_content_unit_count}</span>
                  )}
                  <span>模型：{payload.model}</span>
                  <span>接口：{payload.base_url}</span>
                </div>

                <pre className="llm-payload-code">
                  {JSON.stringify(payload.payload, null, 2)}
                </pre>
              </>
            )}
          </div>
        );
      })}
    </details>
  );
}

/* ================================================================
   App
   ================================================================ */
export default function App() {
  const [mode, setMode] = useState('chat');           // 'chat' | 'analyze' | 'facts' | 'confidentialKb'
  const [scenario, setScenario] = useState('auto');
  const [extractionMode, setExtractionMode] = useState('rule_only');
  const [secondaryCheck, setSecondaryCheck] = useState(false);
  const [confidentialRawDemo, setConfidentialRawDemo] = useState(true);
  const [confidentialCfaEvidence, setConfidentialCfaEvidence] = useState(true);

  // Inputs
  const [userInput, setUserInput] = useState('');
  const [modelOutput, setModelOutput] = useState('');

  // Output
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // LLM Payload debug
  const [llmPayloadItems, setLlmPayloadItems] = useState([]);

  const handleSend = useCallback(async () => {
    if (!userInput.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);
    setLlmPayloadItems([]);

    try {
      let data;
      if (mode === 'chat') {
        data = await cfaChat({
          user_input: userInput,
          scenario,
          mode: extractionMode,
          secondary_check: secondaryCheck,
          include_confidential_raw_demo: scenario === 'confidential' && confidentialRawDemo,
          confidential_raw_demo_mode: 'raw',
          include_confidential_cfa_evidence: scenario === 'confidential' && confidentialCfaEvidence,
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
          include_confidential_cfa_evidence: scenario === 'confidential' && confidentialCfaEvidence,
        });
      }
      setResult(data);

      // 只有对话模式会调用 LLM 生成，按当前 request_id + purpose 读取真实 payload。
      if (mode === 'chat') {
        const refs = Array.isArray(data.llm_debug_refs) ? data.llm_debug_refs : [];
        const sortedRefs = [...refs].sort((a, b) => (a.stage || 99) - (b.stage || 99));
        if (sortedRefs.length > 0) {
          const settled = await Promise.allSettled(
            sortedRefs.map((ref) => fetchLastLlmPayload({
              requestId: ref.request_id || data.request_id,
              purpose: ref.purpose || 'primary_generation',
            }))
          );
          setLlmPayloadItems(sortedRefs.map((ref, idx) => ({
            ref,
            payload: settled[idx].status === 'fulfilled' ? settled[idx].value : null,
            error: settled[idx].status === 'rejected'
              ? (settled[idx].reason?.message || '本阶段没有捕获到对应的 LLM 请求')
              : '',
          })));
        }
      }
    } catch (err) {
      setError(err.message || '请求失败');
    } finally {
      setLoading(false);
    }
  }, [mode, userInput, modelOutput, scenario, extractionMode, secondaryCheck, confidentialRawDemo, confidentialCfaEvidence]);

  const handleClear = useCallback(() => {
    setUserInput('');
    setModelOutput('');
    setResult(null);
    setError('');
    setLlmPayloadItems([]);
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
      <div className={`workspace ${mode === 'confidentialKb' ? 'kb-workspace' : ''}`}>
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
            <div
              className={`tab ${mode === 'confidentialKb' ? 'active' : ''}`}
              onClick={() => setMode('confidentialKb')}
            >
              🛡️ 保密库
            </div>
          </div>

          {/* Input Area */}
          {mode === 'facts' ? (
            <ProtectedFactPanel currentScenario={scenario} />
          ) : mode === 'confidentialKb' ? (
            <ConfidentialKbBrowser />
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

              {scenario === 'confidential' && (
                <div className="demo-toggle-group">
                  {mode === 'chat' && (
                    <label className="demo-toggle">
                      <input
                        type="checkbox"
                        checked={confidentialRawDemo}
                        onChange={(e) => setConfidentialRawDemo(e.target.checked)}
                      />
                      展示 LLM 原始回答（检测前，不拦截）
                    </label>
                  )}
                  <label className="demo-toggle">
                    <input
                      type="checkbox"
                      checked={confidentialCfaEvidence}
                      onChange={(e) => setConfidentialCfaEvidence(e.target.checked)}
                    />
                    展示 CFA 还原受限事实（本地实验对照）
                  </label>
                </div>
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
          <LlmPayloadPanel
            items={llmPayloadItems}
            confidential={result?.routed_scenario === 'confidential'}
          />
        </div>
      </div>
    </>
  );
}