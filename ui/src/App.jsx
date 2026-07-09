import { useState, useCallback } from 'react';
import { cfaChat, cfaAnalyze } from './api';
import DiffView from './DiffView';

/* ================================================================
   Constants
   ================================================================ */
const SCENARIOS = [
  { id: 'auto',       label: '🤖 自动识别' },
  { id: 'healthcare', label: '🏥 医疗健康' },
  { id: 'finance',    label: '💰 金融信贷' },
  { id: 'aerospace',  label: '🛰 航天测控' },
  { id: 'meetings',   label: '📋 会议室'   },
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

  const handleSend = useCallback(async () => {
    if (!userInput.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

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
          </div>

          {/* Input Area */}
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
        </div>

        {/* Right Panel: Output */}
        <div className="panel-right">
          <OutputSection loading={loading} error={error} result={result} />
        </div>
      </div>
    </>
  );
}