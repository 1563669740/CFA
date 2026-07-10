import { useMemo } from 'react';
import { computeDiff } from './diffUtils';

/**
 * Compute inline diff between raw_answer and safe_answer.
 * Renders them side-by-side with color highlighting:
 *   - Green background for identical text
 *   - Red background (strikethrough) for text removed from raw_answer
 *   - Blue background (underline) for text added in safe_answer
 */
export default function DiffView({
  rawAnswer,
  safeAnswer,
  rawLabel = '📝 LLM 原始输出（不一致部分标红）',
  safeLabel = '🔒 CFA 安全回答（不一致部分标蓝）',
}) {
  const diffSegments = useMemo(() => {
    if (!rawAnswer || !safeAnswer) return [];
    return computeDiff(rawAnswer, safeAnswer);
  }, [rawAnswer, safeAnswer]);

  // Build two column views from the unified diff
  const { leftNodes, rightNodes } = useMemo(() => {
    const left = [];
    const right = [];

    diffSegments.forEach((seg, idx) => {
      const key = `s${idx}`;
      if (seg.type === 'equal') {
        left.push(
          <span key={key + 'L'} className="diff-equal">{seg.text}</span>
        );
        right.push(
          <span key={key + 'R'} className="diff-equal">{seg.text}</span>
        );
      } else if (seg.type === 'delete') {
        left.push(
          <span key={key + 'L'} className="diff-delete">{seg.text}</span>
        );
        // Right side: empty placeholder (invisible) to keep alignment
        right.push(
          <span key={key + 'R'} className="diff-placeholder">{'\u200B'}</span>
        );
      } else if (seg.type === 'insert') {
        // Left side: empty placeholder
        left.push(
          <span key={key + 'L'} className="diff-placeholder">{'\u200B'}</span>
        );
        right.push(
          <span key={key + 'R'} className="diff-insert">{seg.text}</span>
        );
      }
    });

    return { leftNodes: left, rightNodes: right };
  }, [diffSegments]);

  if (!rawAnswer || !safeAnswer) {
    return null;
  }

  return (
    <div className="diff-container">
      <div className="diff-header">
        <span className="diff-label" style={{ color: 'var(--red)' }}>
          {rawLabel}
        </span>
        <span className="diff-label" style={{ color: 'var(--accent)' }}>
          {safeLabel}
        </span>
      </div>
      <div className="diff-panels">
        <div className="diff-panel diff-panel-left">
          {leftNodes}
        </div>
        <div className="diff-panel diff-panel-right">
          {rightNodes}
        </div>
      </div>
      <div className="diff-legend">
        <span className="legend-item">
          <span className="legend-swatch diff-equal-inline">相同</span>
          相同内容
        </span>
        <span className="legend-item">
          <span className="legend-swatch diff-delete-inline">删改</span>
          仅原始输出有（CFA已移除/改写）
        </span>
        <span className="legend-item">
          <span className="legend-swatch diff-insert-inline">新增</span>
          仅CFA安全回答有（脱敏替换/新增）
        </span>
      </div>
    </div>
  );
}