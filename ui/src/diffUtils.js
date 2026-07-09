/**
 * Character-level text diff using LCS (Longest Common Subsequence).
 * Produces an array of { type: 'equal' | 'delete' | 'insert', text } segments.
 */

const SEPARATORS = /([，。！？、；：""''（）《》【】\n\r\s])/;

/**
 * Tokenize Chinese + mixed text into meaningful tokens.
 * Splits on Chinese punctuation but keeps punctuation as separate tokens.
 */
function tokenize(text) {
  const tokens = [];
  let buf = '';
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (SEPARATORS.test(ch)) {
      if (buf.length > 0) {
        tokens.push(buf);
        buf = '';
      }
      tokens.push(ch);
    } else {
      buf += ch;
    }
  }
  if (buf.length > 0) tokens.push(buf);
  return tokens;
}

/**
 * Compute LCS table for two token arrays.
 * Returns the dp table and the LCS length.
 */
function computeLCS(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  return dp;
}

/**
 * Backtrack through LCS table to produce diff segments.
 * Returns array of { type: 'equal' | 'delete' | 'insert', text: string }.
 */
function backtrack(a, b, dp) {
  const segments = [];
  let i = a.length;
  let j = b.length;

  // Collect in reverse, then reverse at the end
  const stack = [];

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      stack.push({ type: 'equal', text: a[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      stack.push({ type: 'insert', text: b[j - 1] });
      j--;
    } else if (i > 0 && (j === 0 || dp[i][j - 1] < dp[i - 1][j])) {
      stack.push({ type: 'delete', text: a[i - 1] });
      i--;
    }
  }

  // Reverse the stack
  for (let k = stack.length - 1; k >= 0; k--) {
    segments.push(stack[k]);
  }

  return segments;
}

/**
 * Merge consecutive segments of the same type.
 */
function mergeSegments(segments) {
  if (segments.length === 0) return [];
  const merged = [segments[0]];
  for (let i = 1; i < segments.length; i++) {
    const last = merged[merged.length - 1];
    if (last.type === segments[i].type) {
      last.text += segments[i].text;
    } else {
      merged.push(segments[i]);
    }
  }
  return merged;
}

/**
 * Compute diff between two strings.
 * Returns array of { type: 'equal' | 'delete' | 'insert', text: string }.
 *
 * 'equal'  — same text in both strings (will be green)
 * 'delete' — text only in oldText (will be red / strikethrough)
 * 'insert' — text only in newText (will be blue / highlight)
 */
export function computeDiff(oldText, newText) {
  if (!oldText && !newText) return [];
  if (!oldText) return [{ type: 'insert', text: newText }];
  if (!newText) return [{ type: 'delete', text: oldText }];

  const oldTokens = tokenize(oldText);
  const newTokens = tokenize(newText);
  const dp = computeLCS(oldTokens, newTokens);
  const segments = backtrack(oldTokens, newTokens, dp);
  return mergeSegments(segments);
}

/**
 * Quick equality check for same strings — skip expensive diff.
 */
export function areSameText(a, b) {
  return a === b;
}