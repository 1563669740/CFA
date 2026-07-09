from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import DeepSeekAdapter
from .deepseek import _assert_no_confidential_prompt
from .engine import CFAScoreEngine
from .knowledge import dump_json, load_assets, load_policy, load_public_knowledge, merge_public_knowledge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfa-score",
        description="Detect combination fact restoration risk from user input and model output.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a user-input turn")
    analyze.add_argument("--input", help="Path to user input text file")
    analyze.add_argument("--input-text", help="User input text directly from command line")
    analyze.add_argument(
        "--model-output",
        help="Path to a pre-existing model answer text file (skip DeepSeek generation).",
    )
    analyze.add_argument(
        "--model-output-text",
        help="Pre-existing model answer text (skip DeepSeek generation).",
    )
    analyze.add_argument("--facts", required=True, help="Path to restricted fact pool JSON")
    analyze.add_argument("--policy", required=True, help="Path to field policy JSON")
    analyze.add_argument("--public-knowledge", help="Optional public knowledge rules JSON")
    analyze.add_argument("--env", default=".env", help="Path to .env for DeepSeek settings")
    analyze.add_argument("--system-prompt", help="Override system prompt for DeepSeek generation")
    analyze.add_argument("--out", help="Optional path to write full JSON report")
    analyze.add_argument(
        "--print",
        choices=["summary", "json", "safe", "x"],
        default="summary",
        help="What to print to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        user_input = _read_optional_text(args.input, args.input_text, "input")
        assets = load_assets(args.facts)
        public_rules = load_public_knowledge(args.public_knowledge) if args.public_knowledge else []
        policy = merge_public_knowledge(load_policy(args.policy), public_rules)

        # ---- Resolve model output ----
        model_output = _read_optional_text(args.model_output, args.model_output_text, "model-output")
        if model_output:
            answer_mode = "provided"
        else:
            # No pre-existing answer → generate via DeepSeek
            sp = getattr(args, "system_prompt", None) or None
            adapter = DeepSeekAdapter(env_path=args.env, system_prompt=sp)
            model_output = adapter.generate(
                user_input=user_input,
                context={
                    "fact_pool": assets,
                    "public_knowledge": public_rules,
                    "policy": policy,
                    # P0 安全：通过 _assert_no_confidential_prompt 在底层拦截
                    # 但如果调用方确实在使用 confidential assets，底层会直接阻断
                    "allow_fact_pool_to_llm": True,
                },
            )
            answer_mode = "deepseek"

        engine = CFAScoreEngine(assets, policy)
        result = engine.analyze(model_output, user_input=user_input)

        report = result.to_dict()
        report["answer_mode"] = answer_mode
        report["public_knowledge_rules"] = public_rules

        if args.out:
            dump_json(report, args.out)

        if args.print == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.print == "safe":
            print(result.safe_answer)
        elif args.print == "x":
            print(result.x_replaced_answer)
        else:
            _print_summary(report)
        return 0

    parser.error("unknown command")
    return 2


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _read_optional_text(path: str | None, text: str | None, name: str) -> str:
    if path and text:
        raise SystemExit(f"Please provide only one of --{name} or --{name}-text")
    if path:
        return Path(path).read_text(encoding="utf-8-sig")
    return text or ""


def _print_summary(report: dict) -> None:
    print("== CFA-Score Summary ==")
    if report.get("user_input"):
        print("turn: user input + model output")
    else:
        print("turn: model output only")
    print(f"answer mode: {report.get('answer_mode', 'provided')}")
    print(f"anchors: {len(report['anchors'])}")
    print(f"findings: {len(report['findings'])}")
    for i, finding in enumerate(report["findings"], 1):
        print(
            f"\n[{i}] {finding['risk_level']} score={finding['score']} "
            f"target={finding['target_asset_id']} {finding['target_asset_name']}"
        )
        print(f"restored: {finding['restored_fact']}")
        if finding.get("key_anchor_summary"):
            print("key anchors:")
            for item in finding["key_anchor_summary"]:
                print(f"  * {item}")
        print("reason:", finding["reason"])
        print("reduction:")
        for step in finding["reduction_chain"]:
            print(
                f"  - {step['field_label']}={step['canonical_value']}: "
                f"{step['before_count']} -> {step['after_count']} ({','.join(step['remaining_asset_ids'])})"
            )
    print("\n== X replaced answer ==")
    print(report["x_replaced_answer"])
    print("\n== Safe answer ==")
    print(report["safe_answer"])


if __name__ == "__main__":
    raise SystemExit(main())