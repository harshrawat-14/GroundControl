"""M3 evaluation framework: offline benchmark evaluators, a CompositeEvaluator,
and the OnlineJudge used as the M2 runtime gate.

Public surface:
- app.evaluation.faithfulness.FaithfulnessEvaluator
- app.evaluation.groundedness.GroundednessEvaluator
- app.evaluation.relevance.RelevanceEvaluator
- app.evaluation.correctness.CorrectnessEvaluator
- app.evaluation.composite.CompositeEvaluator
- app.evaluation.online_judge.OnlineJudge  (stable interface for M2's orchestrator)
- app.evaluation.golden_dataset.{load_golden_dataset, run_benchmark, resolve_dataset_path}
"""
