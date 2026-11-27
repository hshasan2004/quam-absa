# QuAM-FT queue (sequential). Skips runs that already have metrics.json. Log: runs/queue_log.txt
# Usage: powershell -File scripts\run_ft.ps1 [job names...]
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "C:\Users\USER\.venvs\quam-absa\Scripts\python.exe"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$P = @("--prompt-aspects")
# ftp_* = main model (aspect-prompted QuAM-FT), 3 seeds; ft_* = no-prompt ablation, seed 42
$jobs = @(
  @("ftp_zeroshot",    "zeroshot", 42, $P),
  @("ftp_grouped",     "grouped",  42, $P),
  @("ftp_official_s1", "official", 1,  $P),
  @("ftp_official_s2", "official", 2,  $P),
  @("ftp_zeroshot_s1", "zeroshot", 1,  $P),
  @("ftp_zeroshot_s2", "zeroshot", 2,  $P),
  @("ftp_grouped_s1",  "grouped",  1,  $P),
  @("ftp_grouped_s2",  "grouped",  2,  $P),
  @("ft_zeroshot",     "zeroshot", 42, @()),
  @("ft_grouped",      "grouped",  42, @()),
  # BABSA (grouped split), aspect-prompted, two encoders
  @("ftp_babsa_banglabert",    "grouped", 42, (@("--data", "babsa", "--model", "csebuetnlp/banglabert") + $P)),
  @("ftp_babsa_xlmr",          "grouped", 42, (@("--data", "babsa") + $P)),
  @("ftp_babsa_banglabert_s1", "grouped", 1,  (@("--data", "babsa", "--model", "csebuetnlp/banglabert") + $P)),
  @("ftp_babsa_banglabert_s2", "grouped", 2,  (@("--data", "babsa", "--model", "csebuetnlp/banglabert") + $P)),
  # ablation: no xLSTM memory (direct scorer only)
  @("ftp_nomem_official", "official", 42, (@("--direct", "only") + $P)),
  @("ftp_nomem_zeroshot", "zeroshot", 42, (@("--direct", "only") + $P)),
  # partly / fully frozen encoder: does the xLSTM memory matter when the encoder is not fully tuned?
  @("ftp_f9_official",        "official", 42, (@("--freeze-layers", "9") + $P)),
  @("ftp_nomem_f9_official",  "official", 42, (@("--freeze-layers", "9", "--direct", "only") + $P)),
  @("ftp_f12_official",       "official", 42, (@("--freeze-layers", "12") + $P)),
  @("ftp_nomem_f12_official", "official", 42, (@("--freeze-layers", "12", "--direct", "only") + $P)),
  # zero-shot: seeds for the no-memory ablation, and a memory-only reader (no direct scorer)
  @("ftp_nomem_zeroshot_s1",  "zeroshot", 1,  (@("--direct", "only") + $P)),
  @("ftp_nomem_zeroshot_s2",  "zeroshot", 2,  (@("--direct", "only") + $P)),
  @("ftp_memonly_zeroshot",   "zeroshot", 42, (@("--direct", "off") + $P)),
  # Gated DeltaNet memory (ICLR 2025) instead of the mLSTM
  @("ftp_gdn_official",       "official", 42, (@("--layout", "gg") + $P)),
  @("ftp_gdn_zeroshot",       "zeroshot", 42, (@("--layout", "gg") + $P)),
  @("ftp_gdn_official_s1",    "official", 1,  (@("--layout", "gg") + $P)),
  @("ftp_nomem_official_s1",  "official", 1,  (@("--direct", "only") + $P)),
  @("ftp_gdn_zeroshot_s1",    "zeroshot", 1,  (@("--layout", "gg") + $P)),
  @("ftp_gdn_official_s2",    "official", 2,  (@("--layout", "gg") + $P)),
  @("ftp_nomem_official_s2",  "official", 2,  (@("--direct", "only") + $P)),
  @("ftp_gdn_zeroshot_s2",    "zeroshot", 2,  (@("--layout", "gg") + $P)),
  # final model (GDN memory): leakage-free split and BABSA
  @("ftp_gdn_grouped",        "grouped",  42, (@("--layout", "gg") + $P)),
  @("ftp_gdn_grouped_s1",     "grouped",  1,  (@("--layout", "gg") + $P)),
  @("ftp_gdn_grouped_s2",     "grouped",  2,  (@("--layout", "gg") + $P)),
  @("ftp_gdn_babsa_banglabert",    "grouped", 42, (@("--layout", "gg", "--data", "babsa", "--model", "csebuetnlp/banglabert") + $P)),
  @("ftp_gdn_babsa_banglabert_s1", "grouped", 1,  (@("--layout", "gg", "--data", "babsa", "--model", "csebuetnlp/banglabert") + $P)),
  @("ftp_gdn_babsa_banglabert_s2", "grouped", 2,  (@("--layout", "gg", "--data", "babsa", "--model", "csebuetnlp/banglabert") + $P)),
  @("ftp_gdn_babsa_xlmr",          "grouped", 42, (@("--layout", "gg", "--data", "babsa") + $P)),
  # TMM-style readout (marker token before each name instead of the mean over the name span)
  @("ftp_gdn_marker_official", "official", 42, (@("--layout", "gg", "--aspect-read", "marker") + $P)),
  @("ftp_gdn_marker_zeroshot", "zeroshot", 42, (@("--layout", "gg", "--aspect-read", "marker") + $P)),
  # extra seeds for the no-prompt ablation
  @("ft_official_s1",         "official", 1,  @()),
  @("ft_zeroshot_s1",         "zeroshot", 1,  @()),
  @("ft_nomem_official_s1",   "official", 1,  @("--direct", "only")),
  @("ft_nomem_zeroshot_s1",   "zeroshot", 1,  @("--direct", "only")),
  # memory-contribution pilots (seed 42): path dropout, direct scorer on memory outputs, longer training
  @("ftp_gdn_pd_official",     "official", 42, (@("--layout", "gg", "--path-drop", "0.3") + $P)),
  @("ftp_gdn_pd_zeroshot",     "zeroshot", 42, (@("--layout", "gg", "--path-drop", "0.3") + $P)),
  @("ftp_gdn_dmem_official",   "official", 42, (@("--layout", "gg", "--direct-src", "mem") + $P)),
  @("ftp_gdn_dmem_zeroshot",   "zeroshot", 42, (@("--layout", "gg", "--direct-src", "mem") + $P)),
  @("ftp_gdn_e15_official",    "official", 42, (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_nomem_e15_official",  "official", 42, (@("--direct", "only", "--epochs", "15", "--patience", "4") + $P)),
  # 15-epoch retraining: final model (3 splits) and no-memory ablation (official, zero-shot), 3 seeds
  @("ftp_gdn_e15_official_s1",   "official", 1,  (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_nomem_e15_official_s1", "official", 1,  (@("--direct", "only", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_gdn_e15_zeroshot",      "zeroshot", 42, (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_nomem_e15_zeroshot",    "zeroshot", 42, (@("--direct", "only", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_gdn_e15_official_s2",   "official", 2,  (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_nomem_e15_official_s2", "official", 2,  (@("--direct", "only", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_gdn_e15_zeroshot_s1",   "zeroshot", 1,  (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_nomem_e15_zeroshot_s1", "zeroshot", 1,  (@("--direct", "only", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_gdn_e15_zeroshot_s2",   "zeroshot", 2,  (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_nomem_e15_zeroshot_s2", "zeroshot", 2,  (@("--direct", "only", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_gdn_e15_grouped",       "grouped",  42, (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_gdn_e15_grouped_s1",    "grouped",  1,  (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  @("ftp_gdn_e15_grouped_s2",    "grouped",  2,  (@("--layout", "gg", "--epochs", "15", "--patience", "4") + $P)),
  # retrieval-augmented multi-script aspect prompting (no memory): full KB, and English-synonym-only KB
#OBSOLETE   @("kb_official",     "official", 42, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json") + $P)),
  @("kb_zeroshot",     "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json") + $P)),
  @("kben_zeroshot",   "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--kb-scripts", "en") + $P)),
  # control: same extra tokens, but each aspect gets another aspect's forms
  @("kbshuf_zeroshot", "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--kb-shuffle") + $P)),
  # official split with a smaller micro-batch (the first try failed with a CUDA error on long prompts)
#OBSOLETE   @("kb_official_b",   "official", 42, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--max-tokens", "1536") + $P)),
  # script-only knowledge (no English synonyms): does Bangla/Banglish alone help?
  @("kbbnbl_zeroshot", "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--kb-scripts", "bn,bl") + $P)),
  # official split again: kb_official_b ran out of GPU memory at epoch 7
  @("kb_official_c",   "official", 42, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--max-tokens", "1024") + $P)),
  # seeds 1-2: full KB (zero-shot, official), English-only KB and shuffled control (zero-shot)
  @("kb_zeroshot_s1",     "zeroshot", 1, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--max-tokens", "1024") + $P)),
  @("kbshuf_zeroshot_s1", "zeroshot", 1, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--kb-shuffle", "--max-tokens", "1024") + $P)),
  @("kben_zeroshot_s1",   "zeroshot", 1, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--kb-scripts", "en", "--max-tokens", "1024") + $P)),
  @("kb_official_s1",     "official", 1, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--max-tokens", "1024") + $P)),
  @("kb_zeroshot_s2",     "zeroshot", 2, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--max-tokens", "1024") + $P)),
  @("kbshuf_zeroshot_s2", "zeroshot", 2, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--kb-shuffle", "--max-tokens", "1024") + $P)),
  @("kben_zeroshot_s2",   "zeroshot", 2, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--kb-scripts", "en", "--max-tokens", "1024") + $P)),
  @("kb_official_s2",     "official", 2, (@("--direct", "only", "--aspect-kb", "data\aspect_kb.json", "--max-tokens", "1024") + $P)),
  # RAMP-R: automatic cross-script retrieval (build_rag_kb.py) + evidence-gated read-out
  @("rag_zeroshot",       "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data/rag_kb_zeroshot.json", "--kb-k", "4", "--form-gate", "--max-tokens", "1024") + $P)),
  @("rag2_zeroshot",       "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data/rag2_kb_zeroshot.json", "--kb-k", "4", "--form-gate", "--max-tokens", "1024") + $P)),
  @("rag2nogate_zeroshot",       "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data/rag2_kb_zeroshot.json", "--kb-k", "4", "--max-tokens", "1024") + $P)),
  @("kbgate_zeroshot",   "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data/aspect_kb.json", "--kb-k", "99", "--form-gate", "--max-tokens", "1024") + $P)),
  @("rag3_zeroshot",     "zeroshot", 42, (@("--direct", "only", "--aspect-kb", "data/rag2_kb_zeroshot.json", "--kb-k", "4", "--form-gate", "--det-noisy-or", "--alias-drop", "0.3", "0.15", "--max-tokens", "1024") + $P))
#DROPPED (XLM-R only)   @("rag2bb_zeroshot",   "zeroshot", 42, (@("--model", "csebuetnlp/banglabert", "--direct", "only", "--aspect-kb", "data/rag2_kb_zeroshot.json", "--kb-k", "4", "--form-gate", "--max-tokens", "1024") + $P))
)
foreach ($j in $jobs) {
  if ($args.Count -and -not ($args -contains $j[0])) { continue }
  if (Test-Path "runs\$($j[0])\metrics.json") { continue }
  $t0 = Get-Date
  Add-Content runs\queue_log.txt "$(Get-Date -Format HH:mm) start $($j[0])"
  $a = @("scripts\train_ft.py", "--name", $j[0], "--eco-split", $j[1], "--seed", $j[2]) + $j[3]
  & $py @a > "runs\$($j[0]).out" 2> "runs\$($j[0]).err"
  Add-Content runs\queue_log.txt "$(Get-Date -Format HH:mm) end   $($j[0]) rc=$LASTEXITCODE $([math]::Round(((Get-Date)-$t0).TotalMinutes,1)) min"
}
