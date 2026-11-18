# Baseline queue (sequential). Skips runs that already have metrics.json. Log: runs/queue_log.txt
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "C:\Users\USER\.venvs\quam-absa\Scripts\python.exe"
$FT = @("--freeze-embeddings", "--grad-ckpt")                                  # XLM-R full fine-tune (fits 4 GB)
$FR = @("--freeze-encoder", "--lr", "1e-3", "--epochs", "15", "--patience", "4") # frozen encoder + MLP head
$jobs = @(
  @("bl_xlmr_frozen_official",    "xlm-roberta-base",      "official", 42, $FR),
  @("bl_xlmr_frozen_grouped",     "xlm-roberta-base",      "grouped",  42, $FR),
  @("bl_xlmr_official",           "xlm-roberta-base",      "official", 42, $FT),
  @("bl_xlmr_grouped",            "xlm-roberta-base",      "grouped",  42, $FT),
  @("bl_xlmr_frozen_official_s1", "xlm-roberta-base",      "official", 1,  $FR),
  @("bl_xlmr_frozen_official_s2", "xlm-roberta-base",      "official", 2,  $FR),
  @("bl_xlmr_frozen_grouped_s1",  "xlm-roberta-base",      "grouped",  1,  $FR),
  @("bl_xlmr_frozen_grouped_s2",  "xlm-roberta-base",      "grouped",  2,  $FR),
  @("bl_banglabert_frozen_official", "csebuetnlp/banglabert", "official", 42, $FR),
  @("bl_banglabert_frozen_grouped",  "csebuetnlp/banglabert", "grouped",  42, $FR)
)
foreach ($j in $jobs) {
  if ($args.Count -and -not ($args -contains $j[0])) { continue }
  if (Test-Path "runs\$($j[0])\metrics.json") { continue }
  $t0 = Get-Date
  Add-Content runs\queue_log.txt "$(Get-Date -Format HH:mm) start $($j[0])"
  $a = @("scripts\baseline_ft.py", "--name", $j[0], "--model", $j[1], "--eco-split", $j[2], "--seed", $j[3]) + $j[4]
  & $py @a > $null 2> "runs\$($j[0]).err"
  Add-Content runs\queue_log.txt "$(Get-Date -Format HH:mm) end   $($j[0]) rc=$LASTEXITCODE $([math]::Round(((Get-Date)-$t0).TotalMinutes,1)) min"
}
