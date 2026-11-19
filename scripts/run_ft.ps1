# QuAM-FT queue (sequential). Skips runs that already have metrics.json. Log: runs/queue_log.txt
# Usage: powershell -File scripts\run_ft.ps1 [job names...]
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "C:\Users\USER\.venvs\quam-absa\Scripts\python.exe"
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
  @("ft_grouped",      "grouped",  42, @())
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
