# Attach the ESP32-S3 to WSL2 so scripts/deploy.sh can flash it.
#
# Requires: the cable in the port silkscreened UART (CH343 bridge), and
# administrator rights (usbipd bind).

$usbipd = 'usbipd'
if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
  $usbipd = Join-Path $env:ProgramFiles 'usbipd-win\usbipd.exe'
}

Write-Host ""
Write-Host "=== attach the board to WSL2 ==="
Write-Host ""

$svc = Get-Service -Name usbipd -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -ne 'Running') {
  Start-Service usbipd -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2
}

# 1a86:55d3 is the CH343 bridge on the UART connector, which is what deploy.sh
# talks to. 303a:1001 (native USB) is accepted as a fallback.
$busid = $null
foreach ($line in (& $usbipd list 2>&1)) {
  if ("$line" -match '^\s*(\d+-\d+)\s+1a86:55d3') { $busid = $matches[1]; break }
}
if (-not $busid) {
  foreach ($line in (& $usbipd list 2>&1)) {
    if ("$line" -match '^\s*(\d+-\d+)\s+303a:1001') { $busid = $matches[1]; break }
  }
}

if (-not $busid) {
  Write-Host "No ESP32-S3 found. Current devices:"
  & $usbipd list
  Read-Host "Press Enter to exit"
  exit 1
}

Write-Host "board on busid $busid"
& $usbipd bind   --busid $busid 2>&1 | Out-Null
& $usbipd attach --wsl --busid $busid 2>&1 | Out-Null
Start-Sleep -Seconds 2

Write-Host ""
& $usbipd list
Write-Host ""
Write-Host "In WSL2, check with:  ls /dev/ttyACM*"
Write-Host "Then flash with:      PORT=/dev/ttyACM0 scripts/deploy.sh tinystories"
Write-Host ""
Read-Host "Press Enter to exit"
