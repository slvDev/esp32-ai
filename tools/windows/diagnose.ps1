# Read-only diagnostics. Changes nothing; writes a report next to this script.

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $dir 'diagnostics.txt'

$usbipd = 'usbipd'
if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
  $usbipd = Join-Path $env:ProgramFiles 'usbipd-win\usbipd.exe'
}

"=== esp32-ai Windows diagnostics ===" | Out-File $log

"`n--- usbipd list ---" | Out-File $log -Append
(& $usbipd list 2>&1 | Out-String) | Out-File $log -Append

"`n--- COM ports visible to .NET ---" | Out-File $log -Append
([System.IO.Ports.SerialPort]::GetPortNames() -join ', ') | Out-File $log -Append

"`n--- Espressif devices (VID_303A) ---" | Out-File $log -Append
Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
  Where-Object { $_.PNPDeviceID -like '*VID_303A*' } |
  Select-Object Name, Status, Service | Format-List | Out-String | Out-File $log -Append

"`n--- CH343 bridge (VID_1A86) ---" | Out-File $log -Append
Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
  Where-Object { $_.PNPDeviceID -like '*VID_1A86*' } |
  Select-Object Name, Status, Service | Format-List | Out-String | Out-File $log -Append

"`n--- serial devices seen by WSL2 ---" | Out-File $log -Append
(wsl -u root -- bash -lc "ls -l /dev/ttyACM* /dev/ttyUSB* 2>&1" 2>&1 | Out-String) | Out-File $log -Append

Get-Content $log
Write-Host "`nSaved to $log"
Read-Host "Press Enter to exit"
