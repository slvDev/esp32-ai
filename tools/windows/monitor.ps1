# Watch the ESP32-S3 serial output directly from Windows.
#
# Reading through WSL2/usbipd reconnects ~6 s after a reset, by which time the
# firmware has already generated most of its output and dropped it (emit() in
# the sketch discards tokens when no host is draining the CDC buffer). Reading
# the COM port from Windows re-enumerates in 1-2 s, fast enough to catch boot.
#
# Requires: the cable in the port silkscreened USB (native USB-Serial-JTAG).
# Run as administrator so the usbipd service can be started if it is stopped.

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $dir 'serial-output.txt'
$VIDPID = '*VID_303A&PID_1001*'

$usbipd = 'usbipd'
if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
  $usbipd = Join-Path $env:ProgramFiles 'usbipd-win\usbipd.exe'
}

Write-Host ""
Write-Host "=== ESP32-S3 serial monitor (direct COM) ==="
Write-Host ""

# The usbipd service must be running for 'detach' to be executed at all.
# Stopping every process named usbipd also stops this service, after which
# detach fails silently and the device stays attached to WSL with no COM port.
$svc = Get-Service -Name usbipd -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -ne 'Running') {
  Write-Host "[1] starting the usbipd service"
  Start-Service usbipd -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2
}

$busid = $null
if (Get-Command $usbipd -ErrorAction SilentlyContinue) {
  foreach ($line in (& $usbipd list 2>&1)) {
    if ("$line" -match '^\s*(\d+-\d+)\s+303a:1001') { $busid = $matches[1] }
  }
}
if ($busid) {
  Write-Host "[2] releasing the board from WSL (busid $busid)"
  & $usbipd detach --busid $busid 2>&1 | Out-Null
  Start-Sleep -Seconds 3
} else {
  Write-Host "[2] board is not attached to WSL"
}

function Find-Com {
  $e = Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
       Where-Object { $_.PNPDeviceID -like $VIDPID -and $_.Name -match 'COM\d+' }
  if ($e) { return [regex]::Match((@($e)[0]).Name, 'COM\d+').Value }
  return $null
}

Write-Host "[3] looking for the board's COM port"
$com = $null
for ($i = 0; $i -lt 20 -and -not $com; $i++) {
  $com = Find-Com
  if (-not $com) { Start-Sleep -Seconds 1 }
}

if (-not $com) {
  Write-Host ""
  Write-Host "Board not found as a COM port."
  Write-Host "Most likely the cable is in the UART connector. Move it to USB."
  Write-Host "COM ports Windows currently sees: $([System.IO.Ports.SerialPort]::GetPortNames() -join ', ')"
  Read-Host "Press Enter to exit"
  exit 1
}

Write-Host "    found on $com"
Write-Host ""
Write-Host "=== listening on $com - press RST on the board now ==="
Write-Host "Generation runs once per boot, so without a reset nothing arrives."
Write-Host ""

$sw = New-Object System.IO.StreamWriter($log, $true)
$sw.AutoFlush = $true
$sw.WriteLine("`n=== monitor started " + (Get-Date -Format 'HH:mm:ss') + " on " + $com + " ===")

$sp = $null
$lastBeat = Get-Date
$sawData = $false

while ($true) {
  try {
    if ($null -eq $sp -or -not $sp.IsOpen) {
      $c = Find-Com
      if (-not $c) { Start-Sleep -Milliseconds 100; continue }
      $sp = New-Object System.IO.Ports.SerialPort -ArgumentList $c, 115200, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
      $sp.DtrEnable = $true
      $sp.RtsEnable = $true
      $sp.ReadTimeout = 200
      $sp.Open()
      $msg = "`n=== connected to $c (" + (Get-Date -Format 'HH:mm:ss') + ") ===`n"
      Write-Host -NoNewline $msg
      $sw.Write($msg)
    }
    $data = $sp.ReadExisting()
    if ($data.Length -gt 0) {
      Write-Host -NoNewline $data
      $sw.Write($data)
      $sawData = $true
      $lastBeat = Get-Date
    } else {
      Start-Sleep -Milliseconds 30
      if (-not $sawData -and ((Get-Date) - $lastBeat).TotalSeconds -ge 12) {
        Write-Host "    ... connected to $com, no data yet. Press RST on the board."
        $lastBeat = Get-Date
      }
    }
  } catch {
    try { if ($sp) { $sp.Close() } } catch {}
    $sp = $null
    Start-Sleep -Milliseconds 100
  }
}
