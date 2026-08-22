# Windows helper scripts

PowerShell tooling for running esp32-ai from Windows with WSL2. See
[`docs/WINDOWS.md`](../../docs/WINDOWS.md) for the full guide
([español](../../docs/WINDOWS.es.md)).

Each `.bat` requests administrator rights and then runs the matching `.ps1`.
Nothing here is required by the project — `usbipd` and any serial terminal do
the same job by hand.

| Script | What it does |
| ------ | ------------ |
| `diagnose.bat` | Read-only. Reports what Windows and WSL2 currently see. |
| `attach-to-wsl.bat` | Binds and attaches the board to WSL2, for `scripts/deploy.sh`. Use the **UART** connector. |
| `monitor.bat` | Releases the board back to Windows and streams its serial output, reconnecting across resets. Use the **USB** connector. |

`monitor.bat` also appends everything to `serial-output.txt` beside the script.

**Ports are not interchangeable:** flash through `UART` (CH343, `1a86:55d3`),
watch through `USB` (native USB-Serial-JTAG, `303a:1001`). The guide explains why.
