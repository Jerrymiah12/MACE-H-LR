# Hold a Windows execution-state request while the paired training job runs.
# The calling Bash process terminates this helper in its EXIT trap, so this
# does not make a persistent change to the active Windows power plan.

$ErrorActionPreference = "Stop"

$source = @"
using System;
using System.Runtime.InteropServices;

public static class MacehKeepAwake
{
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

Add-Type -TypeDefinition $source

$esContinuous = [Convert]::ToUInt32("80000000", 16)
$esSystemRequired = [uint32]0x00000001
$esAwayModeRequired = [uint32]0x00000040
$activeFlags = $esContinuous -bor $esSystemRequired -bor $esAwayModeRequired

$result = [MacehKeepAwake]::SetThreadExecutionState($activeFlags)
if ($result -eq 0) {
    throw "SetThreadExecutionState failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
}

Write-Output "Windows keep-awake request active (display may turn off)."
try {
    while ($true) {
        # Refresh from the active PowerShell thread.  This also protects
        # against hosts that do not retain a cross-WSL execution-state request
        # reliably for the lifetime of a long Start-Sleep call.
        Start-Sleep -Seconds 20
        $result = [MacehKeepAwake]::SetThreadExecutionState($activeFlags)
        if ($result -eq 0) {
            throw "SetThreadExecutionState refresh failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
        }
    }
}
finally {
    [void][MacehKeepAwake]::SetThreadExecutionState($esContinuous)
}
