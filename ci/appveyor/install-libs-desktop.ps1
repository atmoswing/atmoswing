. $PSScriptRoot\libs-common-definitions.ps1

# Options
$TARGET_CPU="x64"
$WX_TARGET_CPU="X64"
$GDAL_WIN64=1

# Setup VS environment
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt
pushd "C:\$PROGRAM_FILES\Microsoft Visual Studio\$VS_VER_YR\Community\VC\Auxiliary\Build"
cmd /c "vcvars64.bat&set" |
foreach {
  if ($_ -match "=") {
    $v = $_.split("="); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
  }
}
popd
Write-Host "`nVisual Studio $VS_VER_YR Command Prompt variables set." -ForegroundColor Yellow

set CL=/MP

. $PSScriptRoot\libs-desktop-install.ps1

Get-ChildItem "$LIB_DIR/include"
Get-ChildItem "$LIB_DIR/lib"
Get-ChildItem "$LIB_DIR/bin"
