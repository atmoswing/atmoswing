# Install WIX
if ($APPVEYOR) {
  Write-Host "`nInstalling WIX" -ForegroundColor Yellow
  cd $TMP_DIR
  $WIX_URL="https://github.com/wixtoolset/wix3/releases/download/wix3111rtm/wix311-binaries.zip"
  if ($APPVEYOR) {
    appveyor DownloadFile $WIX_URL -FileName wix.zip > $null
  } else {
    Invoke-WebRequest -Uri $WIX_URL -OutFile wix.zip
  }
  7z x wix.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\wix311-binaries\*" "$WIX_DIR"
  $env:Path += ";$WIX_DIR\bin"
}
