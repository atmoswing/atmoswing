cd C:\projects\atmoswing\bin
cpack -C release -G ZIP
cpack -C release -G NSIS
cpack -C release -G WIX
cd C:\projects\atmoswing
copy bin\*win64.exe .\
copy bin\*win64.zip .\
copy bin\*win64.msi .\