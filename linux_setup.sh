

sudo apt install libjpeg-turbo-progs -y

uv tool install -e .

mkdir -p ~/.local/share/icons/hicolor/128x128/apps/
cp resources/nsee_icon.png ~/.local/share/icons/hicolor/128x128/apps/nsee.png


cat > ~/.local/share/applications/nsee.desktop << EOF
[Desktop Entry]
Name=NSEE
Comment=Image tool
Exec=nsee %F
Icon=nsee
Terminal=false
Type=Application
Categories=Graphics;
MimeType=image/png;image/jpeg;image/tiff;
EOF

chmod +x ~/.local/share/applications/nsee.desktop

xdg-mime default nsee.desktop image/png
xdg-mime default nsee.desktop image/jpeg
xdg-mime default nsee.desktop image/tiff

update-desktop-database ~/.local/share/applications


desktop-file-validate ~/.local/share/applications/nsee.desktop
xdg-mime query default image/png
