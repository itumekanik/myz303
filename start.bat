@echo off
rem ------------------------------------------------------------------
rem  SimpleCNN Paneli - baslatici
rem  Cift tiklayin: gerekirse sanal ortami kurar, sonra paneli acar.
rem  Ornek:  start.bat --port 8080 --no-browser
rem ------------------------------------------------------------------
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" goto create
rem Onceki kurulum yarida kaldiysa (.venv var ama paketler eksik) tamamlanir
"%VENV_PY%" -c "import importlib.util as u, sys; sys.exit(any(u.find_spec(m) is None for m in ('torch', 'torchvision', 'matplotlib')))" >nul 2>nul
if not errorlevel 1 goto run
echo Sanal ortamdaki kurulum eksik, tamamlaniyor...
goto install

:create
echo Sanal ortam (.venv) bulunamadi, kuruluyor. Bu islem ilk seferde birkac dakika surebilir...
rem PyTorch'un destekledigi bir Python surumu tercih edilir (3.13 > 3.12 > varsayilan)
py -3.13 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || python -m venv .venv
if not exist "%VENV_PY%" (
    echo [HATA] Sanal ortam olusturulamadi. Python kurulu mu?
    goto fail
)
:install
rem --isolated: bilgisayardaki pip.ini ayarlarini (ulasilamayan ek kaynaklar) yok sayar
"%VENV_PY%" -m pip --isolated install --upgrade pip
rem NVIDIA GPU varsa CUDA paketi (yaklasik 2,5 GB), yoksa cok daha kucuk CPU paketi
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
"%VENV_PY%" -m pip --isolated install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [HATA] PyTorch kurulamadi. Internet baglantinizi kontrol edin.
    goto fail
)
"%VENV_PY%" -m pip --isolated install matplotlib
if errorlevel 1 goto fail

:run
rem Panel, web\ klasorunden derlenen sayfalari (web\dist) sunar; ilk seferde derlenir
if exist "web\dist\index.html" goto serve
echo Sayfalar henuz derlenmemis (web\dist). Derleniyor, ilk seferde bir iki dakika surer...
where npm >nul 2>nul
if errorlevel 1 (
    echo [HATA] Node.js bulunamadi. Ya https://nodejs.org adresinden Node.js 22.18 veya ustunu kurun,
    echo ya da https://itumekanik.github.io/myz303/ai-learning-path.zip dosyasini indirip
    echo icindekileri dogrudan web\dist klasorune acin: index.html web\dist icinde olmali.
    goto fail
)
pushd web
call npm ci
if errorlevel 1 goto buildfail
call npm run build
if errorlevel 1 goto buildfail
popd

:serve
echo SimpleCNN paneli baslatiliyor... Kapatmak icin bu pencerede Ctrl+C tuslarina basin.
echo.
"%VENV_PY%" dashboard.py %*
if errorlevel 1 goto fail
exit /b 0

:buildfail
popd
echo [HATA] Sayfalar derlenemedi.

:fail
echo.
echo Panel baslatilamadi. Yukaridaki mesaji kontrol edin.
pause
exit /b 1
